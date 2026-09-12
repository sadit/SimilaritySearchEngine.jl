module Executors

using Dates
using ..Jobs

export AbstractJobExecutor, LocalCLIExecutor
export submit, poll, result_path, kill_job!, run_dispatcher!

"""
    AbstractJobExecutor

Main abstraction for any Job execution backend.
"""
abstract type AbstractJobExecutor end

"""
    LocalCLIExecutor <: AbstractJobExecutor

Local execution backend. Launches child CLI processes to do heavy lifting
without blocking the server or the RocksDB database.

# Fields
- `running_processes::Dict{String, Base.Process}`: A mapping of handles (PIDs) to running processes.
"""
mutable struct LocalCLIExecutor <: AbstractJobExecutor
    running_processes::Dict{String, Base.Process}
    
    LocalCLIExecutor() = new(Dict{String, Base.Process}())
end

"""
    submit(executor::LocalCLIExecutor, cli_args::Vector{String}) -> String

Submits a job by launching a CLI subprocess.

# Arguments
- `executor::LocalCLIExecutor`: The execution backend instance.
- `cli_args::Vector{String}`: Command line arguments to pass to the subprocess.

# Returns
- `String`: A 'handle' (in this case, the PID string) used to query or kill the process.
"""
function submit(executor::LocalCLIExecutor, cli_args::Vector{String})
    # `julia --project=@. -m SimilaritySearchServer ...` requires the package to be
    # declared as an app entry point via `@main` (Julia 1.12's `-m` mechanism), which this
    # dev'd (not `Pkg.add`-installed-as-app) package isn't -- it errors immediately with
    # "main not declared as entry point". Spawn the exact same CLI script a human would run
    # offline instead (`src/apps/similarity-search.jl`), exactly like the CLI tests do.
    julia_exe = Base.julia_cmd()
    cli_script = joinpath(@__DIR__, "apps", "similarity-search.jl")
    project_dir = joinpath(@__DIR__, "..")

    cmd = `$julia_exe --project=$project_dir $cli_script $cli_args`

    # Launch the process in the background (wait=false)
    proc = run(cmd, wait=false)
    handle = string(getpid(proc))

    executor.running_processes[handle] = proc
    return handle
end

"""
    poll(executor::LocalCLIExecutor, handle::String) -> String

Queries the OS-level liveness state of the process.
Note: Logical progress is read from the .job.toml file, but this verifies if the worker is still alive.

# Arguments
- `executor::LocalCLIExecutor`: The execution backend instance.
- `handle::String`: The OS-level handle (PID) returned by `submit`.

# Returns
- `String`: The current status (`"running"`, `"completed"`, `"failed"`, `"unknown"`).
"""
function poll(executor::LocalCLIExecutor, handle::String)
    if !haskey(executor.running_processes, handle)
        return "unknown"
    end
    
    proc = executor.running_processes[handle]
    if process_running(proc)
        return "running"
    elseif process_exited(proc)
        return proc.exitcode == 0 ? "completed" : "failed"
    else
        return "unknown"
    end
end

"""
    result_path(executor::LocalCLIExecutor, handle::String) -> String

Returns the theoretical path where the worker will leave its final results (cursors or logs).

# Arguments
- `executor::LocalCLIExecutor`: The execution backend instance.
- `handle::String`: The OS-level handle (PID).

# Returns
- `String`: The expected path for results.
"""
function result_path(executor::LocalCLIExecutor, handle::String)
    # The CLI writes its result directly to the state files managed by Jobs.jl.
    # This function is kept in case a remote backend (e.g. Slurm) needs to copy the result back.
    return handle # Placeholder, ideally returns the path to the spool.
end

"""
    kill_job!(executor::LocalCLIExecutor, handle::String)

Forces the process to stop if it is running (SIGTERM).

# Arguments
- `executor::LocalCLIExecutor`: The execution backend instance.
- `handle::String`: The OS-level handle (PID).
"""
function kill_job!(executor::LocalCLIExecutor, handle::String)
    if haskey(executor.running_processes, handle)
        proc = executor.running_processes[handle]
        if process_running(proc)
            kill(proc) # Sends termination signal to the process
        end
        delete!(executor.running_processes, handle)
    end
end

"""
    run_dispatcher!(job_mgr::Jobs.JobManager, executor::AbstractJobExecutor; max_concurrent::Int=4, poll_interval::Real=0.5)

Background loop (meant to be run inside `@async`) that owns every `queued/` -> `running/`
-> `completed/|failed/` job transition (PLAN.md §5.5). The HTTP layer only ever calls
`Jobs.create_job!` to enqueue; this loop is what actually dispatches and finishes jobs.
Crash recovery for jobs orphaned in `running/` by a previous process must be done by the
caller (via `Jobs.requeue_stale_running!`) before starting this loop.
"""
function run_dispatcher!(job_mgr::Jobs.JobManager, executor::AbstractJobExecutor; max_concurrent::Int=4, poll_interval::Real=0.5)
    active = Dict{String, String}() # job_id => executor handle

    while true
        # Start new jobs while there is room in the concurrency window.
        while length(active) < max_concurrent
            queued_ids = Jobs.list_jobs(job_mgr, Jobs.Queued)
            isempty(queued_ids) && break

            id = first(queued_ids)
            _, record = Jobs.get_job(job_mgr, id)
            record === nothing && continue

            Jobs.update_job_state!(job_mgr, id, Jobs.Queued, Jobs.Running)
            command = String.(get(record, "command", String[]))
            handle = submit(executor, command)
            Jobs.update_job_content!(job_mgr, Jobs.Running, id, Dict(
                "executor_handle" => handle,
                "started_at" => string(now(UTC)),
            ))
            active[id] = handle
        end

        # Poll running jobs and finalize the ones that finished.
        for (id, handle) in collect(active)
            status = poll(executor, handle)
            if status == "completed"
                Jobs.finish_job!(job_mgr, id, true)
                delete!(active, id)
            elseif status == "failed"
                Jobs.finish_job!(job_mgr, id, false; error="job process exited with a non-zero status")
                delete!(active, id)
            elseif status == "unknown"
                # The executor lost track of the handle -- treat as failed rather than
                # leaving the job stuck in running/ forever.
                Jobs.finish_job!(job_mgr, id, false; error="executor lost track of the job handle")
                delete!(active, id)
            end
        end

        sleep(poll_interval)
    end
end

end # module
