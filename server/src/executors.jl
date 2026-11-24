module Executors

using Dates
using ..Jobs

export AbstractJobExecutor, LocalCLIExecutor
export submit, poll, result_path, kill_job!, run_dispatcher!, job_thread_budget

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
    submit(executor::LocalCLIExecutor, cli_args::Vector{String}; threads::Int=1) -> String

Submits a job by launching a CLI subprocess.

# Arguments
- `executor::LocalCLIExecutor`: The execution backend instance.
- `cli_args::Vector{String}`: Command line arguments to pass to the subprocess.

# Returns
- `String`: A 'handle' (in this case, the PID string) used to query or kill the process.
"""
function submit(executor::LocalCLIExecutor, cli_args::Vector{String}; threads::Int=1)
    # `julia --project=@. -m SimilaritySearchServer ...` requires the package to be
    # declared as an app entry point via `@main` (Julia 1.12's `-m` mechanism), which this
    # dev'd (not `Pkg.add`-installed-as-app) package isn't -- it errors immediately with
    # "main not declared as entry point". Spawn the exact same CLI script a human would run
    # offline instead (`src/apps/similarity-search.jl`), exactly like the CLI tests do.
    cmd = _job_command(cli_args, threads)

    # Launch the process in the background (wait=false). `getpid` raises when the process has
    # already exited and its handle is closed, which can happen when the event loop runs on
    # another thread. Such a process still has an exit code, so it gets a handle of its own.
    proc = run(cmd, wait=false)
    handle = try
        string(getpid(proc))
    catch
        "exited-" * string(objectid(proc); base=16)
    end

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
    exit_code(executor, handle) -> Union{Nothing,Int}

The exit status of the finished process behind `handle`, or `nothing` when the executor does
not know it. The CLI's exit codes name the category of an engine error (`cli_exit_code`), so
the dispatcher records this status in the record of a failed job.
"""
exit_code(::AbstractJobExecutor, ::String) = nothing

function exit_code(executor::LocalCLIExecutor, handle::String)
    proc = get(executor.running_processes, handle, nothing)
    (proc === nothing || !process_exited(proc)) && return nothing
    return Int(proc.exitcode)
end

"""
    result_path(executor::LocalCLIExecutor, handle::String) -> String

Returns the theoretical path where the worker will leave its final results or logs.

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
    job_thread_budget(total_threads::Int, batch_pct::Real; max_jobs::Int=4) -> (max_concurrent, threads_per_job)

How much of the machine job execution may use, from the share of threads reserved for it
(`[resources] batch_threads_pct`, CHANGELOG.md §2).

A job runs as its own process, so what bounds it is a number of processes and a number of
threads each. The reserved share is the product: at most `max_concurrent` jobs at once, each
with `threads_per_job`, and `max_concurrent * threads_per_job` never exceeds the share.

`max_jobs` is the ceiling on concurrency, so that a large machine runs a few jobs with
several threads each rather than many jobs with one thread each; indexing and the operations
over a whole dataset are parallel inside one process, and splitting them across more
processes does not make them faster.

Both results are at least 1: a share that rounds to zero still executes jobs, one at a time
on one thread, because a server that accepts a job and never runs it is worse than a slow
one.

```julia
job_thread_budget(64, 20)   # (4, 3)  -- 12 threads of the 12.8 reserved
job_thread_budget(8, 20)    # (1, 1)
job_thread_budget(32, 50)   # (4, 4)
```
"""
function job_thread_budget(total_threads::Int, batch_pct::Real; max_jobs::Int=4)
    reserved = max(1, floor(Int, max(total_threads, 1) * clamp(batch_pct, 0, 100) / 100))
    max_concurrent = max(1, min(max_jobs, reserved))
    threads_per_job = max(1, reserved ÷ max_concurrent)
    return (max_concurrent, threads_per_job)
end

"""
    _job_command(cli_args::Vector{String}, threads::Int) -> Cmd

The command a job runs, with its thread budget. `Base.julia_cmd()` carries the current
process's `-C`/`-J`/`-g` options but not its `--threads`, so without this a job subprocess
starts with one thread regardless of how the server was started.
"""
function _job_command(cli_args::Vector{String}, threads::Int)
    julia_exe = Base.julia_cmd()
    cli_script = joinpath(@__DIR__, "apps", "similarity-search.jl")
    project_dir = joinpath(@__DIR__, "..")
    cmd = `$julia_exe --project=$project_dir $cli_script $cli_args`
    return addenv(cmd, "JULIA_NUM_THREADS" => string(max(1, threads)))
end

"""
    run_dispatcher!(job_mgr::Jobs.JobManager, executor::AbstractJobExecutor; max_concurrent::Int=4, threads_per_job::Int=1, poll_interval::Real=0.5)

Background loop (meant to be run inside `@async`) that owns every `queued/` -> `running/`
-> `completed/|failed/` job transition (CHANGELOG.md §5.5). The HTTP layer only ever calls
`Jobs.create_job!` to enqueue; this loop is what actually dispatches and finishes jobs.
Crash recovery for jobs orphaned in `running/` by a previous process must be done by the
caller (via `Jobs.requeue_stale_running!`) before starting this loop.

An error while one job is started or finished does not stop the loop. The error is logged,
the job goes to `failed/` with the error in its record, and the loop continues with the next
job. Until 2026-11-20 such an error ended the task without a message, and every job submitted
after it stayed in `queued/`.
"""
function run_dispatcher!(job_mgr::Jobs.JobManager, executor::AbstractJobExecutor;
                         max_concurrent::Int=4, threads_per_job::Int=1, poll_interval::Real=0.5)
    active = Dict{String, String}() # job_id => executor handle

    while true
        # Start new jobs while there is room in the concurrency window.
        while length(active) < max_concurrent
            queued_ids = Jobs.list_jobs(job_mgr, Jobs.Queued)
            isempty(queued_ids) && break

            id = first(queued_ids)
            try
                _dispatch_one!(job_mgr, executor, active, id, threads_per_job)
            catch e
                @error "the job dispatcher could not start a job" id exception=(e, catch_backtrace())
                _fail_dispatched_job!(job_mgr, id, "the dispatcher could not start the job: " * sprint(showerror, e))
                delete!(active, id)
            end
        end

        # Poll running jobs and finalize the ones that finished.
        for (id, handle) in collect(active)
            try
                _finish_if_done!(job_mgr, executor, active, id, handle)
            catch e
                @error "the job dispatcher could not finish a job" id exception=(e, catch_backtrace())
                _fail_dispatched_job!(job_mgr, id, "the dispatcher could not finish the job: " * sprint(showerror, e))
                delete!(active, id)
            end
        end

        sleep(poll_interval)
    end
end

# Moves one queued job to `running/` and starts its process. A job that left `queued/` after
# it was listed (cancelled, blocked or started by someone else) is skipped.
function _dispatch_one!(job_mgr, executor, active, id, threads_per_job)
    state, record = Jobs.get_job(job_mgr, id)
    state === Jobs.Queued || return
    Jobs.update_job_state!(job_mgr, id, Jobs.Queued, Jobs.Running)
    # TOML reads an empty array as `Any[]`, and `String.(Any[])` is still a `Vector{Any}`.
    command = String[string(x) for x in get(record, "command", String[])]
    handle = submit(executor, command; threads=threads_per_job)
    active[id] = handle
    Jobs.update_job_content!(job_mgr, Jobs.Running, id, Dict(
        "executor_handle" => handle,
        "started_at" => string(now(UTC)),
    ))
end

function _finish_if_done!(job_mgr, executor, active, id, handle)
    status = poll(executor, handle)
    if status == "completed"
        Jobs.finish_job!(job_mgr, id, true)
        delete!(active, id)
    elseif status == "failed"
        code = exit_code(executor, handle)
        Jobs.finish_job!(job_mgr, id, false; exit_code=code,
                         error=code === nothing ? "job process exited with a non-zero status" :
                                                  "job process exited with status $code")
        delete!(active, id)
    elseif status == "unknown"
        # The executor lost track of the handle -- treat as failed rather than
        # leaving the job stuck in running/ forever.
        Jobs.finish_job!(job_mgr, id, false; error="executor lost track of the job handle")
        delete!(active, id)
    end
end

# Records a job the dispatcher could not handle as failed, from whichever state it is in.
# A job that is already finished, or that no longer exists, is left as it is.
function _fail_dispatched_job!(job_mgr, id, message)
    try
        state, _ = Jobs.get_job(job_mgr, id)
        if state === Jobs.Queued
            Jobs.update_job_state!(job_mgr, id, Jobs.Queued, Jobs.Running)
            state = Jobs.Running
        end
        state === Jobs.Running && Jobs.finish_job!(job_mgr, id, false; error=message)
    catch e
        @error "the job dispatcher could not record a failed job" id exception=(e, catch_backtrace())
    end
end

end # module
