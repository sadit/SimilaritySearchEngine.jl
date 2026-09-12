module Jobs

using TOML
using Dates
using ..Project # To use generate_id()

export JobManager, JobRecord, JobState, Queued, Running, Blocked, Completed, Failed
export init_job_manager, create_job!, update_job_state!, get_job
export list_jobs, update_job_content!, finish_job!, requeue_stale_running!, purge_finished!, cancel_job!

@enum JobState begin
    Queued
    Running
    Blocked
    Completed
    Failed
end

"""
    JobRecord

In-memory structure representing a Job's metadata.

# Fields
- `id::String`: Unique job identifier.
- `kind::String`: Job kind (e.g., searchbatch, build).
- `command::Vector{String}`: The command arguments for the job.
- `created_at::String`: Timestamp of creation.
- `progress::Float64`: Job progress (0.0 to 1.0).
- `executor_handle::Union{String, Nothing}`: OS-level handle assigned by the executor.
"""
struct JobRecord
    id::String
    kind::String
    command::Vector{String}
    created_at::String
    progress::Float64
    executor_handle::Union{String, Nothing}
end

"""
    JobManager

Job spooling manager.
Maintains the root working directory.

# Fields
- `workdir::String`: Root path of the spooling directories.
"""
mutable struct JobManager
    workdir::String
end

"""
    init_job_manager(workdir::String) -> JobManager

Initializes the atomic spooling directories for jobs.

# Arguments
- `workdir::String`: Base directory path to initialize spooling.

# Returns
- `JobManager`: Initialized manager instance.
"""
function init_job_manager(workdir::String)
    for state in ["queued", "running", "blocked", "completed", "failed"]
        mkpath(joinpath(workdir, "jobs", state))
    end
    # Directory for paginated cursors
    for state in ["open", "exhausted", "expired"]
        mkpath(joinpath(workdir, "cursors", state))
    end
    return JobManager(workdir)
end

function _state_dir(manager::JobManager, state::JobState)
    state_str = lowercase(string(state))
    return joinpath(manager.workdir, "jobs", state_str)
end

function _job_path(manager::JobManager, state::JobState, id::String)
    return joinpath(_state_dir(manager, state), "$(id).job.toml")
end

"""
    create_job!(manager::JobManager, kind::String, command::Vector{String}; id::String=generate_id(), extra::AbstractDict=Dict()) -> String

Creates a job in Queued state by writing it to a temporary file
and atomically renaming it to its destination.

# Arguments
- `manager::JobManager`: The job manager instance.
- `kind::String`: The kind of job being created.
- `command::Vector{String}`: Command arguments for the executor.
- `id::String`: Job id to use; defaults to a fresh generated one. Callers that need to
  know the id before creation (e.g. to embed it in an output file path) can pass their
  own -- see `Server.handle_submit_job`.
- `extra::AbstractDict`: Extra fields merged into the record at creation time (e.g.
  `"result_ref"`). Must be set here, in the same write as the rest of the record, rather
  than via a follow-up `update_job_content!` call -- the background dispatcher
  (`Executors.run_dispatcher!`) can move a job from `queued/` to `running/` the instant
  it's written, and a separate post-creation update would race it (the job may no longer
  be in `queued/` by the time the update runs, and `update_job_content!` errors if it
  can't find the record in the state it expects).

# Returns
- `String`: The job ID (the generated one, or `id` if explicitly given).
"""
function create_job!(manager::JobManager, kind::String, command::Vector{String}; id::String=generate_id(), extra::AbstractDict=Dict())
    record = Dict{String, Any}(
        "id" => id,
        "kind" => kind,
        "command" => command,
        "created_at" => string(now(UTC)),
        "progress" => 0.0,
        "executor_handle" => ""
    )
    merge!(record, extra)

    path = _job_path(manager, Queued, id)
    temp_path = path * ".tmp"
    
    open(temp_path, "w") do io
        TOML.print(io, record)
    end
    
    # Atomic rename (POSIX mv)
    mv(temp_path, path, force=true)
    
    return id
end

"""
    update_job_state!(manager::JobManager, id::String, old_state::JobState, new_state::JobState)

Transitions a job between states by atomically moving it between directories.

# Arguments
- `manager::JobManager`: The job manager instance.
- `id::String`: The job ID.
- `old_state::JobState`: Expected current state.
- `new_state::JobState`: State to transition to.
"""
function update_job_state!(manager::JobManager, id::String, old_state::JobState, new_state::JobState)
    old_path = _job_path(manager, old_state, id)
    new_path = _job_path(manager, new_state, id)
    
    if !isfile(old_path)
        error("Job $id was not found in state $old_state")
    end
    
    mv(old_path, new_path, force=true)
end

"""
    get_job(manager::JobManager, id::String) -> Tuple{Union{JobState, Nothing}, Union{Dict, Nothing}}

Searches for a job across all states and returns its (state, parsed_data).

# Arguments
- `manager::JobManager`: The job manager instance.
- `id::String`: The job ID to look up.

# Returns
- `Tuple`: Returns the JobState and a dictionary of parsed TOML data if found; otherwise `(nothing, nothing)`.
"""
function get_job(manager::JobManager, id::String)
    for state in [Queued, Running, Blocked, Completed, Failed]
        path = _job_path(manager, state, id)
        if isfile(path)
            content = read(path, String)
            return state, TOML.parse(content)
        end
    end
    return nothing, nothing
end

"""
    list_jobs(manager::JobManager, state::JobState) -> Vector{String}

Lists the job ids currently sitting in `state`'s spool directory
(used by the dispatch loop to find work in `queued/`).
"""
function list_jobs(manager::JobManager, state::JobState)
    dir = _state_dir(manager, state)
    ids = String[]
    for fname in readdir(dir)
        endswith(fname, ".job.toml") || continue
        push!(ids, fname[1:end-length(".job.toml")])
    end
    return ids
end

"""
    update_job_content!(manager::JobManager, state::JobState, id::String, updates::Dict)

Rewrites a job's record (while it stays in `state`'s directory) with the given field
updates merged in, via write-to-temp-then-rename so a reader never observes a torn record.
Used by the dispatcher to record the real `executor_handle` once `Executors.submit` returns it.
"""
function update_job_content!(manager::JobManager, state::JobState, id::String, updates::Dict)
    path = _job_path(manager, state, id)
    isfile(path) || error("Job $id was not found in state $state")

    record = TOML.parse(read(path, String))
    merge!(record, updates)

    temp_path = path * ".tmp"
    open(temp_path, "w") do io
        TOML.print(io, record)
    end
    mv(temp_path, path, force=true)
    return record
end

"""
    finish_job!(manager::JobManager, id::String, success::Bool; result=nothing, error=nothing)

Moves a job out of `running/` into its terminal state (`completed/` on success, `failed/`
otherwise), recording the result/error payload.
"""
function finish_job!(manager::JobManager, id::String, success::Bool; result=nothing, error=nothing)
    updates = Dict{String, Any}(
        "finished_at" => string(now(UTC)),
        "progress" => 1.0,
    )
    result !== nothing && (updates["result"] = result)
    error !== nothing && (updates["error"] = error)
    update_job_content!(manager, Running, id, updates)
    update_job_state!(manager, id, Running, success ? Completed : Failed)
end

"""
    cancel_job!(manager::JobManager, id::String) -> Symbol

Cancels a job that hasn't started running yet (PLAN.md §5.5's `DELETE
/api/v1/jobs/{job_id}` -- distinct from `kill`, which only ever applies to an already-
`running` job consuming the reserved compute pool, §5.5's own explicit "kill vs. DELETE"
note). Moves `id` out of `queued/`/`blocked/` straight into `failed/` with
`error="cancelled by operator"` -- the same "terminal state + descriptive error" shape
`finish_job!` already uses for a killed job.

Returns `:cancelled` on success, `:not_cancellable` if `id` is already `running`/
`completed`/`failed` (the caller should reject with `409` -- a running job needs `kill`
instead, not a silent no-op or a fallback to killing it), or `:not_found` if `id` doesn't
exist at all.
"""
function cancel_job!(manager::JobManager, id::String)
    state, _ = get_job(manager, id)
    state === nothing && return :not_found
    state in (Queued, Blocked) || return :not_cancellable

    updates = Dict{String, Any}(
        "finished_at" => string(now(UTC)),
        "progress" => 1.0,
        "error" => "cancelled by operator",
    )
    update_job_content!(manager, state, id, updates)
    update_job_state!(manager, id, state, Failed)
    return :cancelled
end

"""
    requeue_stale_running!(manager::JobManager) -> Vector{String}

Crash-recovery pass: any job left in `running/` when the server starts (because a
previous `serve` process died mid-dispatch) is re-queued so the dispatcher picks it
back up. Returns the ids that were requeued.
"""
function requeue_stale_running!(manager::JobManager)
    ids = list_jobs(manager, Running)
    for id in ids
        update_job_state!(manager, id, Running, Queued)
    end
    return ids
end

"""
    purge_finished!(manager::JobManager; retention_seconds::Real=86400) -> Vector{String}

The actual disk-reclaim half of `similarity-search-ctl jobs gc` (PLAN.md §5.5's warning
that `completed`/`failed` grow unboundedly otherwise, since nothing else ever removes a
finished job's file). Deletes every `completed/`/`failed/` job record whose `finished_at`
is older than `retention_seconds` ago, along with its `result_ref` file if it has one and
it still exists (a purged job's result is unreachable via `GET /api/v1/jobs/{id}/result`
anyway, since the record itself is gone). Returns the ids removed.
"""
function purge_finished!(manager::JobManager; retention_seconds::Real=86400)
    cutoff = now(UTC) - Second(round(Int, retention_seconds))
    removed = String[]
    for state in (Completed, Failed)
        dir = _state_dir(manager, state)
        for fname in readdir(dir)
            endswith(fname, ".job.toml") || continue
            path = joinpath(dir, fname)
            record = TOML.parse(read(path, String))
            finished_at = get(record, "finished_at", nothing)
            finished_at === nothing && continue
            DateTime(finished_at) < cutoff || continue

            result_ref = get(record, "result_ref", nothing)
            (result_ref isa AbstractString) && isfile(result_ref) && rm(result_ref)
            rm(path)
            push!(removed, fname[1:end-length(".job.toml")])
        end
    end
    return removed
end

end # module
