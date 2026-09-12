module Cursors

using JSON3
using Dates
using ..Project # for generate_id()

export CursorManager, CursorState, Open, Exhausted, Expired
export init_cursor_manager, create_cursor!, get_cursor, poll_cursor!, gc_expired!, purge_expired!

@enum CursorState begin
    Open
    Exhausted
    Expired
end

"""
    CursorManager

Cursor spooling manager (PLAN.md §5.4/§4.6), structurally identical to `Jobs.JobManager`
— an opaque result cursor is a spool record with the same create/poll/expire lifecycle a
Job has, just with three states instead of five and JSON records instead of TOML (JSON
round-trips an arbitrary already-materialized result list — a page of search/allknn/etc.
results — more naturally than TOML does).

# Fields
- `workdir::String`: Root path of the spooling directories.
"""
mutable struct CursorManager
    workdir::String
end

"""
    init_cursor_manager(workdir::String) -> CursorManager

Initializes the atomic spooling directories for cursors (`<workdir>/cursors/{open,exhausted,expired}/`).
"""
function init_cursor_manager(workdir::String)
    for state in ("open", "exhausted", "expired")
        mkpath(joinpath(workdir, "cursors", state))
    end
    return CursorManager(workdir)
end

_state_dir(manager::CursorManager, state::CursorState) = joinpath(manager.workdir, "cursors", lowercase(string(state)))
_cursor_path(manager::CursorManager, state::CursorState, id::String) = joinpath(_state_dir(manager, state), "$(id).cursor.json")

"""
    create_cursor!(manager, index_uuid, results::AbstractVector; page_size=20, ttl_seconds=300) -> String

Materializes an already-computed, ordered `results` list behind a fresh opaque cursor id,
written once in `open/` state. PLAN.md §5.4: a cursor exists specifically because paging
through an expensive-to-recompute result set (a large `search`, a filtered listing that
had to over-fetch to satisfy `k`, a job's result) must reuse the same materialized page(s)
on every subsequent request rather than re-running the underlying query per page — so the
full `results` list is stored once, up front, and later `poll_cursor!` calls only ever
slice it.

# Arguments
- `manager::CursorManager`
- `index_uuid::String`: which dataset/index this cursor's results came from (recorded for
  operator visibility — PLAN.md's own path convention nests cursors under it, though this
  implementation keeps the flat layout `Jobs.JobManager` already established for the same
  pragmatic reason: no separate per-index directory bookkeeping to keep in sync).
- `results::AbstractVector`: the full, already-ordered result set to page through. Must be
  JSON-serializable (plain `Dict`/`Vector`/primitive values — the same shapes every other
  handler in `Server` already returns).
- `page_size::Int`: default page size for `poll_cursor!` calls that don't override it.
- `ttl_seconds::Int`: how long the cursor stays pollable before `poll_cursor!`/`gc_expired!`
  treat it as expired.

# Returns
- `String`: the generated cursor id.
"""
function create_cursor!(manager::CursorManager, index_uuid::String, results::AbstractVector; page_size::Int=20, ttl_seconds::Int=300)
    id = Project.generate_id()
    now_str = string(now(UTC))
    record = Dict{String, Any}(
        "cursor_id" => id,
        "index_uuid" => index_uuid,
        "created_at" => now_str,
        "last_polled_at" => now_str,
        "expires_at" => string(now(UTC) + Second(ttl_seconds)),
        "offset" => 0,
        "page_size" => page_size,
        "total" => length(results),
        "results" => results,
    )

    path = _cursor_path(manager, Open, id)
    temp_path = path * ".tmp"
    open(temp_path, "w") do io
        write(io, JSON3.write(record))
    end
    mv(temp_path, path, force=true)

    return id
end

"""
    get_cursor(manager, id) -> (state, record) or (nothing, nothing)

Searches for a cursor across all three states and returns its `(state, parsed_record)`,
mirroring `Jobs.get_job`. Doesn't mutate anything or check expiry — use `poll_cursor!` to
actually advance/expire a cursor; this is the read-only lookup for e.g. reporting status.
"""
function get_cursor(manager::CursorManager, id::String)
    for state in (Open, Exhausted, Expired)
        path = _cursor_path(manager, state, id)
        if isfile(path)
            return state, JSON3.read(read(path, String), Dict{String, Any})
        end
    end
    return nothing, nothing
end

_is_expired(record::AbstractDict) = now(UTC) > DateTime(record["expires_at"])

"""
    poll_cursor!(manager, id; limit=nothing) -> Union{NamedTuple, Nothing}

Returns the next page of an `open/` cursor's materialized results and advances its
`offset`, transitioning it to `exhausted/` once the underlying result set is fully
drained (or to `expired/` if its TTL had already lapsed since the last poll). Returns
`nothing` if `id` isn't currently in `open/` — already exhausted/expired, or never
existed at all; callers that need to tell those two cases apart should check
`get_cursor` first.

# Keyword Arguments
- `limit::Union{Int,Nothing}`: page size for this call only; defaults to the cursor's own
  `page_size` recorded at creation time.

# Returns
A `(results, exhausted, total, offset)` named tuple, or `nothing`.
"""
function poll_cursor!(manager::CursorManager, id::String; limit::Union{Int, Nothing}=nothing)
    path = _cursor_path(manager, Open, id)
    isfile(path) || return nothing

    record = JSON3.read(read(path, String), Dict{String, Any})
    if _is_expired(record)
        mv(path, _cursor_path(manager, Expired, id), force=true)
        return nothing
    end

    offset = record["offset"]
    total = record["total"]
    page_size = limit === nothing ? record["page_size"] : limit
    page = record["results"][(offset + 1):min(offset + page_size, total)]
    new_offset = offset + length(page)
    exhausted = new_offset >= total

    record["offset"] = new_offset
    record["last_polled_at"] = string(now(UTC))

    temp_path = path * ".tmp"
    open(temp_path, "w") do io
        write(io, JSON3.write(record))
    end
    mv(temp_path, path, force=true)

    exhausted && mv(path, _cursor_path(manager, Exhausted, id), force=true)

    return (results=page, exhausted=exhausted, total=total, offset=new_offset)
end

"""
    gc_expired!(manager) -> Vector{String}

Sweeps `open/` for any cursor whose TTL has lapsed without ever being fully drained and
moves it to `expired/` (an `exhausted/` cursor already reached its natural end and isn't
touched here — PLAN.md's `similarity-search-ctl jobs gc` note groups Job and cursor GC
under one operator command, see `Server.handle_jobs_gc`, which calls this and
`purge_expired!` together). Returns the ids that were moved.
"""
function gc_expired!(manager::CursorManager)
    dir = _state_dir(manager, Open)
    moved = String[]
    for fname in readdir(dir)
        endswith(fname, ".cursor.json") || continue
        id = fname[1:end-length(".cursor.json")]
        path = joinpath(dir, fname)
        record = JSON3.read(read(path, String), Dict{String, Any})
        if _is_expired(record)
            mv(path, _cursor_path(manager, Expired, id), force=true)
            push!(moved, id)
        end
    end
    return moved
end

"""
    purge_expired!(manager) -> Vector{String}

The actual disk-reclaim half of cursor GC: deletes every file already sitting in
`expired/` (every one of them is, by construction, past its `ttl_seconds` -- that's why
`gc_expired!` moved it there in the first place, so there's no separate age check here).
Returns the ids removed.
"""
function purge_expired!(manager::CursorManager)
    dir = _state_dir(manager, Expired)
    removed = String[]
    for fname in readdir(dir)
        endswith(fname, ".cursor.json") || continue
        rm(joinpath(dir, fname))
        push!(removed, fname[1:end-length(".cursor.json")])
    end
    return removed
end

end # module
