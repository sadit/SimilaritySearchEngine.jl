module Telemetry

using RocksDB
using JSON3
using Dates
using SimilaritySearch
using ..Project

export log_operation, snapshot_costs, log_request!

"""
    log_operation(manager::ProjectManager, op_type::String, details::Dict{String, Any})

Logs an operation into the op_log Column Family for telemetry.
Includes timestamp, operation type, and arbitrary details (like distance evaluations).

# Arguments
- `manager::ProjectManager`: The dataset manager.
- `op_type::String`: The type of operation being logged.
- `details::Dict{String, Any}`: Arbitrary details about the operation.
"""
function log_operation(manager::ProjectManager, op_type::String, details::Dict{String, Any})
    record = Dict(
        "timestamp" => string(now(UTC)),
        "operation" => op_type,
        "details" => details
    )
    
    # Generate a time-sortable key using nanoseconds
    key_bytes = Vector{UInt8}(string(time_ns()))
    val_bytes = Vector{UInt8}(JSON3.write(record))
    
    RocksDB.put!(manager.db, key_bytes, val_bytes; cf=manager.cf_op_log)
end

"""
    snapshot_costs(ctx) -> Union{Nothing, Vector{Int}}

Copies `ctx.costdists` (the per-batch distance-evaluation counters every
`AbstractContext` -- `GenericContext`/`SearchGraphContext`/`InvertedFileContext` alike --
carries, per `SimilaritySearch.jl`'s own cost-accounting API) before a search/insert, so
the delta against it after the operation gives that operation's own cost rather than the
index's lifetime total (`ctx.costdists` is never reset). `nothing` for `ctx === nothing`
(an untrained text engine has no context yet) -- callers pass that straight through to
`log_request!`, which then just reports 0 distance evaluations instead of erroring.
"""
snapshot_costs(ctx) = ctx === nothing ? nothing : copy(ctx.costdists)

"""
    log_request!(manager, ctx, op_type, index_uuid, t0, snapshot; token=nothing, distance_name=nothing, dimension=nothing, extra=Dict{String,Any}())

Logs one `op_log` record for a request that just finished: elapsed wall-clock time since
`t0` (`time()`-based, matching PLAN.md §4.3's "delta computing time"), and the real
distance-evaluation count for *this* operation via `SimilaritySearch.distance_evaluations`
diffed against `snapshot` (from `snapshot_costs`, taken before the operation ran) -- 0 if
`ctx`/`snapshot` is `nothing` rather than erroring, since not every op touches an index
context (e.g. a soft-delete). `token`/`distance_name`/`dimension` are `nothing` when not
meaningful for `op_type` (see call sites in `server.jl`) or not yet available (no request
authentication is wired up anywhere in this codebase yet -- `token` reports whatever the
caller passed, unauthenticated, not a verified identity). `extra` merges in any
op-type-specific fields (e.g. `append`'s `items_inserted`).
"""
function log_request!(manager::Project.ProjectManager, ctx, op_type::String, index_uuid::String, t0::Float64, snapshot;
                       token=nothing, distance_name=nothing, dimension=nothing, distance_evaluations=nothing, extra::Dict{String, Any}=Dict{String, Any}())
    elapsed_seconds = time() - t0
    evals = distance_evaluations !== nothing ? distance_evaluations :
            (ctx === nothing || snapshot === nothing) ? 0 : SimilaritySearch.distance_evaluations(ctx, snapshot)

    details = Dict{String, Any}(
        "index_uuid" => index_uuid,
        "token" => token,
        "distance_name" => distance_name,
        "dimension" => dimension,
        "distance_evaluations" => evals,
        "elapsed_seconds" => elapsed_seconds,
    )
    merge!(details, extra)

    log_operation(manager, op_type, details)
end

end # module
