module Telemetry

using RocksDB
using JSON3
using Dates
using SimilaritySearch
using ..Project

export log_operation, snapshot_costs, log_request!, identity_fields
export MetricsRegistry, record_metric!, prometheus_lines, LATENCY_BUCKETS

"""
    log_operation(manager::ProjectManager, op_type::String, details::Dict{String, Any})

Logs an operation into the op_log Column Family for telemetry.
Includes timestamp, operation type, and arbitrary details (like distance evaluations).

# Arguments
- `manager::ProjectManager`: The dataset manager.
- `op_type::String`: The type of operation being logged.
- `details::Dict{String, Any}`: Arbitrary details about the operation.
"""
function log_operation(manager::ProjectManager, op_type::String, details::Dict{String, Any}; metrics=nothing)
    record_metric!(metrics, string(get(details, "index_uuid", "")), op_type,
                   Float64(get(details, "elapsed_seconds", 0.0));
                   distance_evaluations=Int(get(details, "distance_evaluations", 0)),
                   items_inserted=Int(get(details, "items_inserted", 0)))
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
carries, per `SimilaritySearch.jl`'s own cost-accounting API) before an operation, so
the delta against it after the operation gives that operation's own cost rather than the
index's lifetime total (`ctx.costdists` is never reset). `nothing` for `ctx === nothing`
(an untrained text engine has no context yet) -- callers pass that straight through to
`log_request!`, which then just reports 0 distance evaluations instead of erroring.

This works for **insertion** (`handle_append`), which runs on the engine's own
`backend.ctx` under an exclusive write lock. It does *not* work for a search: a search runs
on a context borrowed from the engine's pool for the duration of the call, shared with
whatever else is searching concurrently, so `_run_search` asks the engine for the count
instead (`SimilaritySearchEngine.SearchStats`) and passes it to `log_request!` directly.
"""
snapshot_costs(ctx) = ctx === nothing ? nothing : copy(ctx.costdists)

"""
    identity_fields(identity) -> Dict{String,Any}

The `(user, token_fingerprint)` pair as the two fields of a log record: `user` and
`token_fingerprint`. Written by every operation that records one, so a record identifies who
acted without carrying what they acted with.
"""
identity_fields(identity::Tuple) = Dict{String, Any}("user" => identity[1], "token_fingerprint" => identity[2])
identity_fields(::Nothing) = Dict{String, Any}("user" => nothing, "token_fingerprint" => nothing)

"""
    log_request!(manager, ctx, op_type, index_uuid, t0, snapshot; token=nothing, distance_name=nothing, dimension=nothing, extra=Dict{String,Any}())

Logs one `op_log` record for a request that just finished: elapsed wall-clock time since
`t0` (`time()`-based, matching PLAN.md §4.3's "delta computing time"), and the real
distance-evaluation count for *this* operation via `SimilaritySearch.distance_evaluations`
diffed against `snapshot` (from `snapshot_costs`, taken before the operation ran) -- 0 if
`ctx`/`snapshot` is `nothing` rather than erroring, since not every op touches an index
context (e.g. a soft-delete). `identity` is the pair `Server._request_identity` produces, `(user, token_fingerprint)`:
who made the request, when its token was valid, and a fingerprint of the token presented.
`distance_name`/`dimension` are `nothing` when not meaningful for `op_type` (see the call
sites in `server.jl`). `extra` merges in any op-type-specific fields (e.g. `append`'s
`items_inserted`).

Until 2026-09-14 this recorded the `Authorization` header verbatim. That put live credentials
in a column family of the project, in every `dump` of it, and in front of anyone allowed to
read the log.
"""
function log_request!(manager::Project.ProjectManager, ctx, op_type::String, index_uuid::String, t0::Float64, snapshot;
                       identity=(nothing, nothing), distance_name=nothing, dimension=nothing, distance_evaluations=nothing,
                       metrics=nothing, extra::Dict{String, Any}=Dict{String, Any}())
    elapsed_seconds = time() - t0
    evals = distance_evaluations !== nothing ? distance_evaluations :
            (ctx === nothing || snapshot === nothing) ? 0 : SimilaritySearch.distance_evaluations(ctx, snapshot)

    details = Dict{String, Any}(
        "index_uuid" => index_uuid,
        "distance_name" => distance_name,
        "dimension" => dimension,
        "distance_evaluations" => evals,
        "elapsed_seconds" => elapsed_seconds,
    )
    merge!(details, identity_fields(identity))
    merge!(details, extra)

    log_operation(manager, op_type, details; metrics)
end

# ==========================================
# Aggregates for /metrics (PLAN.md §5.8)
# ==========================================

"""
    LATENCY_BUCKETS

Upper bounds, in seconds, of the histogram of request durations. They cover the range this
server answers in: a text or dense query resolved from memory, at one millisecond, up to a
request that waits on storage, at seconds.
"""
const LATENCY_BUCKETS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)

"""
    MetricsRegistry

The counters `/metrics` reports, accumulated as requests are recorded. Every operation that
writes a record into the `op_log` adds the same numbers here, so the endpoint answers from
memory and does not read the log: a metrics collector polls every few seconds, and reading a
column family that grows with every request would make each poll more expensive than the
requests it measures.

The log remains the durable record, and `GET /api/v1/datasets/{id}/log` reads it. These
counters cover this process only, and start at zero when it starts, which is what a
Prometheus counter is expected to do; `simsearch_process_start_time_seconds` reports when
that was, so a collector can tell a restart from a drop in traffic.
"""
struct MetricsRegistry
    lock::ReentrantLock
    requests::Dict{Tuple{String, String}, Int}              # (dataset, operation) => requests
    duration_sum::Dict{Tuple{String, String}, Float64}      # (dataset, operation) => seconds
    duration_buckets::Dict{Tuple{String, String}, Vector{Int}}
    evaluations::Dict{String, Int}                          # dataset => distance computations
    items_inserted::Dict{String, Int}                       # dataset => items
    started_at::Float64
end

MetricsRegistry() = MetricsRegistry(ReentrantLock(),
                                    Dict{Tuple{String, String}, Int}(),
                                    Dict{Tuple{String, String}, Float64}(),
                                    Dict{Tuple{String, String}, Vector{Int}}(),
                                    Dict{String, Int}(),
                                    Dict{String, Int}(),
                                    time())

"""
    record_metric!(registry, dataset, operation, elapsed_seconds; distance_evaluations=0, items_inserted=0)

Adds one request to the counters. Called from [`log_request!`](@ref) and
[`log_operation`](@ref) with the same values they write into the `op_log`, so the two cannot
disagree about what happened. A `nothing` registry records nothing, which is what the command
lines and the tests that call the logging functions directly pass.
"""
function record_metric!(registry::MetricsRegistry, dataset::AbstractString, operation::AbstractString,
                        elapsed_seconds::Real; distance_evaluations::Integer=0, items_inserted::Integer=0)
    key = (String(dataset), String(operation))
    lock(registry.lock) do
        registry.requests[key] = get(registry.requests, key, 0) + 1
        registry.duration_sum[key] = get(registry.duration_sum, key, 0.0) + max(0.0, Float64(elapsed_seconds))
        buckets = get!(() -> zeros(Int, length(LATENCY_BUCKETS)), registry.duration_buckets, key)
        for (i, bound) in enumerate(LATENCY_BUCKETS)
            elapsed_seconds <= bound && (buckets[i] += 1)
        end
        distance_evaluations > 0 &&
            (registry.evaluations[key[1]] = get(registry.evaluations, key[1], 0) + Int(distance_evaluations))
        items_inserted > 0 &&
            (registry.items_inserted[key[1]] = get(registry.items_inserted, key[1], 0) + Int(items_inserted))
    end
    return nothing
end

record_metric!(::Nothing, args...; kwargs...) = nothing

_escape_label(s::AbstractString) = replace(String(s), "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n")

"""
    prometheus_lines(registry::MetricsRegistry) -> Vector{String}

The counters of `registry` in the Prometheus text format: one counter of requests and one
histogram of durations per dataset and operation, and one counter of distance computations
and one of inserted items per dataset. A registry that has recorded nothing produces the
declarations and no samples, which is a valid exposition and lets a collector distinguish
"no traffic" from "this server does not report this".
"""
function prometheus_lines(registry::MetricsRegistry)
    lines = String[]
    lock(registry.lock) do
        push!(lines, "# HELP simsearch_process_start_time_seconds Start time of this process, in seconds since the epoch.")
        push!(lines, "# TYPE simsearch_process_start_time_seconds gauge")
        push!(lines, "simsearch_process_start_time_seconds $(registry.started_at)")

        push!(lines, "# HELP simsearch_requests_total Requests recorded in the operation log since this process started.")
        push!(lines, "# TYPE simsearch_requests_total counter")
        for ((dataset, operation), n) in sort(collect(registry.requests), by=first)
            push!(lines, "simsearch_requests_total{dataset=\"$(_escape_label(dataset))\",operation=\"$(_escape_label(operation))\"} $n")
        end

        push!(lines, "# HELP simsearch_distance_evaluations_total Distance computations performed while answering requests.")
        push!(lines, "# TYPE simsearch_distance_evaluations_total counter")
        for (dataset, n) in sort(collect(registry.evaluations), by=first)
            push!(lines, "simsearch_distance_evaluations_total{dataset=\"$(_escape_label(dataset))\"} $n")
        end

        push!(lines, "# HELP simsearch_items_inserted_total Items accepted by append requests.")
        push!(lines, "# TYPE simsearch_items_inserted_total counter")
        for (dataset, n) in sort(collect(registry.items_inserted), by=first)
            push!(lines, "simsearch_items_inserted_total{dataset=\"$(_escape_label(dataset))\"} $n")
        end

        push!(lines, "# HELP simsearch_request_duration_seconds Time taken to answer a request.")
        push!(lines, "# TYPE simsearch_request_duration_seconds histogram")
        for ((dataset, operation), buckets) in sort(collect(registry.duration_buckets), by=first)
            labels = "dataset=\"$(_escape_label(dataset))\",operation=\"$(_escape_label(operation))\""
            for (i, bound) in enumerate(LATENCY_BUCKETS)
                push!(lines, "simsearch_request_duration_seconds_bucket{$labels,le=\"$bound\"} $(buckets[i])")
            end
            total = registry.requests[(dataset, operation)]
            push!(lines, "simsearch_request_duration_seconds_bucket{$labels,le=\"+Inf\"} $total")
            push!(lines, "simsearch_request_duration_seconds_sum{$labels} $(registry.duration_sum[(dataset, operation)])")
            push!(lines, "simsearch_request_duration_seconds_count{$labels} $total")
        end
    end
    return lines
end

end # module
