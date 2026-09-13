module Server

using HTTP
using Oxygen
using JSON3
using Dates
using RocksDB
using SimilaritySearch
using TextSearch
using SimilaritySearch
using TextSearch: BeamSearch, SearchGraph
using ..Schema
using ..Project
using ..Tokens
using ..IndexEngine
using ..Jobs
using ..Cursors
using ..Executors
using ..Persistence
using ..Telemetry

# The engine's *public* surface, qualified rather than `using`-ed: `search`, `index!` and
# `append_items!` are exported by SimilaritySearch too, and a bare import would make every
# one of them ambiguous here. Qualifying also keeps the boundary visible -- everything this
# module calls through `SSE.` is API the engine promises, while the `..IndexEngine`/`..Project`
# imports above are internals it still reaches for (see DEVELOPMENT_STRATEGY.md; that list
# should only ever shrink).
import SimilaritySearchEngine as SSE
using SparseArrays: SparseArrays

export run_server, AppState, reload_datasets!, parse_index_kind, parse_distance

const JSON_HEADERS = ["Content-Type" => "application/json"]

json_response(status::Int, body; extra_headers=[]) = HTTP.Response(status, vcat(JSON_HEADERS, extra_headers), JSON3.write(body))

"""
`id` values are interpolated directly into filesystem paths (dataset directories,
descriptor sidecar files) — PLAN.md §4.6 flags exactly this class of risk for the
similar `key` field. Whitelist before it ever touches a path.
"""
const DATASET_ID_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"
valid_project_id(id::AbstractString) = occursin(DATASET_ID_PATTERN, id)

"""
A text/lexical join-group member's `key` (PLAN.md §1/§5.3) is user-supplied and gets
echoed back into API responses and compared against elsewhere -- whitelist it the same
way as dataset `id`, and separately reject the literal `"*"`, which is reserved for
`ftsearch`'s fan-out wildcard and must never be a real member's key (PLAN.md §5.3's
validation note).
"""
const KEY_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
valid_key(key::AbstractString) = key != "*" && occursin(KEY_PATTERN, key)

"""
    BEAMSEARCH_SAFETY_MULTIPLIER

The §3 Core execution barrier's safety multiplier: a per-request `beamsearch_overrides`
(§3/§7) may not exceed this factor times the index's own calibrated baseline (or
`IndexEngine.DEFAULT_BEAMSEARCH` if `calibrate!`/`POST .../calibrate` was never called) —
anchored to a real, per-index baseline rather than the arbitrary global constants
(`bsize<=20`, `Δ<1.7`) PLAN.md's own §3 warning flags as unjustified.
"""
const BEAMSEARCH_SAFETY_MULTIPLIER = 3.0

"""
    _resolve_beamsearch_override(engine, overrides) -> (bs_or_nothing, warn_or_errmsg)

Validates a caller-supplied `beamsearch_overrides` dict against `BEAMSEARCH_SAFETY_MULTIPLIER`
times the index's current baseline (PLAN.md §3). Unspecified fields default to the
baseline's own value, so a caller only needs to name the field(s) it actually wants to
raise. Returns:
- `(bs::BeamSearch, exceeds_baseline::Bool)` on success — `exceeds_baseline` is `true`
  when the (still-allowed, sub-cap) override asks for more than the baseline itself, the
  signal `handle_search` uses to attach a `WARN`/response-header (PLAN.md §3's "log a WARN
  event and return an HTTP warning header" for elevated-but-allowed parameters).
- `(nothing, error_message::String)` if the override exceeds the hard safety cap, or if
  `engine`'s index has no `BeamSearch` to override at all (text/exact indices).
"""
function _resolve_beamsearch_override(engine::AbstractSearchEngine, overrides::AbstractDict)
    baseline = IndexEngine.current_beamsearch(engine)
    baseline === nothing && return (nothing, "beamsearch_overrides only applies to a searchgraph (approximate dense) index")

    bsize = Int32(get(overrides, "bsize", baseline.bsize))
    delta = Float32(get(overrides, "delta", baseline.Δ))
    maxvisits = Int64(get(overrides, "maxvisits", baseline.maxvisits))

    cap_bsize = baseline.bsize * BEAMSEARCH_SAFETY_MULTIPLIER
    cap_delta = baseline.Δ * BEAMSEARCH_SAFETY_MULTIPLIER
    cap_maxvisits = baseline.maxvisits * BEAMSEARCH_SAFETY_MULTIPLIER

    if bsize > cap_bsize || delta > cap_delta || maxvisits > cap_maxvisits
        return (nothing, "beamsearch_overrides exceed the $(BEAMSEARCH_SAFETY_MULTIPLIER)x safety multiplier over the calibrated baseline " *
            "(bsize<=$(round(Int, cap_bsize)), delta<=$(round(cap_delta, digits=3)), maxvisits<=$(round(Int, cap_maxvisits)))")
    end

    exceeds_baseline = bsize > baseline.bsize || delta > baseline.Δ || maxvisits > baseline.maxvisits
    return (BeamSearch(bsize=bsize, Δ=delta, maxvisits=maxvisits), exceeds_baseline)
end

"""
    _parse_offset_limit(params, total) -> (offset, limit)

Plain, stateless `offset`/`limit` pagination (PLAN.md §5.4) for listing endpoints cheap
enough to recompute per request (`GET /api/v1/datasets`, `GET /api/v1/jobs`) — distinct
from the opaque `Cursors`-backed pagination `handle_search` uses for an actually
expensive-to-recompute result set. `offset` clamps to `0`; `limit` defaults to `total`
(i.e. "no limit") when absent or unparseable.
"""
function _parse_offset_limit(params::AbstractDict, total::Int)
    offset = max(0, something(tryparse(Int, get(params, "offset", "0")), 0))
    limit = something(tryparse(Int, get(params, "limit", string(total))), total)
    return offset, limit
end

"""
    AppState

Holds the global application state for the server.

# Fields
- `workdir::String`: Root working directory for data and jobs.
- `job_mgr::Any`: Job manager instance.
- `cursor_mgr::Any`: Result-cursor manager instance (§5.4's pagination protocol).
- `executor::Any`: Job executor backend.
- `token_mgr::Any`: Token manager instance (§4.1 identity model).
- `datasets::Dict{String, ProjectManager}`: Cache of open datasets.
- `engines::Dict{String, AbstractSearchEngine}`: Cache of open search engines.
- `lock::ReentrantLock`: Thread-safe lock for modifying caches.
"""
struct AppState
    workdir::String
    job_mgr::Any # JobManager
    cursor_mgr::Any # Cursors.CursorManager
    executor::Any # AbstractJobExecutor
    token_mgr::Any # Tokens.TokenManager
    # One open project per dataset id. This used to be two maps -- a ProjectManager and an
    # AbstractSearchEngine, opened and wired by hand here -- which is how this module ended up
    # depending on the engine's internals and breaking every time they moved. An
    # `SSE.EmbeddedEngine` is the engine's own handle for exactly this pair, and it keeps them
    # consistent (a project opened, its engine restored, its staged backlog intact).
    handles::Dict{String, SSE.EmbeddedEngine}

    lock::ReentrantLock
end

# ==========================================
# Control / Health Endpoints
# ==========================================

"""
    handle_healthz(req) -> HTTP.Response

Liveness (PLAN.md §5.8): the process is up and able to answer HTTP at all. Deliberately
checks nothing else — that's what `/readyz` is for.
"""
handle_healthz(req::HTTP.Request) = json_response(200, Dict("status" => "ok"))

"""
    handle_readyz(req, app) -> HTTP.Response

Readiness (PLAN.md §5.8): the working directory is accessible and the job/token
managers this process depends on for every request were actually constructed. Crash
recovery (`Jobs.requeue_stale_running!`) already runs synchronously in `main` before
`run_server` is ever called, so there's no separate "WAL replay pending" state to check
here — by the time this handler is reachable, that pass has already completed.
"""
function handle_readyz(req::HTTP.Request, app::AppState)
    ready = isdir(app.workdir) && app.job_mgr !== nothing && app.token_mgr !== nothing
    return json_response(ready ? 200 : 503, Dict("status" => ready ? "ok" : "not_ready"))
end

"""
    handle_metrics(req, app) -> HTTP.Response

Prometheus text-format metrics (PLAN.md §5.8), limited to what's actually measurable
today: dataset/job counts and per-loaded-dataset doc/tombstone counts. `op_log`-derived
counters (`distance_evaluations_total`, `query_latency_seconds`) are still omitted here —
`Telemetry.log_operation` is now wired into `search`/`ftsearch`/`append`/`delete` (see
`_run_search`/`handle_append`/`handle_delete_item`), so `cf_op_log` has real data to
aggregate, but scraping/summarizing it into Prometheus counters (across however many
loaded datasets, without re-scanning the whole CF on every `/metrics` poll) is its own
separate design question, not attempted in this pass.
"""
function handle_metrics(req::HTTP.Request, app::AppState)
    lines = String[]

    push!(lines, "# HELP simsearch_datasets_loaded Number of datasets currently open in memory.")
    push!(lines, "# TYPE simsearch_datasets_loaded gauge")
    push!(lines, "simsearch_datasets_loaded $(length(app.handles))")

    push!(lines, "# HELP simsearch_threads Number of Julia threads available to this process.")
    push!(lines, "# TYPE simsearch_threads gauge")
    push!(lines, "simsearch_threads $(Threads.nthreads())")

    active = length(Jobs.list_jobs(app.job_mgr, Jobs.Queued)) + length(Jobs.list_jobs(app.job_mgr, Jobs.Running))
    push!(lines, "# HELP simsearch_active_jobs Jobs currently queued or running.")
    push!(lines, "# TYPE simsearch_active_jobs gauge")
    push!(lines, "simsearch_active_jobs $active")

    push!(lines, "# HELP simsearch_dataset_doc_count Live document count per loaded dataset.")
    push!(lines, "# TYPE simsearch_dataset_doc_count gauge")
    push!(lines, "# HELP simsearch_dataset_tombstone_ratio Soft-deleted fraction of a loaded dataset's documents.")
    push!(lines, "# TYPE simsearch_dataset_tombstone_ratio gauge")
    for (id, handle) in app.handles
        engine = handle.engine
        doc_count = engine.backend.index === nothing ? 0 : length(engine.backend.index)
        ratio = doc_count == 0 ? 0.0 : length(engine.deleted_ids) / doc_count
        push!(lines, "simsearch_dataset_doc_count{dataset=\"$id\"} $doc_count")
        push!(lines, "simsearch_dataset_tombstone_ratio{dataset=\"$id\"} $ratio")
    end

    body = join(lines, "\n") * "\n"
    return HTTP.Response(200, ["Content-Type" => "text/plain; version=0.0.4"], body)
end

# ==========================================
# Dataset Endpoints
# ==========================================

"""
    parse_meta_schema(data::Dict) -> Schema.MetaSchema

Parses an optional `"meta_schema"` array (`[{"name":..., "type":..., "indexed":...}, ...]`)
from a `POST /api/v1/datasets` request body (PLAN.md §4.5/§5.7). Defaults to an empty schema
(today's fully-free-form-via-`extra` behavior) when absent.
"""

"""
    serialize_meta_schema(schema::Schema.MetaSchema) -> Vector

The inverse of `parse_meta_schema` -- turns a `MetaSchema` back into the same
`[{"name":..., "type":..., "indexed":...}, ...]` JSON shape, so it can be persisted into
the dataset's `descriptor.json` at creation time (see `handle_create_dataset`) and read
back by `reload_datasets!` on server startup. Without this, a restarted server would
reopen every dataset with the *default* empty `MetaSchema`, silently losing any
schema-declared secondary-index column families for datasets that had one.
"""

"""
    descriptor_path(app, id) -> String

Path of a dataset's small persistent sidecar (`index_kind`/`distance`/`created_at`/
`meta_schema`) — recorded at creation time so `GET /api/v1/datasets`/`GET
/api/v1/datasets/{id}` can list every dataset without needing it loaded into
`app.handles`, and so `reload_datasets!` can reopen each one (with its original
`meta_schema`, hence its original secondary-index column families) on server startup.
"""
descriptor_path(app::AppState, id::String) = joinpath(app.workdir, "datasets", id, "descriptor.json")

"""
    join_group_members(app, join_group) -> Vector{Dict{String,Any}}

Scans every dataset descriptor under `<workdir>/datasets/` for ones tagged with the given
`join_group` (PLAN.md §1/§7's JoinGroup) -- there's no separate join-group table, the
group is purely an emergent grouping of datasets that happen to share the same tag in
their own descriptor. Reads the raw persistent descriptor directly (not
`dataset_descriptor`'s live-stats merge) since callers here only need the small set of
identifying fields (`id`/`index_kind`/`key`/`holds_metadata`), not per-request doc counts.
"""
function join_group_members(app::AppState, join_group::AbstractString)
    base = joinpath(app.workdir, "datasets")
    isdir(base) || return Dict{String, Any}[]

    members = Dict{String, Any}[]
    for id in readdir(base)
        isdir(joinpath(base, id)) || continue
        dpath = descriptor_path(app, id)
        isfile(dpath) || continue
        desc = JSON3.read(read(dpath, String), Dict{String, Any})
        get(desc, "join_group", nothing) == join_group && push!(members, desc)
    end
    return members
end

function handle_create_dataset(req::HTTP.Request, app::AppState)
    body = String(req.body)
    data = isempty(body) ? Dict{String, Any}() : JSON3.read(body, Dict{String, Any})

    id = get(data, "id", Project.generate_id())
    valid_project_id(id) || return json_response(400, Dict("error" => "invalid dataset id"))

    join_group = get(data, "join_group", nothing)
    join_group !== nothing && (join_group = string(join_group))
    holds_metadata = get(data, "holds_metadata", false) === true

    key = get(data, "key", nothing)
    if key !== nothing
        key = string(key)
        valid_key(key) || return json_response(400, Dict(
            "error" => key == "*" ? "key '*' is reserved for ftsearch's fan-out wildcard and can't be assigned to a dataset" : "invalid key"
        ))
    end

    if holds_metadata && join_group !== nothing
        for m in join_group_members(app, join_group)
            if get(m, "holds_metadata", false) == true
                return json_response(400, Dict("error" => "join_group '$join_group' already has a holds_metadata member ('$(m["id"])')"))
            end
        end
    end

    index_type = get(data, "index_type", get(data, "index_kind", "searchgraph"))
    distance = get(data, "distance", "L2")
    dimension = get(data, "dimension", nothing)

    engine_type, backend_type = try
        parse_index_kind(index_type)
    catch e
        e isa SSE.UnknownBackend || rethrow()
        return json_response(400, Dict("error" => sprint(showerror, e)))
    end

    try
        lock(app.lock) do
            haskey(app.handles, id) && return
            app.handles[id] = SSE.create_project(datasets_root(app), id;
                engine=engine_type, backend=backend_type,
                distance=parse_distance(distance),
                dimension=dimension === nothing ? nothing : Int(dimension),
                textmodel=default_textmodel(engine_type))
        end
    catch e
        e isa SSE.EngineError || rethrow()
        return engine_error_response(e)
    end

    descriptor = Dict(
        "id" => id, "index_kind" => index_type, "distance" => distance, "created_at" => string(now(UTC)),
        "join_group" => join_group, "holds_metadata" => holds_metadata, "key" => key
    )
    write(descriptor_path(app, id), JSON3.write(descriptor))

    return json_response(201, Dict("status" => "created", "id" => id))
end

"""
    _reload_one_dataset!(app, id) -> Bool

Reopens exactly one dataset from disk into `app.handles`, unconditionally
(overwriting any existing entry for `id`) -- the shared piece both `reload_datasets!`
(cold-start, every dataset, skip-if-already-loaded) and `handle_reload_dataset`
(one dataset, on purpose, always overwrite) build on. Returns `false` without touching
`app` if `id` has no `descriptor.json` (not a real dataset directory).
"""
function _reload_one_dataset!(app::AppState, id::String)
    path = joinpath(app.workdir, "datasets", id)
    dpath = descriptor_path(app, id)
    isfile(dpath) || return false
    descriptor = JSON3.read(read(dpath, String), Dict{String, Any})

    index_type = get(descriptor, "index_kind", "searchgraph")
    distance = get(descriptor, "distance", "L2")

    # Whatever the descriptor says about index kind and distance, the project itself records
    # what it is and `open_project` restores it -- index, profile, staged backlog, tombstones
    # and all. This used to rebuild an *empty* engine of the right type and, if a JLD2 snapshot
    # happened to be lying next to it, load that instead: a reopened dataset came back without
    # its contents unless someone had remembered to snapshot it.
    handle = SSE.open_project(datasets_root(app), id)
    lock(app.lock) do
        app.handles[id] = handle
    end
    return true
end

"""
    reload_datasets!(app::AppState) -> Vector{String}

Scans `<workdir>/datasets/` and reopens every dataset found there into `app.handles` -- the fix for the previously-documented gap where a restarted
`similarity-search-serve` could `list`/`describe` existing datasets (their `descriptor.json`
sidecar is enough for that) but not `search`/`append`/`ftsearch`/`calibrate` them until
they were recreated, since nothing ever called `IndexEngine.restore_engine` from the HTTP
startup path. Meant to be called once, right after `AppState` is constructed and before
`run_server` starts accepting requests.

Skips any id already present in `app.handles` (a no-op in the normal
"cold start with empty dicts" case, but this guard also makes the function safe to call
against an already-populated `AppState`, e.g. from a test that only unloaded one dataset).
To force-refresh a dataset that's already loaded, see `handle_reload_dataset` instead
(PLAN.md §5.7's "hot-reload a single dataset" note) -- this function deliberately never
overwrites an already-loaded entry on its own.

A dataset directory with a `descriptor.json` but no `.snapshot.jld2` yet (created via
`POST /api/v1/datasets` but never appended to) reopens with a freshly created empty
engine of its declared `index_kind`/`distance` -- exactly the state it was in right after
creation, before any items existed to snapshot.

Returns the list of dataset ids actually (re)loaded, for a startup log line.
"""
function reload_datasets!(app::AppState)
    base = joinpath(app.workdir, "datasets")
    isdir(base) || return String[]

    reloaded = String[]
    for id in sort(readdir(base))
        isdir(joinpath(base, id)) || continue
        haskey(app.handles, id) && continue
        _reload_one_dataset!(app, id) && push!(reloaded, id)
    end
    return reloaded
end

"""
    handle_delete_dataset(req, app, id)

Really deletes the dataset: closes its RocksDB handle, drops it from the in-memory
caches, and removes its on-disk directory.
"""
function handle_delete_dataset(req::HTTP.Request, app::AppState, id::String)
    valid_project_id(id) || return json_response(400, Dict("error" => "invalid dataset id"))

    lock(app.lock) do
        if haskey(app.handles, id)
            Project.close_project(app.handles[id].project)
            delete!(app.handles, id)
        end
    end

    path = joinpath(app.workdir, "datasets", id)
    isdir(path) && rm(path; force=true, recursive=true)

    return json_response(200, Dict("status" => "deleted", "id" => id))
end

"""
    handle_unload_dataset(req, app, id) -> HTTP.Response

`POST /api/v1/admin/datasets/{id}/unload` (PLAN.md §5.7's "hot-reload a single dataset"
open issue): closes this dataset's RocksDB handle and drops it from `app.handles`, so an external process -- concretely, the CLI `rebuild` (§1) -- can safely
open it for exclusive writing without hitting RocksDB's per-process write lock. The
intended flow for refreshing a live server's view of a dataset after an out-of-band CLI
`rebuild`: `unload` -> run `rebuild` -> `reload` (`handle_reload_dataset` below). `404` if
`id` isn't currently loaded (nothing to unload).
"""
function handle_unload_dataset(req::HTTP.Request, app::AppState, id::String)
    haskey(app.handles, id) || return json_response(404, Dict("error" => "dataset_not_loaded"))
    lock(app.lock) do
        Project.close_project(app.handles[id].project)
        delete!(app.handles, id)
    end
    return json_response(200, Dict("status" => "unloaded", "id" => id))
end

"""
    handle_reload_dataset(req, app, id) -> HTTP.Response

`POST /api/v1/admin/datasets/{id}/reload` (PLAN.md §5.7): (re)opens `id` from disk into
`app.handles` via `_reload_one_dataset!`, replacing any existing entry --
the deliberate counterpart to `reload_datasets!`'s cold-start skip-if-already-loaded
guard. Closes the current in-memory handle first if `id` happens to still be loaded (so
this also works as a plain "refresh this dataset's engine from its snapshot" op, not only
after an explicit `unload`). `404` if `id` has no `descriptor.json` on disk at all.
"""
function handle_reload_dataset(req::HTTP.Request, app::AppState, id::String)
    valid_project_id(id) || return json_response(400, Dict("error" => "invalid dataset id"))

    if haskey(app.handles, id)
        lock(app.lock) do
            Project.close_project(app.handles[id].project)
            delete!(app.handles, id)
        end
    end

    _reload_one_dataset!(app, id) || return json_response(404, Dict("error" => "dataset_not_found"))
    return json_response(200, Dict("status" => "reloaded", "id" => id))
end

"""
    dataset_descriptor(app, id) -> Union{Dict, Nothing}

Merges the persistent sidecar (`index_kind`/`distance`/`created_at`/`join_group`/
`holds_metadata`/`key`) with live stats from `app.handles` when the dataset happens to be
currently loaded (`doc_count`, `tombstone_count`/`_ratio`) — `reload_datasets!` populates
`app.handles` for every on-disk dataset right at server startup, so in the normal case the
two sources agree; `loaded == false` still shows up for the brief window before that
startup reload runs, or if a dataset was explicitly unloaded some other way. Returns
`nothing` if `id` isn't a real dataset directory. `get!`s in the join-group fields with
their pre-join-group defaults so a descriptor written before that feature existed still
round-trips.
"""
function dataset_descriptor(app::AppState, id::String)
    path = joinpath(app.workdir, "datasets", id)
    isdir(path) || return nothing

    dpath = descriptor_path(app, id)
    base = isfile(dpath) ? JSON3.read(read(dpath, String), Dict{String, Any}) :
        Dict{String, Any}("id" => id, "index_kind" => "unknown", "distance" => "unknown", "created_at" => nothing)

    get!(base, "join_group", nothing)
    get!(base, "holds_metadata", false)
    get!(base, "key", nothing)
    get!(base, "beamsearch_baseline", nothing)

    loaded = haskey(app.handles, id)
    base["loaded"] = loaded
    if loaded
        engine = app.handles[id].engine
        doc_count = engine.backend.index === nothing ? 0 : length(engine.backend.index)
        tombstones = length(engine.deleted_ids)
        base["doc_count"] = doc_count
        base["tombstone_count"] = tombstones
        base["tombstone_ratio"] = doc_count == 0 ? 0.0 : tombstones / doc_count

        # The live, in-memory `algo[]` is authoritative over whatever's in the on-disk
        # descriptor (which only gets rewritten on an explicit POST .../calibrate) --
        # report the real current baseline (§5.6), including the DEFAULT_BEAMSEARCH
        # bootstrap value before calibrate has ever run, not a possibly-stale sidecar.
        bs = IndexEngine.current_beamsearch(engine)
        base["beamsearch_baseline"] = bs === nothing ? nothing :
            Dict("bsize" => Int(bs.bsize), "delta" => Float64(bs.Δ), "maxvisits" => Int(bs.maxvisits))
    end
    return base
end

"""
    handle_list_datasets(req, app) -> HTTP.Response

`GET /api/v1/datasets?offset=&limit=` (PLAN.md §5.7/§5.4): lists every dataset directory
found under `<workdir>/datasets/`, not just the ones currently cached in
`app.handles` — this listing works purely off each dataset's `descriptor.json`
sidecar and doesn't need `reload_datasets!` to have run first (see `dataset_descriptor`'s
note). Paginated with plain `offset`/`limit` (§5.4) — cheap to recompute per request,
unlike `handle_search`'s cursor-backed pagination.
"""
function handle_list_datasets(req::HTTP.Request, app::AppState)
    base = joinpath(app.workdir, "datasets")
    ids = isdir(base) ? sort(filter(n -> isdir(joinpath(base, n)), readdir(base))) : String[]
    total = length(ids)

    offset, limit = _parse_offset_limit(queryparams(HTTP.URI(req.target)), total)
    page_ids = ids[(min(offset, total) + 1):min(offset + limit, total)]
    datasets = [dataset_descriptor(app, id) for id in page_ids]

    return json_response(200, Dict("status" => "ok", "datasets" => datasets, "total" => total, "offset" => offset, "limit" => limit))
end

"""
    handle_get_dataset(req, app, id) -> HTTP.Response

`GET /api/v1/datasets/{id}` (PLAN.md §5.7): single-dataset detail, including its
`join_group`/`holds_metadata`/`key` (`null`/`false`/`null` for a dataset never tagged
into a join group — see `dataset_descriptor`).
"""
function handle_get_dataset(req::HTTP.Request, app::AppState, id::String)
    valid_project_id(id) || return json_response(400, Dict("error" => "invalid dataset id"))
    desc = dataset_descriptor(app, id)
    desc === nothing && return json_response(404, Dict("error" => "dataset_not_found"))
    return json_response(200, desc)
end

"""
    handle_get_join_group(req, app, id) -> HTTP.Response

`GET /api/v1/datasets/{id}/join_group` (PLAN.md §5.7/§7): the full JoinGroup `id` belongs
to — its sibling members and which one `holds_metadata`. `members` is empty (not a 404)
if the dataset exists but was never tagged with a `join_group`, since that's a normal,
valid state (most datasets in this pass aren't part of one), not an error.
"""
function handle_get_join_group(req::HTTP.Request, app::AppState, id::String)
    valid_project_id(id) || return json_response(400, Dict("error" => "invalid dataset id"))
    desc = dataset_descriptor(app, id)
    desc === nothing && return json_response(404, Dict("error" => "dataset_not_found"))

    join_group = get(desc, "join_group", nothing)
    join_group === nothing && return json_response(200, Dict("join_group" => nothing, "members" => []))

    members = [
        Dict("index_uuid" => m["id"], "index_kind" => get(m, "index_kind", "unknown"),
             "holds_metadata" => get(m, "holds_metadata", false), "key" => get(m, "key", nothing))
        for m in join_group_members(app, join_group)
    ]
    return json_response(200, Dict("join_group" => join_group, "members" => members))
end

"""
    handle_get_op_log(req, app, id) -> HTTP.Response

`GET /api/v1/datasets/{id}/log?offset=&limit=` (PLAN.md §1's admin `log` command --
backs `similarity-search-ctl log`, the one command from that section's enumeration left
unimplemented until now, see chunk 15's note): a paginated, newest-first dump of `id`'s
`op_log` column family. Real content as of this chunk -- `Telemetry.log_operation` is now
wired into `search`/`ftsearch`/`append`/`delete` (`_run_search`/`handle_append`/
`handle_delete_item`). Cheap-enough-to-recompute full scan + plain `offset`/`limit`
(`_parse_offset_limit`, same as `handle_list_jobs`/`handle_list_datasets`), not a
`Cursors`-backed cursor -- `op_log` is already append-only historical data, none of
`Cursors`' "materialize an expensive-to-recompute result set" concern applies here. `404`
if `id` isn't a currently-loaded dataset (its `op_log` CF is only reachable through a live
`RocksDB.DB` handle, same constraint every other loaded-dataset-only endpoint already has).
"""
function handle_get_op_log(req::HTTP.Request, app::AppState, id::String)
    valid_project_id(id) || return json_response(400, Dict("error" => "invalid dataset id"))
    haskey(app.handles, id) || return json_response(404, Dict("error" => "dataset_not_found"))
    dataset = app.handles[id].project

    entries = Any[JSON3.read(String(v), Dict{String, Any}) for (_, v) in RocksDB.DBIterator(dataset.db; cf=dataset.cf_op_log)]
    reverse!(entries) # newest first -- op_log keys are time_ns()-ordered ascending

    params = queryparams(HTTP.URI(req.target))
    total = length(entries)
    offset, limit = _parse_offset_limit(params, total)
    page = entries[(min(offset, total) + 1):min(offset + limit, total)]

    return json_response(200, Dict("status" => "ok", "entries" => page, "total" => total, "offset" => offset, "limit" => limit))
end

"""
    handle_exists(req, app, index) -> HTTP.Response

`GET /api/v1/simsearch/{index}/exists?ids=a,b,c` (PLAN.md §5.1): lightweight
existence/tombstone check by id, without paying for a full `fetch`'s metadata decode.
Accepts either internal `doc_id`s or original ids, same resolution as `handle_fetch`.
"""
function handle_exists(req::HTTP.Request, app::AppState, index::String)
    haskey(app.handles, index) || return json_response(404, Dict("error" => "dataset_not_found"))
    dataset = app.handles[index].project
    engine = app.handles[index].engine

    params = queryparams(HTTP.URI(req.target))
    raw_ids = filter(!isempty, split(get(params, "ids", ""), ","))

    results = Any[]
    for raw_id in raw_ids
        id_str = string(raw_id)
        record = nothing
        maybe_int = tryparse(Int, id_str)
        maybe_int !== nothing && (record = Project.get_metadata(dataset, maybe_int))
        record === nothing && (record = Project.find_by_doc_id(dataset, id_str))

        exists = record !== nothing
        deleted = exists && engine !== nothing && (record._id in engine.deleted_ids)
        push!(results, Dict("id" => id_str, "exists" => exists, "deleted" => deleted))
    end

    return json_response(200, Dict("status" => "ok", "id" => index, "results" => results))
end

# ==========================================
# Index/Search Endpoints
# ==========================================

"""
    _typed_item(kind::Symbol, item::AbstractDict) -> Union{Nothing,SSE.AbstractItem}

One wire item as the engine's own item type, or `nothing` when it carries no payload for
this kind of project (a text item with no `text`, a dense one with no `vector`).

The metadata half -- `doc_id`, `keywords`, `refs` and the free-form `meta` -- travels on the
item itself, which is why this module no longer writes `MetadataRecord`s by hand: ids are the
engine's to assign, and having two places assign them is how they drift.
"""
function _typed_item(kind::Symbol, item::AbstractDict)
    doc_id = get(item, "doc_id", nothing)
    doc_id = doc_id === nothing ? nothing : string(doc_id)
    keywords = String[string(k) for k in get(item, "keywords", String[])]
    raw_refs = get(item, "refs", get(item, "ref", String[]))
    refs = String[string(r) for r in (raw_refs isa AbstractVector ? raw_refs : [raw_refs])]
    reserved = ("doc_id", "keywords", "refs", "ref", "vector", "indices", "values", "text", "_id", "schema_version")
    # An empty dict, not `nothing`: the item constructors take `meta::AbstractDict`, and an
    # item with no free-form metadata simply has none.
    meta = if haskey(item, "meta") && item["meta"] isa AbstractDict
        Dict{String, Any}(string(k) => v for (k, v) in item["meta"])
    else
        Dict{String, Any}(string(k) => v for (k, v) in item if string(k) ∉ reserved)
    end

    if kind === :text
        haskey(item, "text") || return nothing
        return SSE.TextItem(string(item["text"]); doc_id, keywords, refs, meta)
    elseif kind === :dense
        haskey(item, "vector") || return nothing
        return SSE.DenseItem(convert(Vector{Float32}, item["vector"]); doc_id, keywords, refs, meta)
    end
    # sparse: the caller encoded it, so it sends the encoding rather than a dense vector
    (haskey(item, "indices") && haskey(item, "values")) || return nothing
    ind = Int32[Int32(i) for i in item["indices"]]
    val = Float32[Float32(v) for v in item["values"]]
    dim = Int(get(item, "dimension", maximum(ind; init=Int32(0))))
    return SSE.SparseItem(SparseArrays.sparsevec(ind, val, dim); doc_id, keywords, refs, meta)
end

function handle_append(req::HTTP.Request, app::AppState, index::String)
    haskey(app.handles, index) || return json_response(404, Dict("error" => "dataset_not_found"))

    handle = app.handles[index]
    engine = handle.engine
    dataset = handle.project

    data = JSON3.read(String(req.body), Dict{String, Any})
    items = get(data, "items", [])

    snapshot = Telemetry.snapshot_costs(engine.backend.ctx)
    t0 = time()

    kind = SSE.payload_kind(engine)
    typed = SSE.AbstractItem[]
    for item in items
        it = _typed_item(kind, item)
        it === nothing || push!(typed, it)
    end

    inserted = 0
    try
        if !isempty(typed)
            inserted = SSE.append_items!(handle, typed)
            SSE.index!(handle)
        end
    catch e
        e isa SSE.EngineError || rethrow()
        return engine_error_response(e)
    end

    Telemetry.log_request!(dataset, engine.backend.ctx, "append", index, t0, snapshot;
        token=_request_token(req), distance_name=_engine_distance_name(engine), dimension=_engine_dimension(engine),
        extra=Dict{String, Any}("items_inserted" => inserted))

    return json_response(200, Dict("status" => "appended", "id" => index, "inserted" => inserted))
end

"""
    _format_results(hits, k) -> (results, insufficient_results)

Turns the engine's hits into the wire's shape, and flags `insufficient_results` when fewer
than `k` came back (PLAN.md §5.4's explicit signal, rather than silently returning a short
page).

Soft-deleted hits are dropped here rather than hidden by the engine: `SSE.search` reports
them with `deleted=true` and no `doc_id`, on purpose, and it is this API's choice not to show
them.
"""
function _format_results(hits, k::Int)
    results = Any[]
    for hit in hits
        hit.deleted && continue
        orig_id = hit.doc_id === nothing ? string(hit._id) : hit.doc_id
        push!(results, Dict("id" => orig_id, "doc_id" => hit._id, "distance" => hit.distance))
    end
    return results, length(results) < k
end

"""
    _filter_predicate(filter_spec) -> Function

The wire's filter object as the predicate `SSE.search` takes.

The engine's filtering keyword is a plain Julia function over `(record, meta)`, so the JSON
spec stays this module's business -- `Schema.matches_filter` already knows how to read it --
and the engine never learns a wire format.
"""
_filter_predicate(filter_spec::AbstractDict) =
    (record, meta) -> Schema.matches_filter(record, meta, filter_spec)

"""
    FILTER_CANDIDATES(k) -> Int

How many candidates a filtered search reads before applying the predicate.

`SSE.search` defaults to `2k`, which suits a predicate most items satisfy; this API keeps the
`max(5k, k+20)` budget it has always used, since a metadata filter here is typically narrow
and the caller has no way to pass its own budget yet. A filtered page can still come back
short, and that is what `insufficient_results` reports.
"""
FILTER_CANDIDATES(k::Int) = max(k * 5, k + 20)

"""
    _request_token(req::HTTP.Request) -> Union{Nothing, String}

Best-effort extraction of the caller-presented token from an `Authorization` header
(`Bearer <token>` or the bare token string), for `op_log` telemetry purposes only --
`nothing` if the header is absent. **Not authentication**: no request handler in this
codebase currently validates a presented token against `Tokens.get_token` at all (identity
enforcement per PLAN.md §4.1 isn't wired up yet, a separate, larger prerequisite) -- this
just records whatever the caller happened to send, unverified, so a future telemetry query
already has the field populated once real auth enforcement lands.
"""
function _request_token(req::HTTP.Request)
    raw = HTTP.header(req, "Authorization", "")
    isempty(raw) && return nothing
    startswith(raw, "Bearer ") ? raw[8:end] : raw
end

"""
    _engine_distance_name(engine::AbstractSearchEngine) -> Union{Nothing, String}

The index's distance function name (e.g. `"SqL2"`), for `op_log`'s `distance_name` field
(PLAN.md §4.3) -- the same `string(nameof(typeof(SimilaritySearch.distance(engine.backend.index))))`
pattern `execute_dump`/`execute_describe` already use. `nothing` for a text index (BM25/
weighted-inverted-file scoring isn't a `SimilaritySearch.distance` metric in this sense) or
an untrained/empty one.
"""
function _engine_distance_name(engine::AbstractSearchEngine)
    SSE.payload_kind(engine) === :text && return nothing
    engine.backend.index === nothing && return nothing
    string(nameof(typeof(SimilaritySearch.distance(engine.backend.index))))
end

"""
    _engine_dimension(engine::AbstractSearchEngine) -> Union{Nothing, Int}

The dense vector dimension of `engine`'s first indexed item, for `op_log`'s `dimension`
field -- `nothing` for a text index (vocabulary size isn't "dimension" in the same sense)
or an empty index (nothing indexed yet to measure).
"""
function _engine_dimension(engine::AbstractSearchEngine)
    SSE.payload_kind(engine) === :dense || return nothing
    engine.backend.index === nothing && return nothing
    length(SimilaritySearch.database(engine.backend.index)) == 0 && return nothing
    length(SimilaritySearch.database(engine.backend.index, 1))
end

"""
    _hydrate_raw(handle, raw) -> Vector{SSE.SearchResult}

The engine's raw live-search tuple as the hit type its public `search` returns, so every
caller here sees one shape. Only the beam-override path needs this (see [`_run_search`](@ref)).
"""
function _hydrate_raw(handle, raw)
    out = SSE.SearchResult[]
    for (id, dist, del) in zip(raw.id, raw.dist, raw.deleted)
        _id = Int32(id)
        if del
            push!(out, SSE.SearchResult(_id, nothing, Float32(dist), true))
        else
            record = SSE.get_metadata(handle.project, _id)
            push!(out, SSE.SearchResult(_id, record === nothing ? nothing : record.doc_id, Float32(dist), false))
        end
    end
    out
end

"""
    _run_search(handle, index, req, query, k; filter_spec, bs_override, text) -> Vector{SSE.SearchResult}

Shared search call site for `handle_search` (dense, optionally post-filtered) and
`handle_ftsearch` (single-index text): runs the search and logs exactly one `op_log`
"search" record around it (PLAN.md §4.3), so cost accounting lives in one place instead of
at every search-shaped handler. Not used by `handle_hybrid_search`/`handle_ftsearch_group`,
which span more than one dataset's `op_log` -- attributing one combined operation across
several projects needs its own decision.

Everything goes through the engine's public `search`/`ftsearch` **except a per-request
beam-search override**, which the public surface does not expose (it takes `minrecall`, a
calibrated target, rather than raw beam parameters). That one path still calls
`IndexEngine.search_live`; it is the last search-side internal this module reaches for.
"""
function _run_search(handle, index::String, req::HTTP.Request, query, k::Int;
                     filter_spec=nothing, bs_override=nothing, text::Bool=false)
    engine = handle.engine
    dataset = handle.project
    # The cost of this one search, reported by the engine itself. It cannot be recovered by
    # snapshotting a context around the call the way `append`/`delete` do: a search runs on a
    # context borrowed from the engine's pool and returned at the end, so the counters a
    # caller can reach from out here are shared with whatever else is searching concurrently.
    stats = SSE.SearchStats()
    t0 = time()
    hits = if bs_override !== nothing
        raw = IndexEngine.search_live(engine, query, k;
                                      bs_override=bs_override, minrecall=nothing, policy=nothing)
        stats.distance_evaluations = raw.distance_evaluations
        _hydrate_raw(handle, raw)
    elseif text
        SSE.ftsearch(handle, query, k; stats)
    elseif filter_spec === nothing
        SSE.search(handle, query, k; stats)
    else
        SSE.search(handle, query, k; filter=_filter_predicate(filter_spec), candidates=FILTER_CANDIDATES(k), stats)
    end
    Telemetry.log_request!(dataset, nothing, "search", index, t0, nothing;
        token=_request_token(req), distance_name=_engine_distance_name(engine), dimension=_engine_dimension(engine),
        distance_evaluations=stats.distance_evaluations)
    return hits
end

"""
    _paginate_results(app, index, results, page_size) -> Dict

Materializes `results` behind a fresh `Cursors` cursor (PLAN.md §5.4) and returns its
first page plus the `cursor_id` for polling the rest via
`GET /api/v1/cursors/{cursor_id}`. Purely opt-in — a caller that never sends `page_size`
keeps getting the full `results` list inline, unchanged from before this existed.
"""
function _paginate_results(app::AppState, index::String, results::AbstractVector, page_size::Int)
    cursor_id = Cursors.create_cursor!(app.cursor_mgr, index, results; page_size=page_size)
    page = Cursors.poll_cursor!(app.cursor_mgr, cursor_id; limit=page_size)
    return Dict(
        "status" => "searched", "id" => index, "cursor_id" => cursor_id,
        "results" => page.results, "exhausted" => page.exhausted, "total" => page.total,
    )
end

function handle_search(req::HTTP.Request, app::AppState, index::String)
    haskey(app.handles, index) || return json_response(404, Dict("error" => "dataset_not_found"))
    engine = app.handles[index].engine
    dataset = app.handles[index].project

    data = JSON3.read(String(req.body), Dict{String, Any})
    haskey(data, "vector") || return json_response(400, Dict("error" => "search requires a 'vector' field"))
    k = get(data, "k", 10)
    query = convert(Vector{Float32}, data["vector"])
    filter_spec = get(data, "filter", nothing)

    bs_override = nothing
    extra_headers = []
    overrides = get(data, "beamsearch_overrides", nothing)
    if overrides !== nothing
        bs, warn_or_err = _resolve_beamsearch_override(engine, overrides)
        bs === nothing && return json_response(400, Dict("error" => warn_or_err))
        bs_override = bs
        if warn_or_err === true
            @warn "search on '$index' requested beamsearch_overrides above its calibrated baseline" index overrides
            extra_headers = ["X-Beamsearch-Warning" => "elevated-parameters-above-calibrated-baseline"]
        end
    end

    hits = try
        _run_search(app.handles[index], index, req, query, k; filter_spec, bs_override)
    catch e
        e isa SSE.EngineError || rethrow()
        return engine_error_response(e)
    end

    results, insufficient = _format_results(hits, k)

    page_size = get(data, "page_size", nothing)
    page_size !== nothing && return json_response(200, _paginate_results(app, index, results, Int(page_size)); extra_headers=extra_headers)

    body = Dict{String, Any}("status" => "searched", "id" => index, "results" => results)
    insufficient && (body["insufficient_results"] = true)
    return json_response(200, body; extra_headers=extra_headers)
end

"""
    _update_descriptor!(app, id, updates::Dict)

Reads a dataset's persistent sidecar, merges `updates` in, and writes it back — the same
read-merge-write shape `handle_create_dataset` uses at creation time, but for a
post-creation update (currently only `handle_calibrate`'s `beamsearch_baseline`).
"""
function _update_descriptor!(app::AppState, id::String, updates::Dict)
    dpath = descriptor_path(app, id)
    current = isfile(dpath) ? JSON3.read(read(dpath, String), Dict{String, Any}) : Dict{String, Any}("id" => id)
    merge!(current, updates)
    write(dpath, JSON3.write(current))
end

"""
    handle_calibrate(req, app, index) -> HTTP.Response

`POST /api/v1/simsearch/{index}/calibrate` (PLAN.md §5.6): runs `IndexEngine.calibrate!`'s
real `SearchModels`-driven hyperparameter sweep against `index`'s already-indexed data (or
an explicit `queries` ground-truth set, if given), installs the best-found `BeamSearch` as
the index's new default, and persists it into the dataset's descriptor. Only applies to a
`searchgraph` (approximate dense) index — exact/exhaustive indices have no `BeamSearch` to
tune, and text indices aren't in scope for this pass.

Body (all optional): `{"minrecall": 0.9, "numqueries": 64, "ksearch": 10, "queries": [[...], ...]}`.
"""
function handle_calibrate(req::HTTP.Request, app::AppState, index::String)
    haskey(app.handles, index) || return json_response(404, Dict("error" => "dataset_not_found"))
    engine = app.handles[index].engine

    engine.backend.index isa SearchGraph || return json_response(400, Dict(
        "error" => "calibrate requires a searchgraph (approximate dense) index"
    ))
    length(engine.backend.index) == 0 && return json_response(400, Dict("error" => "index has no data to calibrate against"))

    body = String(req.body)
    data = isempty(body) ? Dict{String, Any}() : JSON3.read(body, Dict{String, Any})
    minrecall = Float64(get(data, "minrecall", 0.9))
    numqueries = Int(get(data, "numqueries", 64))
    ksearch = Int(get(data, "ksearch", 10))
    queries = get(data, "queries", nothing)

    bs = SSE.calibrate!(app.handles[index]; levels=minrecall, numqueries, ksearch, queries)
    algo = bs[Float32(minrecall)]
    baseline = Dict("bsize" => Int(algo.bsize), "delta" => Float64(algo.Δ), "maxvisits" => Int(algo.maxvisits))
    _update_descriptor!(app, index, Dict("beamsearch_baseline" => baseline))

    return json_response(200, Dict("status" => "calibrated", "id" => index, "baseline" => baseline))
end

function handle_ftsearch(req::HTTP.Request, app::AppState, index::String)
    haskey(app.handles, index) || return json_response(404, Dict("error" => "dataset_not_found"))
    engine = app.handles[index].engine
    dataset = app.handles[index].project

    data = JSON3.read(String(req.body), Dict{String, Any})
    haskey(data, "text") || return json_response(400, Dict("error" => "ftsearch requires a 'text' field"))
    k = get(data, "k", 10)

    hits = try
        _run_search(app.handles[index], index, req, data["text"], k; text=true)
    catch e
        e isa SSE.EngineError || rethrow()
        return engine_error_response(e)
    end
    results, insufficient = _format_results(hits, k)
    body = Dict{String, Any}("status" => "searched", "id" => index, "results" => results)
    insufficient && (body["insufficient_results"] = true)
    return json_response(200, body)
end

"""
    handle_delete_item(req, app, index)

Soft-deletes a document (PLAN.md §1): marks its `doc_id` in the engine's tombstone set
so future searches exclude it, without removing it from the underlying index.

Also resaves the dataset's JLD2 snapshot, mirroring `handle_append`'s reasoning: a CLI
`rebuild` (which is what actually purges tombstoned items) runs as a separate subprocess
that only ever sees on-disk state, so a tombstone that lives purely in this process's
in-memory `engine.deleted_ids` would be invisible to it. Without this resave, `rebuild`
would have no way to know which documents were ever soft-deleted.
"""
function handle_delete_item(req::HTTP.Request, app::AppState, index::String)
    haskey(app.handles, index) || return json_response(404, Dict("error" => "dataset_not_found"))
    data = JSON3.read(String(req.body), Dict{String, Any})
    haskey(data, "doc_id") || return json_response(400, Dict("error" => "delete requires a 'doc_id' field"))

    doc_id = Int(data["doc_id"])
    # `delete_item!` persists the tombstone itself; the JLD2 snapshot this used to write
    # afterwards existed only because the engine did not persist its own state back then.
    try
        SSE.delete_item!(app.handles[index], doc_id)
    catch e
        e isa SSE.EngineError || rethrow()
        return engine_error_response(e)
    end

    # No ctx/timing ceremony here (unlike search/append) -- a soft-delete doesn't evaluate
    # any distances, so log_operation directly rather than through log_request!.
    Telemetry.log_operation(app.handles[index].project, "delete", Dict{String, Any}("index_uuid" => index, "doc_id" => doc_id, "token" => _request_token(req)))

    return json_response(200, Dict("status" => "soft_deleted", "id" => index, "doc_id" => doc_id))
end

"""
    handle_fetch(req, app, index)

Batch metadata retrieval by id (PLAN.md §5.1). Accepts either the internal integer
`doc_id` or the caller-supplied original `id` (resolved via a linear scan — see
`Project.find_by_doc_id`; there is no secondary index from external id to `doc_id`
in this pass).
"""
function handle_fetch(req::HTTP.Request, app::AppState, index::String)
    haskey(app.handles, index) || return json_response(404, Dict("error" => "dataset_not_found"))
    dataset = app.handles[index].project

    data = JSON3.read(String(req.body), Dict{String, Any})
    ids = get(data, "ids", [])

    results = Any[]
    for raw_id in ids
        record = nothing
        maybe_int = tryparse(Int, string(raw_id))
        if maybe_int !== nothing
            record = Project.get_metadata(dataset, maybe_int)
        end
        record === nothing && (record = Project.find_by_doc_id(dataset, string(raw_id)))
        record === nothing && continue

        meta = Project.get_meta(dataset, record._id; lazy=false)
        meta_dict = meta isa AbstractDict ? Dict{String, Any}(string(k) => v for (k, v) in meta) : Dict{String, Any}()
        entry = merge(Dict{String, Any}(
            "id" => something(record.doc_id, record._id),
            "doc_id" => record.doc_id,
            "_id" => record._id,
            "keywords" => record.keywords,
            "refs" => record.refs,
            "schema_version" => record.schema_version
        ), meta_dict)
        push!(results, entry)
    end

    return json_response(200, Dict("status" => "fetched", "id" => index, "results" => results))
end

"""
    rrf_fuse(rank_lists; k=60) -> (ids, scores)

Reciprocal Rank Fusion (PLAN.md §6): combines several already-ranked (best-first) id
lists into one score per id, `sum(1 / (rank + k))` across the lists it appears in.
"""
function rrf_fuse(rank_lists::Vector; k::Int=60)
    scores = Dict{Int32, Float64}()
    for ranks in rank_lists
        for (i, id) in enumerate(ranks)
            scores[id] = get(scores, id, 0.0) + 1.0 / (i + k)
        end
    end
    ids = sort(collect(keys(scores)); by=id -> -scores[id])
    return ids, scores
end

"""
    handle_hybrid_search(req, app)

Fuses a dense (`dense_index`) and lexical (`lexical_index`) search over the same shared
`doc_id` space via RRF (PLAN.md §6), hydrating metadata once at the end from whichever of
the two datasets actually has it for a given id.
"""
function handle_hybrid_search(req::HTTP.Request, app::AppState)
    data = JSON3.read(String(req.body), Dict{String, Any})
    dense_id = get(data, "dense_index", nothing)
    lexical_id = get(data, "lexical_index", nothing)
    (dense_id === nothing || lexical_id === nothing) &&
        return json_response(400, Dict("error" => "hybrid_search requires 'dense_index' and 'lexical_index'"))
    (haskey(app.handles, dense_id) && haskey(app.handles, lexical_id)) ||
        return json_response(404, Dict("error" => "dataset_not_found"))

    k = get(data, "k", 10)

    dense_hits = try
        haskey(data, "vector") ? SSE.search(app.handles[dense_id], convert(Vector{Float32}, data["vector"]), k) : SSE.SearchResult[]
    catch e
        e isa SSE.EngineError || rethrow()
        return engine_error_response(e)
    end
    lexical_hits = try
        haskey(data, "text") ? SSE.ftsearch(app.handles[lexical_id], data["text"], k) : SSE.SearchResult[]
    catch e
        e isa SSE.EngineError || rethrow()
        return engine_error_response(e)
    end

    fused_ids, scores = rrf_fuse([Int32[h._id for h in dense_hits], Int32[h._id for h in lexical_hits]])
    top_ids = first(fused_ids, min(k, length(fused_ids)))

    results = Any[]
    for id in top_ids
        record = Project.get_metadata(app.handles[dense_id].project, id)
        record === nothing && (record = Project.get_metadata(app.handles[lexical_id].project, id))
        orig_id = record === nothing ? string(id) : something(record.doc_id, string(id))
        push!(results, Dict("id" => orig_id, "doc_id" => id, "score" => scores[id]))
    end

    return json_response(200, Dict("status" => "searched", "results" => results))
end

"""
    _hydrate_join_id(app, ordered_members, doc_id) -> String

Resolves `doc_id`'s original/external id by checking each of `ordered_members` in turn
until one has metadata for it — the same "hydrate once, from whichever member actually
has it" idea `handle_hybrid_search` already applies ad hoc between exactly two datasets,
generalized to however many members a join group has. Callers sort `ordered_members` so
the group's `holds_metadata` member (if any) is tried first.
"""
function _hydrate_join_id(app::AppState, ordered_members::Vector, doc_id)
    for m in ordered_members
        haskey(app.handles, m["id"]) || continue
        record = Project.get_metadata(app.handles[m["id"]].project, doc_id)
        record !== nothing && return something(record.doc_id, string(doc_id))
    end
    return string(doc_id)
end

"""
    handle_ftsearch_group(req, app)

`POST /api/v1/simsearch/ftsearch` (PLAN.md §5.3/§7): join-group-aware full-text search,
distinct from the simpler per-index `POST /api/v1/simsearch/{index}/ftsearch` (still
supported unchanged, e.g. by `test_02_full_text_search.jl`). Takes a `join_group` and a
`key`: either one specific key string (routes to the one text-kind member tagged with
that key) or the reserved wildcard `"*"` (fans out to every text-kind member of the
group). Responses for `key: "*"` are grouped by key, not fused into one ranked list —
PLAN.md is explicit that per-field lexical fusion is a different, harder scoring problem
out of scope here; a caller wanting a single list across fields does that client-side.
Metadata is hydrated once at the end via `_hydrate_join_id`, preferring the group's
`holds_metadata` member when one is tagged.
"""
function handle_ftsearch_group(req::HTTP.Request, app::AppState)
    data = JSON3.read(String(req.body), Dict{String, Any})
    join_group = get(data, "join_group", nothing)
    key = get(data, "key", nothing)
    (join_group === nothing || key === nothing) &&
        return json_response(400, Dict("error" => "ftsearch requires 'join_group' and 'key' ('*' to fan out to every text member)"))
    haskey(data, "text") || return json_response(400, Dict("error" => "ftsearch requires a 'text' field"))
    join_group = string(join_group)
    key = string(key)
    k = get(data, "k", 10)

    members = join_group_members(app, join_group)
    isempty(members) && return json_response(404, Dict("error" => "join_group '$join_group' not found or has no members"))

    text_members = filter(members) do m
        haskey(app.handles, m["id"]) && SSE.payload_kind(app.handles[m["id"]].engine) === :text
    end
    key != "*" && (text_members = filter(m -> get(m, "key", nothing) == key, text_members))
    isempty(text_members) && return json_response(404, Dict(
        "error" => key == "*" ? "join_group '$join_group' has no text members" : "no text member of join_group '$join_group' is tagged with key '$key'"
    ))

    # Try the holds_metadata member first, regardless of which member's search a given
    # doc_id actually came from.
    ordered_members = sort(members; by = m -> get(m, "holds_metadata", false) === true ? 0 : 1)

    grouped = Dict{String, Any}()
    for m in text_members
        hits = try
            SSE.ftsearch(app.handles[m["id"]], data["text"], k)
        catch e
            e isa SSE.EngineError || rethrow()
            return engine_error_response(e)
        end
        entries = Any[]
        for hit in hits
            hit.deleted && continue
            orig_id = _hydrate_join_id(app, ordered_members, hit._id)
            push!(entries, Dict("id" => orig_id, "doc_id" => hit._id, "score" => hit.distance))
        end
        grouped[something(get(m, "key", nothing), m["id"])] = entries
    end

    return json_response(200, Dict("status" => "searched", "join_group" => join_group, "results" => grouped))
end

# ==========================================
# Jobs Endpoints
# ==========================================

"""
    HEAVY_JOB_KINDS

Job kinds this pass actually implements as real CLI subcommands (`src/cli_handlers.jl`'s
`execute_allknn`/`execute_fft`/`execute_neardup`/`execute_hsp`/`execute_dump`/
`execute_load`), as opposed to arbitrary kinds only reachable via an explicit `"command"`
array (see `handle_submit_job`). `dump`/`load` complete PLAN.md §5.5's job-kind enum --
chunk 10 shipped them as CLI-only; this pass adds the `POST /api/v1/jobs/dump|load`
surface on top of the same `execute_dump`/`execute_load` CLI functions, unchanged.
"""
const HEAVY_JOB_KINDS = ("allknn", "fft", "neardup", "hsp", "dump", "load")

"""
    build_heavy_job_command(kind, data, output_path, workdir) -> Union{Vector{String}, Nothing}

Translates a friendly JSON job-submission body (`{"dataset": ..., "k": ...}`) into the
exact CLI args `Executors.submit` will spawn for one of `HEAVY_JOB_KINDS`. Returns
`nothing` if a required field is missing, which the caller turns into a `400`.
"""
function build_heavy_job_command(kind::String, data::AbstractDict, output_path::String, workdir::String)
    haskey(data, "dataset") || return nothing
    dataset = string(data["dataset"])

    if kind == "allknn"
        k = string(get(data, "k", 10))
        return ["allknn", "--dataset", dataset, "--k", k, "--output", output_path, "--workdir", workdir]
    elseif kind == "fft"
        haskey(data, "k") || return nothing
        return ["fft", "--dataset", dataset, "--k", string(data["k"]), "--output", output_path, "--workdir", workdir]
    elseif kind == "neardup"
        haskey(data, "epsilon") || return nothing
        return ["neardup", "--dataset", dataset, "--epsilon", string(data["epsilon"]), "--output", output_path, "--workdir", workdir]
    elseif kind == "hsp"
        haskey(data, "queries") || return nothing
        k = string(get(data, "k", 10))
        return ["hsp", "--dataset", dataset, "--queries", string(data["queries"]), "--k", k, "--output", output_path, "--workdir", workdir]
    elseif kind == "dump"
        # `dataset` here is the source dataset being exported. `output_path` (computed by
        # the caller, guaranteed not to already exist) doubles as the bundle *directory*
        # `execute_dump` creates -- unlike every other HEAVY_JOB_KINDS result, this one is
        # a directory of 3 files (dataset.avro/index.snapshot.jld2/manifest.json), not a
        # single file (see `handle_get_job_result`'s `isdir` branch).
        return ["dump", "--dataset", dataset, "--workdir", workdir, "--output", output_path]
    elseif kind == "load"
        # `dataset` here is the *target* dataset id being created (must not already exist
        # under `workdir` -- `execute_load` itself enforces that). `output_path` is unused;
        # the real result is the new dataset directory itself (see `handle_submit_job`'s
        # kind == "load" branch for its `result_ref`).
        haskey(data, "bundle") || return nothing
        return ["load", "--bundle", string(data["bundle"]), "--dataset", dataset, "--workdir", workdir]
    else
        return nothing
    end
end

"""
    handle_submit_job(req, app, kind)

Enqueues a job (PLAN.md §5.5). Actual dispatch (`queued/` -> `running/` ->
`completed/`/`failed/`) is owned exclusively by the background loop started alongside the
server (`Executors.run_dispatcher!`), never by this handler.

Two ways to submit, in priority order: (1) an explicit `"command"` array in the body is
always honored as-is (the original, low-level escape hatch — `test_e2e.jl`'s `build` job
relies on this); (2) for `kind in HEAVY_JOB_KINDS`, a friendly params body
(`dataset`/`k`/`epsilon`/`queries`) is translated into the real CLI invocation via
`build_heavy_job_command`, with a `result_ref` pointing at the output file the CLI
subprocess will write passed into `Jobs.create_job!`'s `extra` so it's part of the job's
very first write (see that function's docstring for why a separate post-creation update
would race the dispatcher). `Jobs.finish_job!`'s `merge!`-based content update preserves
it through every later state transition, so `GET /api/v1/jobs/{id}/result` can serve it
once `status == completed` with no further plumbing. Anything else is a `400`.
"""
function handle_submit_job(req::HTTP.Request, app::AppState, kind::String)
    body = String(req.body)
    data = isempty(body) ? Dict{String, Any}() : JSON3.read(body, Dict{String, Any})

    if haskey(data, "command")
        command = String.(data["command"])
        job_id = Jobs.create_job!(app.job_mgr, kind, command)
        return json_response(202, Dict("status" => "accepted", "job_id" => job_id))
    end

    kind in HEAVY_JOB_KINDS || return json_response(400, Dict(
        "error" => "unsupported job kind '$kind' -- pass an explicit 'command' array, or use one of: $(join(HEAVY_JOB_KINDS, ", "))"
    ))

    results_dir = joinpath(app.workdir, "jobs", "results")
    mkpath(results_dir)
    job_id = Project.generate_id()
    output_path = joinpath(results_dir, "$(job_id).$(kind).result")

    command = build_heavy_job_command(kind, data, output_path, app.workdir)
    command === nothing && return json_response(400, Dict("error" => "missing required params for job kind '$kind'"))

    # `load`'s real result is the new dataset directory it creates, not `output_path`
    # (which `build_heavy_job_command` never even puts in the `load` command) -- point
    # `result_ref` at its `descriptor.json` so `GET .../result` can confirm what landed.
    result_ref = kind == "load" ? joinpath(app.workdir, "datasets", string(data["dataset"]), "descriptor.json") : output_path

    Jobs.create_job!(app.job_mgr, kind, command; id=job_id, extra=Dict("result_ref" => result_ref))

    return json_response(202, Dict("status" => "accepted", "job_id" => job_id))
end

"""
    JOB_STATE_BY_NAME

Lowercase-name lookup for `Jobs.JobState`, so `handle_list_jobs`'s `?status=` query
param can resolve to the enum without hand-rolling a string-match chain.
"""
const JOB_STATE_BY_NAME = Dict(lowercase(string(s)) => s for s in instances(Jobs.JobState))

"""
    handle_list_jobs(req, app) -> HTTP.Response

`GET /api/v1/jobs?status=&kind=&offset=&limit=` (PLAN.md §5.4/§5.5): lists jobs across
every spool state, optionally filtered to one `status` and/or one `kind`, paginated with
plain `offset`/`limit` (§5.4) — cheap enough to recompute per request that it doesn't
need `Cursors`. Doesn't support PLAN.md's `index_uuid` filter: job records in this pass
don't track which dataset/index they targeted as a queryable field, and adding that is
its own separate bookkeeping change, not part of this endpoint.
"""
function handle_list_jobs(req::HTTP.Request, app::AppState)
    params = queryparams(HTTP.URI(req.target))

    states = (Jobs.Queued, Jobs.Running, Jobs.Blocked, Jobs.Completed, Jobs.Failed)
    if haskey(params, "status")
        haskey(JOB_STATE_BY_NAME, lowercase(params["status"])) || return json_response(400, Dict("error" => "unknown status '$(params["status"])'"))
        states = (JOB_STATE_BY_NAME[lowercase(params["status"])],)
    end
    kind_filter = get(params, "kind", nothing)

    entries = Any[]
    for state in states
        for id in Jobs.list_jobs(app.job_mgr, state)
            _, record = Jobs.get_job(app.job_mgr, id)
            record === nothing && continue
            kind_filter !== nothing && get(record, "kind", nothing) != kind_filter && continue
            push!(entries, Dict("id" => id, "kind" => get(record, "kind", nothing), "status" => lowercase(string(state))))
        end
    end

    total = length(entries)
    offset, limit = _parse_offset_limit(params, total)
    page = entries[(min(offset, total) + 1):min(offset + limit, total)]

    return json_response(200, Dict("status" => "ok", "jobs" => page, "total" => total, "offset" => offset, "limit" => limit))
end

function handle_get_job(req::HTTP.Request, app::AppState, job_id::String)
    state, record = Jobs.get_job(app.job_mgr, job_id)
    state === nothing && return json_response(404, Dict("error" => "not_found"))
    return json_response(200, Dict("status" => lowercase(string(state)), "job" => record))
end

"""
    handle_get_job_result(req, app, job_id)

`GET /api/v1/jobs/{job_id}/result` (PLAN.md §5.5): streams back the raw file a completed
heavy job's CLI subprocess wrote (see `handle_submit_job`'s `result_ref` bookkeeping).
`409` if the job hasn't reached `completed` yet, `404` if there's no `result_ref` at all
(e.g. a job submitted via the raw `"command"` escape hatch, which doesn't set one).

A `dump` job's `result_ref` is a bundle *directory* (`dataset.avro`/
`index.snapshot.jld2`/`manifest.json`), not a single file -- unlike every other
`HEAVY_JOB_KINDS` result, it can't be streamed back as one HTTP body (and the bundle is
meant to stay on the shared local filesystem the `LocalCLIExecutor`/operator already has
access to, not leave it over HTTP). For that case this returns a small JSON pointer
(`bundle_dir` + the parsed `manifest.json`) instead of raw bytes; a `load` job's
`result_ref` (the new dataset's `descriptor.json`) is a plain file and falls through to
the normal byte-streaming path unchanged.
"""
function handle_get_job_result(req::HTTP.Request, app::AppState, job_id::String)
    state, record = Jobs.get_job(app.job_mgr, job_id)
    state === nothing && return json_response(404, Dict("error" => "not_found"))
    state === Jobs.Completed || return json_response(409, Dict("error" => "job not completed", "status" => lowercase(string(state))))

    result_ref = get(record, "result_ref", nothing)
    (result_ref === nothing || !ispath(result_ref)) && return json_response(404, Dict("error" => "result not found"))

    if isdir(result_ref)
        manifest_path = joinpath(result_ref, "manifest.json")
        manifest = isfile(manifest_path) ? JSON3.read(read(manifest_path, String), Dict{String, Any}) : nothing
        return json_response(200, Dict("bundle_dir" => result_ref, "manifest" => manifest))
    end

    return HTTP.Response(200, JSON_HEADERS, read(result_ref))
end

"""
    handle_block_job(req, app, job_id) -> HTTP.Response

`POST /api/v1/jobs/{job_id}/block` (PLAN.md §5.5, admin-only: `similarity-search-ctl
jobs block`): moves a *queued* job to `blocked/`, where the dispatcher never scans for
work -- distinct from `kill` (stops something already running) and the best-effort
`DELETE`. `404` if the job doesn't exist; `409` if it's not currently `queued` (e.g.
already running, or already blocked) since the request landed too late for a "pause
before it starts" operation to make sense.
"""
function handle_block_job(req::HTTP.Request, app::AppState, job_id::String)
    state, _ = Jobs.get_job(app.job_mgr, job_id)
    state === nothing && return json_response(404, Dict("error" => "not_found"))
    state === Jobs.Queued || return json_response(409, Dict("error" => "job is not queued", "status" => lowercase(string(state))))
    Jobs.update_job_state!(app.job_mgr, job_id, Jobs.Queued, Jobs.Blocked)
    return json_response(200, Dict("status" => "blocked", "job_id" => job_id))
end

"""
    handle_resume_job(req, app, job_id) -> HTTP.Response

`POST /api/v1/jobs/{job_id}/resume` (PLAN.md §5.5): the inverse of `handle_block_job` --
moves a `blocked` job back to `queued/`. `409` if it isn't currently `blocked`.
"""
function handle_resume_job(req::HTTP.Request, app::AppState, job_id::String)
    state, _ = Jobs.get_job(app.job_mgr, job_id)
    state === nothing && return json_response(404, Dict("error" => "not_found"))
    state === Jobs.Blocked || return json_response(409, Dict("error" => "job is not blocked", "status" => lowercase(string(state))))
    Jobs.update_job_state!(app.job_mgr, job_id, Jobs.Blocked, Jobs.Queued)
    return json_response(200, Dict("status" => "queued", "job_id" => job_id))
end

"""
    handle_kill_job(req, app, job_id) -> HTTP.Response

`POST /api/v1/jobs/{job_id}/kill` (PLAN.md §5.5, admin-only): forcibly terminates a
*running* job via the `JobExecutor`'s `kill_job!(executor, handle)` (§5.5's `kill(handle)`
abstraction) and marks it `failed` with `error: "killed by operator"`. Unlike `DELETE`
(best-effort, only for a not-yet-started job), this is the explicit, always-available
override for a job already consuming the reserved compute pool. `409` if the job isn't
currently `running` (nothing to kill) -- including the race where the dispatcher's own
poll loop finishes the job (naturally, right as this request lands) before `finish_job!`
below gets to run: that raises inside `Jobs.update_job_content!` (the record is no longer
in `running/`), caught here and reported as the same `409` rather than a `500`.
"""
function handle_kill_job(req::HTTP.Request, app::AppState, job_id::String)
    state, record = Jobs.get_job(app.job_mgr, job_id)
    state === nothing && return json_response(404, Dict("error" => "not_found"))
    state === Jobs.Running || return json_response(409, Dict("error" => "job is not running", "status" => lowercase(string(state))))

    handle = get(record, "executor_handle", "")
    Executors.kill_job!(app.executor, handle)
    try
        Jobs.finish_job!(app.job_mgr, job_id, false; error="killed by operator")
    catch
        return json_response(409, Dict("error" => "job finished before it could be killed"))
    end
    return json_response(200, Dict("status" => "killed", "job_id" => job_id))
end

"""
    handle_delete_job(req, app, job_id) -> HTTP.Response

`DELETE /api/v1/jobs/{job_id}` (PLAN.md §5.5): best-effort cancellation of a job that
hasn't started running yet -- moves it straight to `failed/` with `error: "cancelled by
operator"` via `Jobs.cancel_job!`. Distinct from `kill` (`POST .../kill`), the only way to
stop an already-`running` job: `DELETE` against a running or already-finished job is a
`409`, not a silent no-op and not a fallback to killing it (PLAN.md §1's own note
explicitly separates the two). Guards the same dispatcher race `handle_kill_job` already
does above (the job transitions `queued/` -> `running/` in the same instant this request
lands) with a `try`/`catch`, reported as the same `409` rather than a `500`.
"""
function handle_delete_job(req::HTTP.Request, app::AppState, job_id::String)
    result = try
        Jobs.cancel_job!(app.job_mgr, job_id)
    catch
        :not_cancellable
    end
    result === :not_found && return json_response(404, Dict("error" => "not_found"))
    result === :not_cancellable && return json_response(409, Dict("error" => "job is already running or finished -- use kill to terminate a running job"))
    return json_response(200, Dict("status" => "cancelled", "job_id" => job_id))
end

"""
    handle_jobs_gc(req, app) -> HTTP.Response

`POST /api/v1/admin/jobs/gc` (PLAN.md §5.5/§1's `similarity-search-ctl jobs gc`, and the
note that job GC and result-cursor GC share one command since they're the same kind of
spool-record lifecycle): sweeps `open/` cursors whose TTL lapsed into `expired/`
(`Cursors.gc_expired!`), then purges every file now sitting in `expired/`
(`Cursors.purge_expired!`) and every `completed`/`failed` job record older than
`retention_seconds` (`Jobs.purge_finished!`, default 24h, overridable in the body --
`{"retention_seconds": 3600}` -- since a fixed default would make this awkward to exercise
in a hardened test without actually waiting a day).
"""
function handle_jobs_gc(req::HTTP.Request, app::AppState)
    body = String(req.body)
    data = isempty(body) ? Dict{String, Any}() : JSON3.read(body, Dict{String, Any})
    retention_seconds = get(data, "retention_seconds", 86400)

    cursors_expired = Cursors.gc_expired!(app.cursor_mgr)
    cursors_purged = Cursors.purge_expired!(app.cursor_mgr)
    jobs_purged = Jobs.purge_finished!(app.job_mgr; retention_seconds=retention_seconds)

    return json_response(200, Dict(
        "status" => "ok",
        "cursors_expired" => cursors_expired,
        "cursors_purged" => cursors_purged,
        "jobs_purged" => jobs_purged,
    ))
end

# ==========================================
# Cursor Endpoints
# ==========================================

"""
    handle_poll_cursor(req, app, cursor_id) -> HTTP.Response

`GET /api/v1/cursors/{cursor_id}?limit=N` (PLAN.md §5.4): returns the next page of a
result cursor's materialized results and advances it (see `handle_search`'s opt-in
`page_size` param for how a cursor gets created in the first place). `404` if the cursor
never existed at all; `409` if it did but isn't `open` anymore (already `exhausted` or
`expired`) — the same "not found" vs. "found but not in the right state" distinction
`handle_get_job_result` already draws for jobs.
"""
function handle_poll_cursor(req::HTTP.Request, app::AppState, cursor_id::String)
    state, _ = Cursors.get_cursor(app.cursor_mgr, cursor_id)
    state === nothing && return json_response(404, Dict("error" => "not_found"))
    state === Cursors.Open || return json_response(409, Dict("error" => "cursor is $(lowercase(string(state))), nothing more to poll"))

    params = queryparams(HTTP.URI(req.target))
    limit = haskey(params, "limit") ? tryparse(Int, params["limit"]) : nothing

    page = Cursors.poll_cursor!(app.cursor_mgr, cursor_id; limit=limit)
    page === nothing && return json_response(409, Dict("error" => "cursor is no longer open"))

    return json_response(200, Dict(
        "cursor_id" => cursor_id, "results" => page.results,
        "exhausted" => page.exhausted, "total" => page.total, "offset" => page.offset,
    ))
end

# ==========================================
# Token Admin Endpoints
# ==========================================

function handle_create_token(req::HTTP.Request, app::AppState)
    data = JSON3.read(String(req.body), Dict{String, Any})
    user = get(data, "user", "anonymous")
    # `String.(...)` over an empty `Vector{Any}` (no permissions at all) stays a `Vector{Any}`,
    # which `create_token!` does not accept -- collect into the element type instead.
    permissions = collect(String, get(data, "permissions", String[]))
    expires_at = get(data, "expires_at", nothing)

    token_str = Tokens.create_token!(app.token_mgr, user, permissions; expires_at=expires_at)
    return json_response(201, Dict("status" => "created", "token" => token_str))
end

function handle_revoke_token(req::HTTP.Request, app::AppState, token::String)
    Tokens.revoke_token!(app.token_mgr, token)
    return json_response(200, Dict("status" => "revoked"))
end

"""
    handle_list_tokens(req, app) -> HTTP.Response

`GET /api/v1/admin/tokens` (backs `similarity-search-ctl log-tokens`, PLAN.md §1's admin
command surface): an audit listing of every token currently stored, expired or not --
`token`/`user`/`permissions`/`created_at`/`expires_at` for each, same fields `add-token`'s
own response already exposes in plaintext (this is a local admin/control-plane surface,
not a public API -- see PLAN.md §1's note on how `-ctl` reaches it).
"""
function handle_list_tokens(req::HTTP.Request, app::AppState)
    tokens = [Dict(
        "token" => rec.token_str, "user" => rec.user, "permissions" => rec.permissions,
        "created_at" => rec.created_at, "expires_at" => rec.expires_at,
    ) for rec in Tokens.list_tokens(app.token_mgr)]
    return json_response(200, Dict("status" => "ok", "tokens" => tokens))
end

"""
    handle_prune_tokens(req, app) -> HTTP.Response

`POST /api/v1/admin/tokens/prune` (backs `similarity-search-ctl prune-tokens`): revokes
every token whose `expires_at` has already passed, via `Tokens.prune_expired!`.
"""
function handle_prune_tokens(req::HTTP.Request, app::AppState)
    pruned = Tokens.prune_expired!(app.token_mgr)
    return json_response(200, Dict("status" => "ok", "pruned" => pruned))
end

# ==========================================
# Router Setup
# ==========================================

"""
    run_server(host::String, port::Int, app::AppState; async::Bool=false)

Registers all HTTP routes and starts the Oxygen.jl server.

# Arguments
- `host::String`: The host to bind the server to (e.g., "127.0.0.1").
- `port::Int`: The port to listen on.
- `app::AppState`: The global application state.
- `async::Bool`: If `true`, returns immediately instead of blocking (used by tests).
"""
function run_server(host::String, port::Int, app::AppState; async::Bool=false)
    @get "/healthz" req -> handle_healthz(req)
    @get "/readyz" req -> handle_readyz(req, app)
    @get "/metrics" req -> handle_metrics(req, app)

    @post "/api/v1/datasets" req -> handle_create_dataset(req, app)
    @get "/api/v1/datasets" req -> handle_list_datasets(req, app)
    @get "/api/v1/datasets/{id}/join_group" (req, id) -> handle_get_join_group(req, app, id)
    @get "/api/v1/datasets/{id}/log" (req, id) -> handle_get_op_log(req, app, id)
    @get "/api/v1/datasets/{id}" (req, id) -> handle_get_dataset(req, app, id)
    @delete "/api/v1/datasets/{id}" (req, id) -> handle_delete_dataset(req, app, id)

    @post "/api/v1/simsearch/hybrid_search" req -> handle_hybrid_search(req, app)
    @post "/api/v1/simsearch/ftsearch" req -> handle_ftsearch_group(req, app)
    @post "/api/v1/simsearch/{index}/append" (req, index) -> handle_append(req, app, index)
    @post "/api/v1/simsearch/{index}/search" (req, index) -> handle_search(req, app, index)
    @post "/api/v1/simsearch/{index}/calibrate" (req, index) -> handle_calibrate(req, app, index)
    @post "/api/v1/simsearch/{index}/ftsearch" (req, index) -> handle_ftsearch(req, app, index)
    @post "/api/v1/simsearch/{index}/delete" (req, index) -> handle_delete_item(req, app, index)
    @post "/api/v1/simsearch/{index}/fetch" (req, index) -> handle_fetch(req, app, index)
    @get "/api/v1/simsearch/{index}/exists" (req, index) -> handle_exists(req, app, index)

    @post "/api/v1/jobs/{kind}" (req, kind) -> handle_submit_job(req, app, kind)
    @post "/api/v1/jobs/{job_id}/block" (req, job_id) -> handle_block_job(req, app, job_id)
    @post "/api/v1/jobs/{job_id}/resume" (req, job_id) -> handle_resume_job(req, app, job_id)
    @post "/api/v1/jobs/{job_id}/kill" (req, job_id) -> handle_kill_job(req, app, job_id)
    @delete "/api/v1/jobs/{job_id}" (req, job_id) -> handle_delete_job(req, app, job_id)
    @get "/api/v1/jobs/{job_id}/result" (req, job_id) -> handle_get_job_result(req, app, job_id)
    @get "/api/v1/jobs/{job_id}" (req, job_id) -> handle_get_job(req, app, job_id)
    @get "/api/v1/jobs" req -> handle_list_jobs(req, app)

    @get "/api/v1/cursors/{cursor_id}" (req, cursor_id) -> handle_poll_cursor(req, app, cursor_id)

    @post "/api/v1/admin/tokens" req -> handle_create_token(req, app)
    @get "/api/v1/admin/tokens" req -> handle_list_tokens(req, app)
    @post "/api/v1/admin/tokens/prune" req -> handle_prune_tokens(req, app)
    @delete "/api/v1/admin/tokens/{token}" (req, token) -> handle_revoke_token(req, app, token)
    @post "/api/v1/admin/jobs/gc" req -> handle_jobs_gc(req, app)
    @post "/api/v1/admin/datasets/{id}/unload" (req, id) -> handle_unload_dataset(req, app, id)
    @post "/api/v1/admin/datasets/{id}/reload" (req, id) -> handle_reload_dataset(req, app, id)

    println("Starting SimilaritySearchServer at http://$host:$port")
    serve(host=host, port=port, async=async)
end


"""
    parse_index_kind(kind::AbstractString) -> (engine_type, backend_type)

The wire's name for a kind of project, as the pair the engine actually takes.

A project is two independent choices now -- what it holds (`DenseEngine`, `SparseEngine`,
`FullTextEngine`) and what indexes it (`SearchGraph`, `BM25InvertedFile`, ...) -- while this
API has always had a single `index_kind` string. The old names keep working and map to the
pair they always meant; `sparse_invfile` is new, and is the one that also needs `dimension`.
Raises [`SSE.UnknownBackend`](@ref), so a handler reports it the same way it reports any other
invalid request.
"""
function parse_index_kind(kind::AbstractString)
    kind == "searchgraph" && return (SSE.DenseEngine, SimilaritySearch.SearchGraph)
    kind == "exhaustive_search" && return (SSE.DenseEngine, SimilaritySearch.ExhaustiveSearch)
    kind == "parallel_exhaustive_search" && return (SSE.DenseEngine, SimilaritySearch.ParallelExhaustiveSearch)
    kind == "bm25_invfile" && return (SSE.FullTextEngine, TextSearch.BM25InvertedFile)
    kind == "invfile" && return (SSE.FullTextEngine, TextSearch.TextInvertedFile)
    kind == "sparse_invfile" && return (SSE.SparseEngine, TextSearch.InvertedFile)
    throw(SSE.UnknownBackend("unknown index kind $(repr(kind)); expected searchgraph, " *
        "exhaustive_search, parallel_exhaustive_search, bm25_invfile, invfile or sparse_invfile"))
end

"The directory every dataset of this server lives under."
datasets_root(app::AppState) = joinpath(app.workdir, "datasets")

"""
    default_textmodel(engine_type) -> Union{Nothing,AbstractTextModelSpec}

What a project of this kind is created with when the request does not say.

A text project must state its text model explicitly -- the engine refuses to guess, because
the ways to obtain a vocabulary are not interchangeable -- so this server picks
`FitFromCorpus`, which is what it has always used: fit from whatever the dataset itself is
given. Anything else needs a profile shipped to the server, which the API does not do yet.
"""
default_textmodel(engine_type) =
    engine_type === SSE.FullTextEngine ? SSE.FitFromCorpus(TextSearch.TextConfig()) : nothing

"""
    engine_error_response(e::SSE.EngineError) -> HTTP.Response

The whole error boundary, in one table: an engine error becomes a status code by its
*category*, never by its message.

- [`SSE.NotFound`](@ref) -> 404. Something named does not exist.
- [`SSE.ConflictingState`](@ref) -> 409. The project is real and the request is fine, but the
  project is not in a state that can serve it (a staged backlog, an untrained profile).
  Retrying after the state changes works, which is what makes this a 409 and not a 400.
- [`SSE.InvalidRequest`](@ref) -> 400. Retrying unchanged cannot help.
- [`SSE.StorageFailure`](@ref) -> 500. Not the caller's fault.

The body carries `error` (the message, for a human) and `kind` (the concrete type name, for a
client that wants to branch without parsing prose).
"""
function engine_error_response(e::SSE.EngineError)
    status = e isa SSE.NotFound ? 404 :
             e isa SSE.ConflictingState ? 409 :
             e isa SSE.InvalidRequest ? 400 : 500
    json_response(status, Dict("error" => sprint(showerror, e), "kind" => string(nameof(typeof(e)))))
end

function parse_distance(dist::Union{AbstractString,Nothing})
    dist === nothing && return nothing
    d = lowercase(dist)
    d == "l2" && return SimilaritySearch.Dist.SqL2() # Use SqL2 everywhere L2 was asked for
    d == "sql2" && return SimilaritySearch.Dist.SqL2()
    d == "cosine" && return SimilaritySearch.Dist.Cosine()
    d == "normcosine" && return SimilaritySearch.Dist.NormCosine()
    d == "angle" && return SimilaritySearch.Dist.Angle()
    d == "normangle" && return SimilaritySearch.Dist.NormAngle()
    d == "jaccard" && return SimilaritySearch.Dist.Jaccard()
    d == "" && return nothing # Some places pass empty string
    error("Unknown distance: $dist")
end
end # module
