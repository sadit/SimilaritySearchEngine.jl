# Friendly, function-per-operation embedded API (PLAN.md §8.5, chunk 22): a script running
# on the same machine against the same --workdir a similarity-search/similarity-search-serve
# process already uses, with no HTTP, no CLI subprocess, no running server required. Every
# function here mirrors an existing cli_handlers.jl/server.jl code path exactly (same
# on-disk layout, same insertion/search/persistence semantics) so a dataset touched through
# this API stays fully interoperable with the CLI and HTTP server -- this is a second way to
# reach the same shared engine, not a separate one.
#
# Scope is deliberately bounded to dataset lifecycle + core CRUD + calibrate + one
# representative heavy op (allknn). fft/neardup/hsp/rebuild!/dump_dataset/load_dataset are
# not built here -- a documented follow-up (PLAN.md §8.5), not an oversight.

"""
    EmbeddedHandle

Bundles everything a script needs to keep working against one open dataset: its directory
layout plus the live `Dataset.DatasetManager`/`IndexEngine.SearchEngineWrapper` pair
`cli_handlers.jl`'s `_load_dataset_and_engine` already threads through separately for a
single CLI command -- worth bundling here since an embedded-API caller makes many calls
against the same open dataset instead of running once and exiting.
"""
mutable struct EmbeddedHandle
    workdir::String
    id::String
    dir::String
    ds::Dataset.DatasetManager
    engine::IndexEngine.SearchEngineWrapper
end

_snapshot_path(dir::String, id::String) = joinpath(dir, "$(id).snapshot.jld2")

function _save_snapshot!(handle::EmbeddedHandle)
    engine = handle.engine
    Persistence.save_snapshot(_snapshot_path(handle.dir, handle.id),
        (index=engine.index, text_kind=engine.text_kind, voc=engine.voc, model=engine.model, deleted_ids=engine.deleted_ids))
end

"""
    create_dataset(workdir, id; index_type="searchgraph", distance="L2", schema=MetaSchema()) -> EmbeddedHandle

Creates a brand-new dataset directly on disk at `<workdir>/<id>`, with no HTTP server or CLI
subprocess involved -- the same on-disk layout `similarity-search build` already produces
(`<workdir>/<id>/<id>.snapshot.jld2`), so a dataset created this way can later be inspected/
rebuilt/dumped by the CLI, or reopened by [`open_dataset`](@ref).
"""
function create_dataset(workdir::String, id::String; index_type::String="searchgraph", distance::String="L2", schema::MetaSchema=MetaSchema())
    dir = joinpath(workdir, id)
    mkpath(dir)
    ds = Dataset.open_dataset(dir, id, schema)
    engine = IndexEngine.create_engine(index_type, distance)
    handle = EmbeddedHandle(workdir, id, dir, ds, engine)
    _save_snapshot!(handle)
    return handle
end

"""
    open_dataset(workdir, id; read_only=false) -> EmbeddedHandle

Reopens a dataset previously created by [`create_dataset`](@ref) (or by
`similarity-search build`, since both use the same
`<workdir>/<id>/<id>.snapshot.jld2` layout), restoring its search engine from the JLD2
snapshot. Falls back to a fresh, empty `searchgraph`/`L2` engine if no snapshot exists yet
at this path, mirroring `Server._reload_one_dataset!`'s own "create if nothing persisted
yet" fallback.

Pass `read_only=true` to inspect a dataset a live `similarity-search-serve` process (or
another script) still has open for writing (mirrors `Dataset.open_dataset`'s own
`read_only` kwarg, used the same way by the CLI's `describe` command) -- a plain
(non-`read_only`) open against a directory something else already has open for writing
raises RocksDB's own real lock error, not a friendly one this function invents.
"""
function open_dataset(workdir::String, id::String; read_only::Bool=false)
    dir = joinpath(workdir, id)
    snap_path = _snapshot_path(dir, id)
    engine = if isfile(snap_path)
        snap = Persistence.load_snapshot(snap_path)
        IndexEngine.restore_engine(snap.index, snap.text_kind, snap.voc, snap.model, snap.deleted_ids)
    else
        IndexEngine.create_engine("searchgraph", "L2")
    end
    ds = Dataset.open_dataset(dir, id; read_only)
    return EmbeddedHandle(workdir, id, dir, ds, engine)
end

"""
    close_dataset!(handle::EmbeddedHandle)

Closes the dataset's RocksDB connection. Does not persist anything -- `append_items!`/
`delete_item!`/`calibrate!` already resave the JLD2 snapshot themselves after every
mutation, so there is nothing left to flush here.
"""
function close_dataset!(handle::EmbeddedHandle)
    Dataset.close_dataset(handle.ds)
    return nothing
end

"""
    append_items!(handle::EmbeddedHandle, items) -> Int

Appends a batch of items (each an `AbstractDict` with a `"vector"` key for a dense dataset
or a `"text"` key for a text one, plus optional metadata fields) and resaves the JLD2
snapshot -- mirrors `Server.handle_append`'s exact insertion + persistence sequence, so a
dataset written to via this embedded API stays consistent with what a CLI `describe`/
`rebuild`, or a `similarity-search-serve` process that later opens the same directory,
would see. Returns the number of items actually inserted (an item missing its required key
is skipped, not an error, matching `handle_append`'s behavior).
"""
function append_items!(handle::EmbeddedHandle, items)
    engine = handle.engine
    ds = handle.ds
    is_text = IndexEngine.is_text_index(engine)

    if is_text
        texts = [item["text"] for item in items if haskey(item, "text")]
        IndexEngine.ensure_trained!(engine, texts)
    end

    inserted = 0
    for item in items
        if is_text
            haskey(item, "text") || continue
            IndexEngine.add_item!(engine, item["text"])
        else
            haskey(item, "vector") || continue
            IndexEngine.add_item!(engine, convert(Vector{Float32}, item["vector"]))
        end

        doc_id = length(engine.index)
        raw_dict = merge(Dict{String, Any}(item), Dict{String, Any}(get(item, "meta", Dict())))
        put_metadata!(ds, MetadataRecord(doc_id, ds.schema, raw_dict))
        inserted += 1
    end

    inserted > 0 && _save_snapshot!(handle)
    return inserted
end

function _search_with_filter(engine::IndexEngine.SearchEngineWrapper, ds::Dataset.DatasetManager, query, k::Int, filter_spec::AbstractDict)
    overfetch = max(k * 5, k + 20)
    raw = IndexEngine.search_live(engine, query, overfetch)

    ids = Int32[]
    dists = Float32[]
    for (id, dist) in zip(raw.id, raw.dist)
        record = get_metadata(ds, id)
        record === nothing && continue
        matches_filter(record, filter_spec) || continue
        push!(ids, id)
        push!(dists, dist)
        length(ids) == k && break
    end
    return (id=ids, dist=dists)
end

function _hydrate_results(ds::Dataset.DatasetManager, res_knn)
    results = NamedTuple[]
    for (doc_id, dist) in zip(res_knn.id, res_knn.dist)
        record = get_metadata(ds, doc_id)
        orig_id = record === nothing ? string(doc_id) : something(get_field(record, "id"), string(doc_id))
        push!(results, (id=orig_id, doc_id=doc_id, distance=dist))
    end
    return results
end

"""
    search(handle::EmbeddedHandle, vector; k=10, filter=nothing) -> Vector{<:NamedTuple}

Dense vector search, hydrated with each hit's original id (mirrors `Server.handle_search`
minus the HTTP/telemetry/pagination machinery). `filter`, if given, is a `Dict` of declared
metadata field => value, applied via `Schema.matches_filter` the same way the HTTP API's
`filter` body field is (over-fetches candidates, drops non-matching ones, keeps up to `k`).
Returns a `Vector` of `(id, doc_id, distance)` named tuples, possibly fewer than `k` if
post-filtering (deletions or `filter`) leaves too few candidates.
"""
function search(handle::EmbeddedHandle, vector; k::Int=10, filter=nothing)
    query = convert(Vector{Float32}, vector)
    res_knn = filter === nothing ?
        IndexEngine.search_live(handle.engine, query, k) :
        _search_with_filter(handle.engine, handle.ds, query, k, filter)
    return _hydrate_results(handle.ds, res_knn)
end

"""
    ftsearch(handle::EmbeddedHandle, text; k=10) -> Vector{<:NamedTuple}

Text search against a bm25/weighted-inverted-file dataset (mirrors `Server.handle_ftsearch`).
"""
function ftsearch(handle::EmbeddedHandle, text::AbstractString; k::Int=10)
    res_knn = IndexEngine.search_live(handle.engine, text, k)
    return _hydrate_results(handle.ds, res_knn)
end

"""
    delete_item!(handle::EmbeddedHandle, doc_id::Integer)

Soft-deletes `doc_id` (future searches exclude it, the underlying index is untouched) and
resaves the JLD2 snapshot -- mirrors `Server.handle_delete_item`.
"""
function delete_item!(handle::EmbeddedHandle, doc_id::Integer)
    IndexEngine.mark_deleted!(handle.engine, Int(doc_id))
    _save_snapshot!(handle)
    return nothing
end

"""
    fetch_items(handle::EmbeddedHandle, ids) -> Vector{Dict{String,Any}}

Batch metadata retrieval by id -- each element of `ids` may be the internal integer
`doc_id` or the caller-supplied original `id` (resolved via `Dataset.find_by_original_id`).
Mirrors `Server.handle_fetch`; an id that resolves to nothing is silently skipped.
"""
function fetch_items(handle::EmbeddedHandle, ids)
    ds = handle.ds
    results = Dict{String, Any}[]
    for raw_id in ids
        record = nothing
        maybe_int = tryparse(Int, string(raw_id))
        maybe_int !== nothing && (record = get_metadata(ds, maybe_int))
        record === nothing && (record = find_by_original_id(ds, string(raw_id)))
        record === nothing && continue

        extra = isempty(record.extra) ? Dict{String, Any}() : JSON.parse(String(copy(record.extra)))
        push!(results, merge(Dict{String, Any}("doc_id" => record.doc_id), record.declared_fields, extra))
    end
    return results
end

"""
    exists(handle::EmbeddedHandle, ids) -> Vector{<:NamedTuple}

For each id in `ids` (internal `doc_id` or original id), reports whether a record exists
and, if so, whether it's been soft-deleted. Mirrors `Server.handle_exists`.
"""
function exists(handle::EmbeddedHandle, ids)
    ds = handle.ds
    engine = handle.engine
    results = NamedTuple[]
    for raw_id in ids
        id_str = string(raw_id)
        record = nothing
        maybe_int = tryparse(Int, id_str)
        maybe_int !== nothing && (record = get_metadata(ds, maybe_int))
        record === nothing && (record = find_by_original_id(ds, id_str))
        found = record !== nothing
        deleted = found && (record.doc_id in engine.deleted_ids)
        push!(results, (id=id_str, exists=found, deleted=deleted))
    end
    return results
end

"""
    calibrate!(handle::EmbeddedHandle; minrecall=0.9, numqueries=64, ksearch=10, queries=nothing) -> BeamSearch

Runs `IndexEngine.calibrate!`'s real hyperparameter sweep and resaves the JLD2 snapshot so
the calibrated `BeamSearch` (stored inside `engine.index.algo[]`, itself part of the saved
`index`) survives a later `close_dataset!` + `open_dataset` round-trip -- there is no
`descriptor.json` in this embedded API for `Server.handle_calibrate`'s separate persistence
path to write to, so the JLD2 snapshot is this API's sole persistence mechanism.
"""
function calibrate!(handle::EmbeddedHandle; minrecall::Real=0.9, numqueries::Int=64, ksearch::Int=10, queries=nothing)
    bs = IndexEngine.calibrate!(handle.engine; minrecall, numqueries, ksearch, queries)
    _save_snapshot!(handle)
    return bs
end

"""
    allknn(handle::EmbeddedHandle; k=10) -> Vector{<:NamedTuple}

Runs `SimilaritySearch.allknn` synchronously against the dataset's dense index -- no
Job/spool machinery at all, unlike the HTTP API's `POST /api/v1/jobs/allknn` (PLAN.md
§8.5's "runs in-process" framing for the embedded surface). Errors if the dataset is a text
index or empty (mirrors `cli_handlers.jl`'s `_require_dense`). Returns one
`(id, neighbors, dists)` named tuple per item, `id` being the item's 1-based position in
insertion order (matching `execute_allknn`'s own output convention), not its `doc_id`.
"""
function allknn(handle::EmbeddedHandle; k::Int=10)
    engine = handle.engine
    IndexEngine.is_text_index(engine) && error("allknn requires a dense (vector) index, but this dataset is a text index")
    (engine.index === nothing || length(engine.index) == 0) && error("allknn requires a non-empty dense index")

    ids, dists = SimilaritySearch.allknn(engine.index, engine.ctx, k)
    n = size(ids, 2)
    results = NamedTuple[]
    for i in 1:n
        neighbor_ids = Int[]
        neighbor_dists = Float64[]
        for j in 1:k
            ids[j, i] == 0 && break
            push!(neighbor_ids, Int(ids[j, i]))
            push!(neighbor_dists, Float64(dists[j, i]))
        end
        push!(results, (id=i, neighbors=neighbor_ids, dists=neighbor_dists))
    end
    return results
end
