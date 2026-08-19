# Friendly, function-per-operation embedded API (PLAN.md §8.5, chunk 22): a script running
# on the same machine against the same --workdir a similarity-search/similarity-search-serve
# process already uses, with no HTTP, no CLI subprocess, no running server required. Every
# function here mirrors an existing cli_handlers.jl/server.jl code path exactly (same
# on-disk layout, same insertion/search/persistence semantics) so a project touched through
# this API stays fully interoperable with the CLI and HTTP server -- this is a second way to
# reach the same shared engine, not a separate one.
#
# Scope is deliberately bounded to project lifecycle + core CRUD + calibrate + one
# representative heavy op (allknn). fft/neardup/hsp/rebuild!/dump_dataset/load_dataset are
# not built here -- a documented follow-up (PLAN.md §8.5), not an oversight.

"""
    EmbeddedEngine

Bundles everything a script needs to keep working against one open project: its directory
layout plus the live `Project.ProjectManager`/`IndexEngine.AbstractSearchEngine` pair
`cli_handlers.jl`'s `_load_dataset_and_engine` already threads through separately for a
single CLI command, plus the `Persistence.EngineStore` (a RocksDB column family shared
with `project.db`, see `Persistence.open_engine_store`) that engine mutations persist
into field-by-field, and `pending_flush` -- a flag a `GenericEngine`'s
`IndexEngine.CallbackLog` callback sets (see [`create_project`](@ref)) that this module
checks and clears from *outside* the engine's own insertion call, once it's known safe to
do so (see [`_maybe_flush_index!`](@ref); unused for `SearchGraphEngine`/`BM25Engine`/
`InvertedFileEngine`, which each persist their index incrementally by themselves -- see
[`_searchgraph_on_change`](@ref)/[`_invertedfile_on_change`](@ref)) -- worth bundling here
since an embedded-API caller makes many calls against the same open project instead of
running once and exiting.
"""
mutable struct EmbeddedEngine
    workdir::String
    dataset::String
    dir::String
    project::Project.ProjectManager
    engine::IndexEngine.AbstractSearchEngine
    store::Persistence.EngineStore
    pending_flush::Base.RefValue{Bool}
    schema_version::Int
    dense_vectors::Base.RefValue{Union{Nothing,MMapMatrixDatabase}}
end

"""
    _maybe_flush_index!(handle::EmbeddedEngine)

Persists `handle.engine.index` if (and only if) `handle.pending_flush[]` is set, then
clears the flag. Callers must only call this from a point where the just-finished mutation
is fully done -- in particular, *not* from inside an `IndexEngine.CallbackLog` callback
itself. Right after an `IndexEngine.add_item!`/`index!` call returns to this
module is always safe. A no-op for `SearchGraphEngine`/`BM25Engine`/`InvertedFileEngine`:
each has its own dedicated incremental `on_change` (see [`_searchgraph_on_change`](@ref)/
[`_invertedfile_on_change`](@ref)) that never touches `pending_flush`; only `GenericEngine`
(`ExhaustiveSearch`/`ParallelExhaustiveSearch`) still uses this whole-index path.
"""
function _maybe_flush_index!(handle::EmbeddedEngine)
    if handle.pending_flush[]
        handle.pending_flush[] = false
        Persistence.save_field!(handle.store, :index, handle.engine.index)
    end
end

"""
    _searchgraph_on_change(store::Persistence.EngineStore, adj_store::Persistence.AdjacencyStore) -> Function

The `on_change` callback for a `SearchGraphEngine`: fires from *inside*
`IndexEngine.index!(engine::IndexEngine.SearchGraphEngine)` (never from `add_item!`/
`append_items!` anymore -- see that function's docstring for why staging and indexing are
now two separate steps), once per `sp:ep` range `SimilaritySearch.index!`'s own internal
batching processes. Saves each object `i` in that range's own direct-links-only neighbor
list under its own key in `adj_store` (`Persistence.save_neighbors!`,
`IndexEngine.direct_neighbors`) -- adjacency lives in its own RocksDB column family, keyed
by object id (see `Persistence.AdjacencyStore`) -- never the whole (growing) graph as one
value -- and persists `ep` as the new `:graph_len` engine field (`Persistence`'s small
per-field store), so a restart mid-catch-up resumes exactly where this range left off
rather than redoing (or worse, silently forgetting) it.

Raw vectors are *not* handled here -- unlike before, they're persisted directly by
[`append_items!`](@ref) at stage time (see its own docstring), since they're durable and
meaningful the moment they're staged, well before any graph-linking work happens; only
adjacency and `:graph_len` genuinely need to wait for `SimilaritySearch.index!` itself to
compute them, which is exactly what makes it safe to do this synchronously, right inside
the callback (unlike the whole-graph save every other engine kind uses, see
[`_maybe_flush_index!`](@ref)): this is exactly the direct-links-only slice
`IndexEngine.CallbackLog` hands over at that point, and reconstruction
(`IndexEngine.build_searchgraph`, used by [`open_project`](@ref)) reconnects every reverse
link itself, once, up to the restored `:graph_len`, after replaying every staged vector
and looking up every graph-indexed object's saved adjacency.
"""
function _searchgraph_on_change(store::Persistence.EngineStore, adj_store::Persistence.AdjacencyStore)
    return (index, sp, ep) -> begin
        for i in sp:ep
            Persistence.save_neighbors!(adj_store, i, IndexEngine.direct_neighbors(index, i))
        end
        Persistence.save_field!(store, :graph_len, ep)
    end
end

"""
    _invertedfile_on_change(obj_store::Persistence.InvertedFileObjectStore) -> Function

The `on_change` callback for a `BM25Engine`/`InvertedFileEngine`: on every
`push_item!`/`append_items!` report for range `sp:ep`, saves *only* that range's raw
indexed objects (bags-of-words or `SparseVector`s -- see `IndexEngine.invertedfile_objects`)
as a new, never-again-rewritten block in `obj_store` (`Persistence.append_objects!`) --
never the whole (growing) index as one value. Safe to do synchronously, right inside the
callback (unlike the whole-index save `GenericEngine` uses, see
[`_maybe_flush_index!`](@ref)): an inverted file's `LOG` only fires after a call's
mutation is fully done, so there's no partial-state hazard here the way there is for a
`SearchGraph` (see `IndexEngine.CallbackLog`'s docstring). Reconstruction
(`IndexEngine.build_bm25invertedfile`/`build_invertedfile`, used by [`open_project`](@ref))
rebuilds the whole index by replaying every saved object back through the library's own
insertion -- see `Persistence.InvertedFileObjectStore`'s docstring for the scaling
trade-off that implies (fine up to a few million documents; not a design for
billion-document corpora needing disk-backed posting lists).
"""
_invertedfile_on_change(obj_store::Persistence.InvertedFileObjectStore) =
    (index, sp, ep) -> Persistence.append_objects!(obj_store, sp, IndexEngine.invertedfile_objects(index, sp, ep))

"""
    create_project(workdir, dataset; index_type=SearchGraph, distance=nothing, minrecall=0.9, schema_version=1) -> EmbeddedEngine

Creates a brand-new project directly on disk at `<workdir>/<dataset>`, with no HTTP server or
CLI subprocess involved -- the same on-disk layout `similarity-search build` already produces,
so a project created this way can later be inspected/rebuilt/dumped by the CLI, or reopened by
[`open_project`](@ref). `distance`, left at its default `nothing`, defers to whichever default
`IndexEngine.create_engine` picks for `index_type` (e.g. `SqL2()` for a `SearchGraph`,
`NormCosine()` for an `InvertedFile`) instead of this function imposing one blanket default
across every index kind. `minrecall` is the target recall a `SearchGraphEngine` autotunes
`BeamSearch` toward as it grows (see `IndexEngine.SearchGraphEngine`); it must be given here,
at creation, since it's carried on the engine and restored verbatim by [`open_project`](@ref)
rather than re-derived later -- `calibrate!` remains available afterwards as a separate,
explicit re-optimization pass. Ignored for index types with no `BeamSearch` to autotune.
`schema_version` is stamped onto every `Schema.MetadataRecord` [`append_items!`](@ref)
writes for this project's lifetime (not persisted/restored itself -- like `minrecall`
before it was carried on the engine, a caller reopening this same project later must pass
the same `schema_version` again to keep tagging new records consistently with old ones);
it's never validated against `meta`'s actual shape, purely a version tag for the caller's
own interpretation of it.

The engine's index itself is persisted incrementally as it grows, not rewritten wholesale
on every `append_items!` call: for a `SearchGraph`, every insertion report saves just its
own direct-links-only block (see [`_searchgraph_on_change`](@ref)); for every other index
kind, a full save happens right after the `add_item!` call that triggered it returns (see
[`_maybe_flush_index!`](@ref)). [`close_project!`](@ref) forces one final flush of the
latter so nothing recent is lost -- the former never has anything left to flush, since
each block is already durable the instant it's saved.
"""
function create_project(workdir::String, dataset::String; index_type::Type=SearchGraph, distance=nothing, minrecall::Union{Nothing,Real}=0.9, schema_version::Int=1)
    dir = joinpath(workdir, dataset)
    mkpath(dir)
    project = Project.open_project(dir, dataset; extra_cf_names=[Persistence.ENGINE_CF, Persistence.ADJACENCY_CF, Persistence.INVFILE_DB_CF])
    store = Persistence.open_engine_store(project.db)
    pending_flush = Ref(false)
    dense_vectors = Ref{Union{Nothing,MMapMatrixDatabase}}(nothing)
    on_change = if index_type === SearchGraph
        _searchgraph_on_change(store, Persistence.open_adjacency_store(project.db))
    elseif index_type === BM25InvertedFile || index_type === InvertedFile
        _invertedfile_on_change(Persistence.open_invertedfile_object_store(project.db))
    else
        (_, __, ___) -> (pending_flush[] = true)
    end
    engine = distance === nothing ?
        IndexEngine.create_engine(index_type; minrecall, on_change) :
        IndexEngine.create_engine(index_type; distance, minrecall, on_change)
    Persistence.save_fields!(store, IndexEngine.snapshot_state(engine))
    return EmbeddedEngine(workdir, dataset, dir, project, engine, store, pending_flush, schema_version, dense_vectors)
end

"""
    open_project(workdir, dataset; read_only=false) -> EmbeddedEngine

Reopens a project previously created by [`create_project`](@ref), restoring its search
engine -- including the `minrecall` target and any calibrated `opt_beamsearch` it was
created/calibrated with -- field-by-field from its `Persistence.EngineStore` (a
`SearchGraphEngine`'s index specifically via `IndexEngine.build_searchgraph` replaying its
saved insertion blocks, see [`_searchgraph_on_change`](@ref)). Falls back to a fresh,
empty `SearchGraph`/`SqL2`/`minrecall=0.9` engine if this project has never been saved
yet, mirroring `Server._reload_one_dataset!`'s own "create if nothing persisted yet"
fallback.

Pass `read_only=true` to inspect a project a live `similarity-search-serve` process (or
another script) still has open for writing (mirrors `Project.open_project`'s own
`read_only` kwarg, used the same way by the CLI's `describe` command) -- a plain
(non-`read_only`) open against a directory something else already has open for writing
raises RocksDB's own real lock error, not a friendly one this function invents.
"""
function open_project(workdir::String, dataset::String; read_only::Bool=false, schema_version::Int=1)
    dir = joinpath(workdir, dataset)
    project = Project.open_project(dir, dataset; read_only, extra_cf_names=[Persistence.ENGINE_CF, Persistence.ADJACENCY_CF, Persistence.INVFILE_DB_CF])
    store = Persistence.open_engine_store(project.db)
    pending_flush = Ref(false)
    dense_vectors = Ref{Union{Nothing,MMapMatrixDatabase}}(nothing)

    kind = Persistence.load_field(store, :kind)
    engine = if kind === nothing
        on_change = _searchgraph_on_change(store, Persistence.open_adjacency_store(project.db))
        engine = IndexEngine.create_engine(SearchGraph; on_change)
        Persistence.save_fields!(store, IndexEngine.snapshot_state(engine))
        engine
    elseif kind === IndexEngine.SearchGraphEngine
        dense_vectors[] = Persistence.open_dense_vectors(dir; read_only)
        adj_store = Persistence.open_adjacency_store(project.db)
        state = (
            kind=kind,
            distance=Persistence.load_field(store, :distance),
            vector_blocks=Persistence.load_dense_vector_blocks(dense_vectors[]),
            load_neighbors=(i -> Persistence.load_neighbors(adj_store, i)),
            graph_len=Persistence.load_field(store, :graph_len, 0),
            minrecall=Persistence.load_field(store, :minrecall),
            opt_beamsearch=Persistence.load_field(store, :opt_beamsearch),
            deleted_ids=Persistence.load_field(store, :deleted_ids),
        )
        IndexEngine.restore_engine(state; on_change=_searchgraph_on_change(store, adj_store))
    elseif kind === IndexEngine.BM25Engine
        obj_store = Persistence.open_invertedfile_object_store(project.db)
        state = (
            kind=kind,
            voc=Persistence.load_field(store, :voc),
            object_blocks=Persistence.load_object_blocks(obj_store),
            deleted_ids=Persistence.load_field(store, :deleted_ids),
        )
        IndexEngine.restore_engine(state; on_change=_invertedfile_on_change(obj_store))
    elseif kind === IndexEngine.InvertedFileEngine
        obj_store = Persistence.open_invertedfile_object_store(project.db)
        state = (
            kind=kind,
            voc=Persistence.load_field(store, :voc),
            model=Persistence.load_field(store, :model),
            distance=Persistence.load_field(store, :distance),
            object_blocks=Persistence.load_object_blocks(obj_store),
            deleted_ids=Persistence.load_field(store, :deleted_ids),
        )
        IndexEngine.restore_engine(state; on_change=_invertedfile_on_change(obj_store))
    else
        on_change = (_, __, ___) -> (pending_flush[] = true)
        common = (index=Persistence.load_field(store, :index), deleted_ids=Persistence.load_field(store, :deleted_ids))
        extra = NamedTuple{IndexEngine.extra_state_fields(kind)}(map(f -> Persistence.load_field(store, f), IndexEngine.extra_state_fields(kind)))
        state = (kind=kind, common..., extra...)
        IndexEngine.restore_engine(state; on_change)
    end
    return EmbeddedEngine(workdir, dataset, dir, project, engine, store, pending_flush, schema_version, dense_vectors)
end

"""
    close_project!(handle::EmbeddedEngine)

Flushes the engine's index one final time and closes the project's RocksDB connection
(and, for a `SearchGraphEngine` that ever had a vector appended, its `MMapMatrixDatabase`
file too -- see [`_searchgraph_on_change`](@ref)). For `SearchGraphEngine`/`BM25Engine`/
`InvertedFileEngine` there is nothing left to flush -- every insertion block was already
saved the instant it was reported -- so this only does the final `:index` save for
`GenericEngine`. Every other field (`deleted_ids`, `opt_beamsearch`, ...) is, for every
kind, already persisted immediately by whichever call changed it (`delete_item!`,
`calibrate!`, ...).
"""
function close_project!(handle::EmbeddedEngine)
    handle.pending_flush[] = false
    if !(handle.engine isa Union{IndexEngine.SearchGraphEngine, IndexEngine.BM25Engine, IndexEngine.InvertedFileEngine})
        Persistence.save_field!(handle.store, :index, handle.engine.index)
    end
    handle.dense_vectors[] === nothing || close(handle.dense_vectors[])
    Project.close_project(handle.project)
    return nothing
end

"""
    index!(handle::EmbeddedEngine, corpus)

Explicitly trains a text project's `Vocabulary` (and builds its real
`BM25InvertedFile`/`InvertedFile`) from `corpus` -- extends `SimilaritySearch.index!`, the
same generic function [`fft`](@ref)/[`closestpairs`](@ref)'s underlying calls rely on for
a dense project, given its own meaning here: "make this text project ready to accept
items." Must be called exactly once, before any [`append_items!`](@ref), for a
`BM25InvertedFile`/`InvertedFile` project -- errors for a dense `SearchGraph` project
(which has no training step at all), for an already-trained engine, or for an empty
`corpus`. See `IndexEngine.index!`'s own docstring for why `corpus` doesn't have to be the
exact batch you go on to insert afterwards.

Persists `voc`/`model`/`distance` together immediately once training succeeds (a one-time,
must-not-lose transition -- `index` itself has nothing to save yet at that point, since
training only constructs an empty index; its first indexed object comes from a later
[`append_items!`](@ref) call instead, already covered by the engine's own incremental
`on_change`).
"""
function index!(handle::EmbeddedEngine, corpus)
    engine = handle.engine
    IndexEngine.is_text_index(engine) || error("index!(handle, corpus) only applies to a text (BM25InvertedFile/InvertedFile) project -- see index!(handle) for a dense SearchGraph project")
    IndexEngine.index!(engine, corpus)
    # `snapshot_state` already knows exactly which fields this concrete engine kind has
    # (BM25Engine has no `distance`, InvertedFileEngine does, etc.) -- reuse it rather
    # than re-deriving the field set by hand here.
    Persistence.save_fields!(handle.store, IndexEngine.snapshot_state(engine))
    return handle
end

"""
    index!(handle::EmbeddedEngine)

For a dense (`SearchGraph`) project: builds graph connections for every vector staged via
[`append_items!`](@ref) since the last call (or since creation) -- see
`IndexEngine.index!(engine::IndexEngine.SearchGraphEngine)`'s own docstring for why this is
idempotent and safe to call repeatedly (it only ever processes the backlog). Nothing
becomes visible to [`search`](@ref)/[`allknn`](@ref)/[`fft`](@ref)/[`closestpairs`](@ref)/
[`bichromatic_kclosestpairs`](@ref) until this runs at least once -- staging
(`append_items!`) alone is not enough.

Errors for a text project (train it via [`index!`](@ref index!(::EmbeddedEngine, ::Any))
instead) or for any other index kind, which has no staged-vs-indexed split to catch up
(`GenericEngine`/`BM25Engine`/`InvertedFileEngine` already index synchronously on every
`add_item!`, so there's never a backlog for this to process).
"""
function index!(handle::EmbeddedEngine)
    engine = handle.engine
    engine isa IndexEngine.SearchGraphEngine || error("index!(handle) with no corpus only applies to a dense SearchGraph project -- see index!(handle, corpus) for a text project")
    IndexEngine.index!(engine)
    return handle
end

"""
    append_items!(handle::EmbeddedEngine, items) -> Int

Appends a batch of items (each an `AbstractDict` with a `"vector"` key for a dense project
or a `"text"` key for a text one, plus optional metadata fields) -- mirrors
`Server.handle_append`'s exact insertion sequence, so a project written to via this
embedded API stays consistent with what a CLI `describe`/`rebuild`, or a
`similarity-search-serve` process that later opens the same directory, would see. Returns
the number of items actually inserted (an item missing its required key is skipped, not
an error, matching `handle_append`'s behavior).

For a text (`BM25InvertedFile`/`InvertedFile`) project, [`index!`](@ref index!(::EmbeddedEngine, ::Any))
must already have been called on `handle` -- there is no implicit auto-training on the
first batch here anymore; an untrained text engine errors on the very first item instead
of silently dropping every item forever (see `IndexEngine.add_item!`'s docstring).

For a dense (`SearchGraph`) project, this only *stages* raw vectors -- durably (see
"Performance" below) but without any graph-linking work -- so freshly appended items are
*not* yet visible to [`search`](@ref)/[`allknn`](@ref)/etc. until an explicit
[`index!`](@ref index!(::EmbeddedEngine)) call catches up the backlog. Every other engine
kind (`GenericEngine`/`BM25Engine`/`InvertedFileEngine`) still indexes synchronously, on
every `add_item!`, exactly as before -- only `SearchGraph` has this stage-then-index
split, since it's the only one `SimilaritySearch.jl` itself supports it for (see
`IndexEngine.index!(engine::IndexEngine.SearchGraphEngine)`'s docstring).

Persistence: for a dense project, every batch's raw vectors are appended directly to the
project's `MMapMatrixDatabase` (see [`index!`](@ref index!(::EmbeddedEngine))'s docstring
and the "Performance" note below) right here, at stage time -- not via `on_change`, unlike
adjacency, since a vector is already durable-worthy the moment it's staged, long before any
graph-linking happens. Every other engine kind reports its own insertions through
`IndexEngine.CallbackLog`: `BM25Engine`/`InvertedFileEngine` persist incrementally,
synchronously, inside that callback (see [`_invertedfile_on_change`](@ref));
`GenericEngine` instead flags a pending whole-index save that happens right after each
`add_item!` call returns (see [`_maybe_flush_index!`](@ref)).

# Performance: batch size matters a lot for a `SearchGraph` (dense) project

Every call here does *one* `SimilaritySearch.append_items!` call into the project's
`MMapMatrixDatabase` for its whole batch of staged vectors -- and that type flushes and
`fsync`s once per `append_items!` call (never per individual vector), so the number of
calls to *this* function a caller makes (not the total vector count) is what drives
wall-clock time. Confirmed empirically, 200,000 128-dim vectors written as one project's
dense store: one 200,000-item call (1 fsync) took ~0.10s; the same total split into calls
of 1,000 items (200 fsyncs) took ~0.14s; split into calls of 100 items (2,000 fsyncs) took
~0.31s; split into calls of 10 items (20,000 fsyncs) took ~2.1s. Calling this function once
per single item (the worst case, one fsync per item) is dramatically worse still --
measured independently at roughly 1000x slower than an all-at-once call for the same total
count. Prefer fewer, larger calls to this function over many small ones whenever a caller
controls the batching.
"""
function append_items!(handle::EmbeddedEngine, items)
    engine = handle.engine
    project = handle.project
    is_text = IndexEngine.is_text_index(engine)
    is_dense_graph = engine isa IndexEngine.SearchGraphEngine
    staged_vectors = Vector{Float32}[]

    inserted = 0
    for item in items
        if is_text
            haskey(item, "text") || continue
            IndexEngine.add_item!(engine, item["text"])
        else
            haskey(item, "vector") || continue
            v = convert(Vector{Float32}, item["vector"])
            IndexEngine.add_item!(engine, v)
            is_dense_graph && push!(staged_vectors, v)
        end
        _maybe_flush_index!(handle)

        _id = Int32(_current_size(engine))
        record, meta = Schema.split_item(_id, handle.schema_version, Dict{String,Any}(item))
        put_metadata!(project, record, meta)
        inserted += 1
    end

    if is_dense_graph && !isempty(staged_vectors)
        if handle.dense_vectors[] === nothing
            handle.dense_vectors[] = MMapMatrixDatabase(Persistence.dense_vectors_path(handle.dir), length(staged_vectors[1]), Float32)
        end
        SimilaritySearch.append_items!(handle.dense_vectors[], staged_vectors)
    end

    return inserted
end

"""
    _current_size(engine::IndexEngine.AbstractSearchEngine) -> Int

The count that determines the next item's `_id`: for a `SearchGraphEngine`, the number of
*staged* vectors (`length(database(engine.index))`, i.e. `engine.index.db`'s own count) --
since [`append_items!`](@ref) only stages, `length(engine.index)` itself (the
graph-indexed count) would lag behind and hand out the wrong, already-taken `_id`s. Every
other engine kind still indexes synchronously on `add_item!`, so `length(engine.index)`
already reflects the item just added, same as before.
"""
_current_size(engine::IndexEngine.SearchGraphEngine) = length(SimilaritySearch.database(engine.index))
_current_size(engine::IndexEngine.AbstractSearchEngine) = length(engine.index)

function _search_with_filter(engine::IndexEngine.AbstractSearchEngine, project::Project.ProjectManager, query, k::Int, predicate; minrecall=nothing)
    overfetch = max(k * 5, k + 20)
    raw = IndexEngine.search_live(engine, query, overfetch; minrecall)

    ids = Int32[]
    dists = Float32[]
    for (_id, dist, deleted) in zip(raw.id, raw.dist, raw.deleted)
        # A soft-deleted id has no metadata a caller is allowed to see, so it can never
        # satisfy a metadata predicate -- skip it without even attempting the lookup.
        deleted && continue
        record = get_metadata(project, _id)
        record === nothing && continue
        meta = get_meta(project, _id)
        predicate(record, meta) || continue
        push!(ids, _id)
        push!(dists, dist)
        length(ids) == k && break
    end
    return (id=ids, dist=dists, deleted=falses(length(ids)))
end

function _hydrate_results(project::Project.ProjectManager, res_knn)
    results = NamedTuple[]
    for (_id, dist, deleted) in zip(res_knn.id, res_knn.dist, res_knn.deleted)
        if deleted
            # Metadata for a soft-deleted _id is not accessible through search --
            # report the deletion marker instead of the (hidden) original doc_id.
            push!(results, (id=string(_id), doc_id=_id, distance=dist, deleted=true))
            continue
        end
        record = get_metadata(project, _id)
        orig_id = record === nothing ? string(_id) : something(record.doc_id, string(_id))
        push!(results, (id=orig_id, doc_id=_id, distance=dist, deleted=false))
    end
    return results
end

"""
    search(handle::EmbeddedEngine, vector; k=10, filter=nothing) -> Vector{<:NamedTuple}

Dense vector search, hydrated with each hit's original id (mirrors `Server.handle_search`
minus the HTTP/telemetry/pagination machinery). `filter`, if given, is a
`(record::Schema.MetadataRecord, meta) -> Bool` predicate function called on each
overfetched candidate's record and (lazily decoded, see `Schema.decode_meta`) `meta`
(over-fetches candidates, drops the ones the predicate rejects, keeps up to `k`) -- e.g.
`(record, meta) -> meta !== nothing && meta.year >= 2020`. A caller here already has real
Julia functions to work with, not a JSON wire format to encode a filter into, so `filter`
is just that function, not a name-keyed spec for some interpreter to replay. `meta` is
fetched (one extra `Project.get_meta` read per overfetched candidate) even if `record`
alone would've been enough for a given predicate -- the common case (filtering on a
`meta` field) needs it, and there's no way to know a predicate won't from here.

Without `filter`, this is a plain top-`k` search: soft-deleted candidates are not hidden
or backfilled -- they're returned with `deleted=true` and no hydrated metadata (see
[`IndexEngine.search_live`](@ref)), so `id` falls back to `string(doc_id)` for them.
Getting `k` *live* results back is a paging concern for a layer above this one (e.g. a
server walking successive windows via a cursor), not something this function does itself.

`minrecall`, for a `SearchGraphEngine`, searches at (approximately) that target recall
using a calibrated `BeamSearch` from `engine.opt_beamsearch` instead of its current
default (see [`IndexEngine.search_live`](@ref)) -- if that table is still empty, this
triggers a one-off `calibrate!` over `IndexEngine.DEFAULT_MINRECALL_LEVELS` and persists
the resulting `opt_beamsearch` so that calibration isn't silently repeated on every future
search. Ignored for any other engine kind.

Returns a `Vector` of `(id, doc_id, distance, deleted)` named tuples.
"""
function search(handle::EmbeddedEngine, vector; k::Int=10, filter=nothing, minrecall=nothing)
    query = convert(Vector{Float32}, vector)
    needs_save = minrecall !== nothing && handle.engine isa IndexEngine.SearchGraphEngine && isempty(handle.engine.opt_beamsearch)
    res_knn = filter === nothing ?
        IndexEngine.search_live(handle.engine, query, k; minrecall) :
        _search_with_filter(handle.engine, handle.project, query, k, filter; minrecall)
    needs_save && Persistence.save_field!(handle.store, :opt_beamsearch, handle.engine.opt_beamsearch)
    return _hydrate_results(handle.project, res_knn)
end

"""
    ftsearch(handle::EmbeddedEngine, text; k=10) -> Vector{<:NamedTuple}

Text search against a bm25/weighted-inverted-file project (mirrors `Server.handle_ftsearch`).
Same soft-delete marker behavior as [`search`](@ref); `minrecall` is accepted only for
parity with `search` and is a no-op here since no text engine has a `BeamSearch` to
calibrate.
"""
function ftsearch(handle::EmbeddedEngine, text::AbstractString; k::Int=10, minrecall=nothing)
    res_knn = IndexEngine.search_live(handle.engine, text, k; minrecall)
    return _hydrate_results(handle.project, res_knn)
end

"""
    delete_item!(handle::EmbeddedEngine, _id::Integer)

Soft-deletes `_id` (future searches exclude it, the underlying index is untouched) and
immediately persists just the `deleted_ids` field -- mirrors `Server.handle_delete_item`.
"""
function delete_item!(handle::EmbeddedEngine, _id::Integer)
    IndexEngine.mark_deleted!(handle.engine, _id)
    Persistence.save_field!(handle.store, :deleted_ids, handle.engine.deleted_ids)
    return nothing
end

"""
    fetch_items(handle::EmbeddedEngine, ids) -> Vector{Dict{String,Any}}

Batch metadata retrieval by id -- each element of `ids` may be the internal `_id` or the
caller-supplied `doc_id` (resolved via `Project.find_by_doc_id`). Mirrors
`Server.handle_fetch`; an id that resolves to nothing is silently skipped. Each result
merges the record's own fields (`_id`, `doc_id`, `schema_version`, `keywords`, `ref`)
with its `meta` (fully decoded, see `Schema.decode_meta`'s `lazy=false` mode, since the
caller gets a plain merged `Dict` back here, not something meant for further JSON3 use).
"""
function fetch_items(handle::EmbeddedEngine, ids)
    project = handle.project
    results = Dict{String, Any}[]
    for raw_id in ids
        record = nothing
        maybe_int = tryparse(Int, string(raw_id))
        maybe_int !== nothing && (record = get_metadata(project, maybe_int))
        record === nothing && (record = find_by_doc_id(project, string(raw_id)))
        record === nothing && continue

        meta = something(get_meta(project, record._id; lazy=false), Dict{String,Any}())
        fields = Dict{String, Any}(
            "_id" => record._id, "doc_id" => record.doc_id,
            "schema_version" => record.schema_version,
            "keywords" => record.keywords, "ref" => record.refs,
        )
        push!(results, merge(fields, meta))
    end
    return results
end

"""
    exists(handle::EmbeddedEngine, ids) -> Vector{<:NamedTuple}

For each id in `ids` (internal `_id` or caller-supplied `doc_id`), reports whether a
record exists and, if so, whether it's been soft-deleted. Mirrors `Server.handle_exists`.
"""
function exists(handle::EmbeddedEngine, ids)
    project = handle.project
    engine = handle.engine
    results = NamedTuple[]
    for raw_id in ids
        id_str = string(raw_id)
        record = nothing
        maybe_int = tryparse(Int, id_str)
        maybe_int !== nothing && (record = get_metadata(project, maybe_int))
        record === nothing && (record = find_by_doc_id(project, id_str))
        found = record !== nothing
        deleted = found && (record._id in engine.deleted_ids)
        push!(results, (id=id_str, exists=found, deleted=deleted))
    end
    return results
end

"""
    calibrate!(handle::EmbeddedEngine; levels=IndexEngine.DEFAULT_MINRECALL_LEVELS, numqueries=64, ksearch=10, queries=nothing) -> IndexEngine.OptBeamSearch

Runs `IndexEngine.calibrate!`'s real hyperparameter sweep for each recall level in
`levels` and immediately persists just the resulting `opt_beamsearch` table (consulted by
`search`'s own `minrecall` keyword) so it survives a later `close_project!` +
`open_project` round-trip -- there is no `descriptor.json` in this embedded API for
`Server.handle_calibrate`'s separate persistence path to write to, so this `EngineStore`
field is this API's sole persistence mechanism for it.
"""
function calibrate!(handle::EmbeddedEngine; levels=IndexEngine.DEFAULT_MINRECALL_LEVELS, numqueries::Int=64, ksearch::Int=10, queries=nothing)
    opt_bs = IndexEngine.calibrate!(handle.engine; levels, numqueries, ksearch, queries)
    Persistence.save_field!(handle.store, :opt_beamsearch, opt_bs)
    return opt_bs
end

"""
    allknn(handle::EmbeddedEngine; k=10) -> Vector{<:NamedTuple}

Runs `SimilaritySearch.allknn` synchronously against the project's dense index -- no
Job/spool machinery at all, unlike the HTTP API's `POST /api/v1/jobs/allknn` (PLAN.md
§8.5's "runs in-process" framing for the embedded surface). Errors if the project is a text
index or empty (mirrors `cli_handlers.jl`'s `_require_dense`). Returns one
`(id, neighbors, dists)` named tuple per item, `id` being the item's 1-based position in
insertion order (matching `execute_allknn`'s own output convention), not its `doc_id`.
"""
function allknn(handle::EmbeddedEngine; k::Int=10)
    engine = handle.engine
    IndexEngine.is_text_index(engine) && error("allknn requires a dense (vector) index, but this project is a text index")
    length(engine.index) == 0 && error("allknn requires a non-empty dense index")

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

"""
    fft(handle::EmbeddedEngine, k::Integer; start::Int=0, verbose::Bool=false) -> NamedTuple

Runs `SimilaritySearch.fft`'s Farthest-First-Traversal synchronously against the
project's dense index -- picks `k` well-separated items (e.g. as diverse candidate
cluster centers), the same "runs in-process, no job queue" framing as [`allknn`](@ref).
Errors if the project is a text index or empty (same guard as `allknn`). Unlike
`allknn`/`closestpairs`/`bichromatic_kclosestpairs`, this is a brute-force sweep over the
raw vectors (`SimilaritySearch.distance`/`database` of the engine's index) rather than a
graph search, so it works the same regardless of how well-tuned (or untuned) the
project's `BeamSearch` is.

Returns exactly what `SimilaritySearch.fft` itself does (`centers`, `nn`, `dists`, `ε`,
`costdists`, `costblocks` -- see its own docstring). `centers`/`nn` are internal `_id`s
(1-based position in the index), not hydrated with metadata -- look them up yourself
(e.g. via [`fetch_items`](@ref)) if you need it.
"""
function fft(handle::EmbeddedEngine, k::Integer; start::Int=0, verbose::Bool=false)
    engine = handle.engine
    IndexEngine.is_text_index(engine) && error("fft requires a dense (vector) index, but this project is a text index")
    length(engine.index) == 0 && error("fft requires a non-empty dense index")
    SimilaritySearch.fft(SimilaritySearch.distance(engine.index), SimilaritySearch.database(engine.index), k; start, verbose)
end

"""
    closestpairs(handle::EmbeddedEngine; k::Int=1, min_k::Int=max(k, 8)) -> Vector{Tuple{Int32,Int32,Float32}}

Runs `SimilaritySearch.closestpairs` synchronously against the project's dense index --
the `k` globally closest pairs among every item currently indexed (approximate if the
index itself is, e.g. an untuned `SearchGraph`), same in-process framing as
[`allknn`](@ref). Errors if the project is a text index or empty (same guard as
`allknn`).

Returns up to `k` `(i, j, dist)` tuples, ascending by distance -- `i`/`j` are internal
`_id`s, not hydrated with metadata.
"""
function closestpairs(handle::EmbeddedEngine; k::Int=1, min_k::Int=max(k, 8))
    engine = handle.engine
    IndexEngine.is_text_index(engine) && error("closestpairs requires a dense (vector) index, but this project is a text index")
    length(engine.index) == 0 && error("closestpairs requires a non-empty dense index")
    SimilaritySearch.closestpairs(engine.index, engine.ctx; k, min_k)
end

"""
    bichromatic_kclosestpairs(handle::EmbeddedEngine, B; k::Int=1, min_k::Int=max(k, 8)) -> Vector{Tuple{Int32,Int32,Float32}}

Runs `SimilaritySearch.bichromatic_kclosestpairs` synchronously between the project's own
dense index (playing the role of `idxA`, already indexed) and `B` -- a second, separate
collection of vectors (any `SimilaritySearch.AbstractDatabase`, e.g. a plain
`Vector{Vector{Float32}}` or `MatrixDatabase`) that is *not* itself indexed by this
project, or by anything else -- e.g. the same LSI space's vectors for a different half of
a corpus than the one this project indexes. Errors if the project is a text index or
empty (same guard as `allknn`).

There is deliberately no way to pass two `EmbeddedEngine` projects here: this embedded API
manages one project (one index) at a time, and `B` never needs its own project/persistence
of its own to take part in this -- it's queried directly, in memory, exactly as
`SimilaritySearch.bichromatic_kclosestpairs` itself expects.

Returns up to `k` `(i, j, dist)` tuples, ascending by distance -- `i` is this project's
own internal `_id` (in `idxA`), `j` is `B`'s own position (1-based, in `B`'s own order,
*not* related to any project's `_id` space) -- neither is hydrated with metadata.
"""
function bichromatic_kclosestpairs(handle::EmbeddedEngine, B; k::Int=1, min_k::Int=max(k, 8))
    engine = handle.engine
    IndexEngine.is_text_index(engine) && error("bichromatic_kclosestpairs requires a dense (vector) index, but this project is a text index")
    length(engine.index) == 0 && error("bichromatic_kclosestpairs requires a non-empty dense index")
    SimilaritySearch.bichromatic_kclosestpairs(engine.index, engine.ctx, B; k, min_k)
end
