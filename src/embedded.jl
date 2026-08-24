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
    SearchResult

One hit: the internal `_id` the engine ranked, the caller's own `doc_id` for it (`nothing`
when the item was appended without one, and when the hit is soft-deleted), the `distance`, and
whether the item is soft-deleted.

The field names match [`Schema.MetadataRecord`](@ref) on purpose. The named tuple this
replaced called the caller's external id `id` and the internal one `doc_id` -- exactly
backwards from the record, so `result.doc_id` and `record.doc_id` were different things and one
of them was always the wrong guess.

A soft-deleted hit is reported, not hidden (see [`IndexEngine.search_live`](@ref)), and carries
no `doc_id`: its metadata is not accessible through search.
"""
struct SearchResult
    _id::Int32
    doc_id::Union{String,Nothing}
    distance::Float32
    deleted::Bool
end

"""
    ExistsResult

What [`exists`](@ref) reports per queried id: the id as asked for, whether a record was found,
and whether it is soft-deleted.
"""
struct ExistsResult
    id::String
    exists::Bool
    deleted::Bool
end

"""
    KnnRow

One row of [`allknn`](@ref): item `_id` and its `k` nearest neighbours, ascending by distance.
`_id` and `neighbors` are internal ids -- 1-based positions in insertion order -- not `doc_id`s.
"""
struct KnnRow
    _id::Int32
    neighbors::Vector{Int32}
    dists::Vector{Float32}
end

"""
    FFTResult

What [`fft`](@ref) returns: the chosen `centers`, each item's nearest center (`nn`) and
distance to it (`dists`), the final covering radius `epsilon`, and the two cost counters
`SimilaritySearch.fft` reports.

A faithful, typed restatement of that function's own named tuple -- `epsilon` spelled out
rather than `ε`, and ids as `Int32` to match every other id this package hands back.
"""
struct FFTResult
    centers::Vector{Int32}
    nn::Vector{Int32}
    dists::Vector{Float32}
    epsilon::Float32
    costdists::Int
    costblocks::Int
end

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
(`IndexEngine.build_bm25invertedfile`/`build_textinvertedfile`, used by [`open_project`](@ref))
rebuilds the whole index by replaying every saved object back through the library's own
insertion -- see `Persistence.InvertedFileObjectStore`'s docstring for the scaling
trade-off that implies (fine up to a few million documents; not a design for
billion-document corpora needing disk-backed posting lists).
"""
_invertedfile_on_change(obj_store::Persistence.InvertedFileObjectStore) =
    (index, sp, ep) -> Persistence.append_objects!(obj_store, sp, IndexEngine.invertedfile_objects(index, sp, ep))

"""
    create_project(workdir, dataset; index_type=SearchGraph, textmodel=nothing, distance=nothing, minrecall=0.9, schema_version=1) -> EmbeddedEngine

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
`textmodel` says where a *text* project's vocabulary comes from, and a text project cannot be
created without it (see `IndexEngine.AbstractTextModelSpec` for why it has no default). Three
forms, in the order worth preferring them:

- `textmodel=DefaultProfile(:es)` — **recommended.** The published profile for a language,
  refitted to this project's own corpus at the first [`index!`](@ref index!(::EmbeddedEngine))
  call: a vocabulary fitted over a whole Wikipedia edition, with its stopword set, lemma map and
  expansion network, adapted without fitting an embedding. Resolved from the profile library
  `textsearch install` maintains; `IndexEngine.default_profile_path` says so, and how to install
  it, when the one you asked for is not there.
- `textmodel=BaseProfile(load_profile("path/to/profile.zip"))` — a profile you already have,
  used as it is. The project is trained before its first item is staged.
- `textmodel=FitFromCorpus(TextConfig(language=:es); min_ndocs=3)` — deliberately no base
  profile: fit one at the first `index!` call from at most `max_documents` of this project's own
  corpus, delegated whole to `TextSearch.fit_profile`. Bounded in cost and thin in vocabulary,
  so terms appended later that the sample never held are dropped from then on.

Passing a `textmodel` to a *dense* project is an error rather than an ignored keyword: there is
no reading under which it does anything, and swallowing it silently is how a project ends up
not being the kind its author thought it was. The spec is persisted with the project and
restored verbatim by [`open_project`](@ref).

`schema_version` is stamped onto every `Schema.MetadataRecord` [`append_items!`](@ref)
writes for this project's lifetime (not persisted/restored itself -- like `minrecall`
before it was carried on the engine, a caller reopening this same project later must pass
the same `schema_version` again to keep tagging new records consistently with old ones);
it's never validated against `meta`'s actual shape, purely a version tag for the caller's
own interpretation of it.

The engine's index itself is persisted incrementally as it grows, not rewritten wholesale
on every `append_items!` call: for a `SearchGraph`, every [`index!`](@ref index!(::EmbeddedEngine))
call saves just the direct-links-only block it just built (see
[`_searchgraph_on_change`](@ref)); for a `BM25InvertedFile`/`InvertedFile` project, every
`index!` call likewise saves just the newly-encoded objects it just indexed (see
[`_invertedfile_on_change`](@ref)) -- both engine kinds have a staging split
([`append_items!`](@ref) itself durably stages raw vectors/text right away, see that
function's docstring). Only `GenericEngine` (`ExhaustiveSearch`/`ParallelExhaustiveSearch`,
no staging split at all) instead flags a pending whole-index save that happens right after
each `add_item!` call returns (see [`_maybe_flush_index!`](@ref)); [`close_project!`](@ref)
forces one final flush of that so nothing recent is lost -- the other two kinds never have
anything left to flush there, since each block is already durable the instant it's saved.
"""
function create_project(workdir::String, dataset::String; index_type::Type=SearchGraph, distance=nothing, minrecall::Union{Nothing,Real}=0.9,
                        textmodel::Union{Nothing,IndexEngine.AbstractTextModelSpec}=nothing, schema_version::Int=1)
    # Before anything is created on disk: `create_engine` checks this too, but only after the
    # project's directory exists and RocksDB's write lock is held, so a call that fails here
    # would otherwise leave both behind.
    IndexEngine.validate_textmodel(index_type, textmodel)
    dir = joinpath(workdir, dataset)
    mkpath(dir)
    project = Project.open_project(dir, dataset; extra_cf_names=[Persistence.ENGINE_CF, Persistence.ADJACENCY_CF, Persistence.INVFILE_DB_CF, Persistence.STAGED_TEXT_CF])
    store = Persistence.open_engine_store(project.db)
    pending_flush = Ref(false)
    dense_vectors = Ref{Union{Nothing,MMapMatrixDatabase}}(nothing)
    on_change = if index_type === SearchGraph
        _searchgraph_on_change(store, Persistence.open_adjacency_store(project.db))
    elseif IndexEngine.is_text_index_type(index_type)
        _invertedfile_on_change(Persistence.open_invertedfile_object_store(project.db))
    else
        (_, __, ___) -> (pending_flush[] = true)
    end
    engine = distance === nothing ?
        IndexEngine.create_engine(index_type; minrecall, textmodel, on_change) :
        IndexEngine.create_engine(index_type; distance, minrecall, textmodel, on_change)
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
    project = Project.open_project(dir, dataset; read_only, extra_cf_names=[Persistence.ENGINE_CF, Persistence.ADJACENCY_CF, Persistence.INVFILE_DB_CF, Persistence.STAGED_TEXT_CF])
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
        staged_store = Persistence.open_staged_text_store(project.db)
        state = (
            kind=kind,
            profile=Persistence.load_field(store, :profile),
            fitspec=Persistence.load_field(store, :fitspec),
            object_blocks=Persistence.load_object_blocks(obj_store),
            staged=vcat(String[], Persistence.load_staged_text_blocks(staged_store)...),
            deleted_ids=Persistence.load_field(store, :deleted_ids),
        )
        IndexEngine.restore_engine(state; on_change=_invertedfile_on_change(obj_store))
    elseif kind === IndexEngine.InvertedFileEngine
        obj_store = Persistence.open_invertedfile_object_store(project.db)
        staged_store = Persistence.open_staged_text_store(project.db)
        state = (
            kind=kind,
            profile=Persistence.load_field(store, :profile),
            fitspec=Persistence.load_field(store, :fitspec),
            distance=Persistence.load_field(store, :distance),
            object_blocks=Persistence.load_object_blocks(obj_store),
            staged=vcat(String[], Persistence.load_staged_text_blocks(staged_store)...),
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
    index!(handle::EmbeddedEngine)

Catches up encoding/indexing over whatever's been staged via [`append_items!`](@ref)
since the last call (or since creation) -- one uniform entry point for every index kind
with a staging split (`SearchGraph`, `BM25InvertedFile`, `InvertedFile`): idempotent, safe
to call repeatedly, only ever processes the backlog (see
`IndexEngine.index!(engine::IndexEngine.SearchGraphEngine)`'s docstring for the exact
contract, shared verbatim by the text engines). Nothing staged via `append_items!` becomes
visible to [`search`](@ref)/[`ftsearch`](@ref)/[`allknn`](@ref)/[`fft`](@ref)/
[`closestpairs`](@ref)/[`bichromatic_kclosestpairs`](@ref) until this runs at least once.

For a text project created with a `FitFromCorpus`, the first call additionally runs that fit
(producing a `TextSearch.TextProfile`: vocabulary, weights and lineage) over every item staged
so far and builds the real index against it -- there is no separate training entry point;
whatever you've `append_items!`-ed before the first `index!` call *is* the training corpus,
and tokens absent from it are out-of-vocabulary forever after. That one-time transition
(`:profile`) is persisted immediately after it happens; every later call only ever touches
the encoded posting-list blocks already covered by the engine's own incremental `on_change`,
so this does not re-save `:deleted_ids` or anything else on every catch-up call.

A project created with a `BaseProfile` (see [`create_project`](@ref)) was already trained
before its first item was staged, so there is no such transition and every call here is a
pure catch-up.

Errors for a `GenericEngine` (`ExhaustiveSearch`/`ParallelExhaustiveSearch`) project --
the one index kind with no staging split at all, since it always evaluates directly
against `db`; there is never a backlog for this to catch up.
"""
function index!(handle::EmbeddedEngine)
    engine = handle.engine
    if engine isa IndexEngine.SearchGraphEngine
        IndexEngine.index!(engine)
    elseif IndexEngine.is_text_index(engine)
        # `:fitspec`/`:distance` were already written by create_project's full snapshot_state
        # save, and neither ever changes afterwards -- only `:profile` can go from nothing to a
        # fitted model, and only once, so that is the only field this has to write back.
        just_fitted = IndexEngine.text_profile(engine) === nothing
        IndexEngine.index!(engine)
        just_fitted && Persistence.save_field!(handle.store, :profile, IndexEngine.text_profile(engine))
    else
        error("index!(handle) is not supported for this project's index kind -- ExhaustiveSearch/ParallelExhaustiveSearch have no staging split, items are searchable immediately on append_items!")
    end
    return handle
end

"""
    append_items!(handle::EmbeddedEngine, items) -> Int

Appends a batch of typed items -- [`DenseItem`](@ref Schema.DenseItem)s for a dense project,
[`TextItem`](@ref Schema.TextItem)s for a text one, each already carrying its own `doc_id`,
`keywords`, `refs` and `meta`. Returns the number inserted, which is always `length(items)`:
every item in the batch is inserted, and an item of the wrong kind for this project raises
rather than being skipped.

That is the change from the dictionary form this replaced, and it is worth being explicit
about. Appending `Dict("txt" => "...")` to a text project used to insert nothing and report
zero, because the item had no `"text"` key and the loop skipped it -- a typo in a key name
produced a silent no-op and a count the caller had to think to check. There is no key to
misspell now, and nothing is skipped.

For a dense (`SearchGraph`) project or a text (`BM25InvertedFile`/`InvertedFile`) project
alike, this only *stages* items -- durably (see "Performance" below) but without any
graph-linking/encoding work -- so freshly appended items are *not* yet visible to
[`search`](@ref)/[`ftsearch`](@ref)/[`allknn`](@ref)/etc. until an explicit
[`index!`](@ref index!(::EmbeddedEngine)) call catches up the backlog (for a text
project's very first `index!` call, that also trains its `Vocabulary` -- there's no
separate training entry point or "must be trained first" error here anymore).
`GenericEngine` (`ExhaustiveSearch`/`ParallelExhaustiveSearch`) is the one engine kind that
still indexes synchronously, on every `add_item!`, since it has no staging split at all
(see `IndexEngine.index!(engine::IndexEngine.SearchGraphEngine)`'s docstring for why
`SimilaritySearch.jl`/`TextSearch.jl` support the split for the other three).

Persistence: for a dense project, every batch's raw vectors are appended directly to the
project's `MMapMatrixDatabase` (see [`index!`](@ref index!(::EmbeddedEngine))'s docstring
and the "Performance" note below); for a text project, every batch's raw text is appended
directly to its own `Persistence.StagedTextStore` -- both happen right here, at stage
time, not via `on_change`, since an item is already durable-worthy the moment it's staged,
long before any graph-linking/encoding happens. Once an [`index!`](@ref index!(::EmbeddedEngine))
call actually encodes/indexes a text project's backlog, `IndexEngine.CallbackLog` persists
each newly-encoded object incrementally (see [`_invertedfile_on_change`](@ref)).
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
# A project indexes one kind of thing, and the item type says which. Checked per item rather
# than once per batch so a heterogeneous `Vector{AbstractItem}` is caught on the offending
# element instead of on whatever happened to be first.
_reject_item(engine, item) = error(
    "this project indexes $(IndexEngine.is_text_index(engine) ? "text" : "vectors"), " *
    "so it takes $(IndexEngine.is_text_index(engine) ? "TextItem" : "DenseItem")s; got a $(typeof(item))" *
    (item.doc_id === nothing ? "" : " (doc_id $(item.doc_id))"))

_check_item(engine, item::Schema.TextItem) =
    IndexEngine.is_text_index(engine) || _reject_item(engine, item)
_check_item(engine, item::Schema.DenseItem) =
    IndexEngine.is_text_index(engine) && _reject_item(engine, item)

function append_items!(handle::EmbeddedEngine, items::AbstractVector{<:Schema.AbstractItem})
    engine = handle.engine
    project = handle.project
    is_text = IndexEngine.is_text_index(engine)
    is_dense_graph = engine isa IndexEngine.SearchGraphEngine
    staged_vectors = Vector{Float32}[]
    staged_texts = String[]
    text_sp = is_text ? _current_size(engine) + 1 : 0

    for item in items
        _check_item(engine, item)
        p = Schema.payload(item)
        IndexEngine.add_item!(engine, p)
        if is_text
            push!(staged_texts, p)
        elseif is_dense_graph
            push!(staged_vectors, p)
        end
        _maybe_flush_index!(handle)

        _id = Int32(_current_size(engine))
        # An empty `meta` is stored as nothing rather than as an empty JSON object: there is
        # no difference to read back (`get_meta` answers `nothing` either way) and one of them
        # is a write that never had to happen.
        meta = isempty(item.meta) ? nothing : item.meta
        put_metadata!(project, Schema.metadata_record(item, _id, handle.schema_version), meta)
    end

    if is_dense_graph && !isempty(staged_vectors)
        if handle.dense_vectors[] === nothing
            handle.dense_vectors[] = MMapMatrixDatabase(Persistence.dense_vectors_path(handle.dir), length(staged_vectors[1]), Float32)
        end
        SimilaritySearch.append_items!(handle.dense_vectors[], staged_vectors)
    end

    if is_text && !isempty(staged_texts)
        Persistence.append_staged_texts!(Persistence.open_staged_text_store(project.db), text_sp, staged_texts)
    end

    return length(items)
end

"""
    append_items!(handle::EmbeddedEngine, item::Schema.AbstractItem) -> Int

One item, for a caller that has one. Note the performance warning above: this is the
single-item call, and for a dense project it costs one `fsync` for one vector.
"""
append_items!(handle::EmbeddedEngine, item::Schema.AbstractItem) = append_items!(handle, [item])

"""
    _current_size(engine::IndexEngine.AbstractSearchEngine) -> Int

The count that determines the next item's `_id`: for a `SearchGraphEngine`, the number of
*staged* vectors (`length(database(engine.index))`, i.e. `engine.index.db`'s own count),
and for a `BM25Engine`/`InvertedFileEngine`, the number of *staged* texts
(`length(engine.staged)`) -- since [`append_items!`](@ref) only stages for any of these
three, `length(engine.index)` itself (the encoded/indexed count) would lag behind and hand
out the wrong, already-taken `_id`s. `GenericEngine` (`ExhaustiveSearch`/
`ParallelExhaustiveSearch`) is the one engine kind that still indexes synchronously on
`add_item!`, so `length(engine.index)` already reflects the item just added there.
"""
_current_size(engine::IndexEngine.SearchGraphEngine) = length(SimilaritySearch.database(engine.index))
_current_size(engine::Union{IndexEngine.BM25Engine, IndexEngine.InvertedFileEngine}) = length(engine.staged)
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
    results = SearchResult[]
    for (_id, dist, deleted) in zip(res_knn.id, res_knn.dist, res_knn.deleted)
        if deleted
            # Metadata for a soft-deleted _id is not accessible through search, so there is no
            # doc_id to report -- `nothing`, rather than the internal id stringified into the
            # external id's field, which is what the named-tuple version did and which a caller
            # had no way to tell apart from a real doc_id that happened to look like a number.
            push!(results, SearchResult(Int32(_id), nothing, Float32(dist), true))
            continue
        end
        record = get_metadata(project, _id)
        push!(results, SearchResult(Int32(_id), record === nothing ? nothing : record.doc_id,
                                    Float32(dist), false))
    end
    return results
end

"""
    _resolve_record(project, raw_id) -> Union{Schema.MetadataRecord, Nothing}

The record `raw_id` names, whether it is an internal `_id` or a caller-supplied `doc_id`.

Tries the numeric reading first, because it is a direct key lookup, and falls back to
`Project.find_by_doc_id`'s linear scan. A numeric-looking `doc_id` therefore resolves as an
`_id` if one exists with that value -- an ambiguity inherent to accepting both in one argument,
and the reason [`fetch_items`](@ref) and [`exists`](@ref) share this function rather than each
inventing its own precedence.
"""
function _resolve_record(project::Project.ProjectManager, raw_id)
    id_str = string(raw_id)
    maybe_int = tryparse(Int, id_str)
    if maybe_int !== nothing
        record = get_metadata(project, maybe_int)
        record === nothing || return record
    end
    find_by_doc_id(project, id_str)
end

"""
    search(handle::EmbeddedEngine, vector, k::Int=10; filter=nothing) -> Vector{SearchResult}

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
[`IndexEngine.search_live`](@ref)), so `doc_id` is `nothing` for them.
Getting `k` *live* results back is a paging concern for a layer above this one (e.g. a
server walking successive windows via a cursor), not something this function does itself.

`minrecall`, for a `SearchGraphEngine`, searches at (approximately) that target recall
using a calibrated `BeamSearch` from `engine.opt_beamsearch` instead of its current
default (see [`IndexEngine.search_live`](@ref)) -- if that table is still empty, this
triggers a one-off `calibrate!` over `IndexEngine.DEFAULT_MINRECALL_LEVELS` and persists
the resulting `opt_beamsearch` so that calibration isn't silently repeated on every future
search. Ignored for any other engine kind.

Returns a `Vector{`[`SearchResult`](@ref)`}` -- `_id` is the internal id, `doc_id` the
caller's own, and note that those two field names sit the opposite way round from the named
tuple this replaced.
"""
function search(handle::EmbeddedEngine, vector, k::Int=10; filter=nothing, minrecall=nothing)
    query = convert(Vector{Float32}, vector)
    needs_save = minrecall !== nothing && handle.engine isa IndexEngine.SearchGraphEngine && isempty(handle.engine.opt_beamsearch)
    res_knn = filter === nothing ?
        IndexEngine.search_live(handle.engine, query, k; minrecall) :
        _search_with_filter(handle.engine, handle.project, query, k, filter; minrecall)
    needs_save && Persistence.save_field!(handle.store, :opt_beamsearch, handle.engine.opt_beamsearch)
    return _hydrate_results(handle.project, res_knn)
end

"""
    ftsearch(handle::EmbeddedEngine, text, k::Int=10; policy=QueryPolicy()) -> Vector{SearchResult}

Text search against a bm25/weighted-inverted-file project (mirrors `Server.handle_ftsearch`).
Same soft-delete marker behavior as [`search`](@ref); `minrecall` is accepted only for
parity with `search` and is a no-op here since no text engine has a `BeamSearch` to
calibrate.

`policy::TextSearch.QueryPolicy` says how to treat the query: whether to correct its
spelling against the project's vocabulary (`correction=:auto` by default, `:off` to search
exactly what was typed, `:always` to bridge every token) and whether to widen it with the
profile's query-expansion network (`expansion`, `expansion_k`). Both are guesses about
intent rather than properties of the index, which is why they travel with the query instead
of being fixed at project creation. The default is inert for a project fitted under the
default `TextConfig()`, whose policy already folds case and diacritics; it starts mattering
for one built on a profile that preserves them. Use [`ftexplain`](@ref) to see what a query
was actually searched as.
"""
function ftsearch(handle::EmbeddedEngine, text::AbstractString, k::Int=10; minrecall=nothing, policy::QueryPolicy=QueryPolicy())
    res_knn = IndexEngine.search_live(handle.engine, text, k; minrecall, policy)
    return _hydrate_results(handle.project, res_knn)
end

"""
    text_profile(handle::EmbeddedEngine) -> Union{Nothing, TextSearch.TextProfile}

The text model this project searches with: the one handed to [`create_project`](@ref), or the
one its first [`index!`](@ref index!(::EmbeddedEngine)) call fitted from the staged corpus.
`nothing` for a dense project, and for a text project that has not been indexed yet.

Worth reaching for after a `FitFromCorpus`: `save_profile(dir, text_profile(h))` writes the
fitted vocabulary and weights out as a portable profile, so a sibling project can be created
against the same model (`textmodel=BaseProfile(load_profile(dir))`) instead of fitting its own
and ending up with a different vocabulary over the same language.

Written as an explicit extension of `IndexEngine.text_profile` rather than as a bare
`text_profile(handle::EmbeddedEngine) = ...`. The bare form compiles -- `using .IndexEngine`
makes the name available but not yet resolved in this scope, so Julia quietly creates a
*second*, unrelated function shadowing the first -- and that is the hazard, not a convenience:
the two would then be one name with two disjoint method tables, and which one a call site got
would depend on whether it had already touched the imported binding. Extending keeps a single
generic function whose methods cover an engine and a handle alike (the same reasoning as the
`open_project` note in `SimilaritySearchEngine.jl`).
"""
IndexEngine.text_profile(handle::EmbeddedEngine) = IndexEngine.text_profile(handle.engine)

"""
    ftexplain(handle::EmbeddedEngine, text; policy=QueryPolicy()) -> Vector{String}

One human-readable line per query token that [`ftsearch`](@ref) would search as something
other than what was typed -- `"musica appears in only 9 documents, searched as música
instead"` -- and an empty vector when the query is searched verbatim.

A search that silently substitutes what was asked for owes the person a way to see it: this
is the "showing results for ..." half, and `QueryPolicy(correction=:off)` is the "search
instead for ..." half. Resolves the query the same way `ftsearch` does under the same
`policy`, so the two never disagree; it does not run the search itself.

Errors for a project whose text engine has not been trained yet, and for a dense project
(which has no query text to resolve).
"""
function ftexplain(handle::EmbeddedEngine, text::AbstractString; policy::QueryPolicy=QueryPolicy())
    engine = handle.engine
    IndexEngine.is_text_index(engine) ||
        error("ftexplain is only meaningful for a text project (BM25InvertedFile/InvertedFile); this one is $(typeof(engine))")
    return TextSearch.explain(IndexEngine.resolve_query(engine, text, policy).resolution)
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
    fetch_items(handle::EmbeddedEngine, ids) -> Vector{Schema.StoredItem}

Batch retrieval by id -- each element of `ids` may be the internal `_id` or the
caller-supplied `doc_id` (see [`_resolve_record`](@ref)). Mirrors `Server.handle_fetch`; an id
that resolves to nothing is skipped, so the result can be shorter than `ids`.

Each item comes back whole: the record's fixed fields, its `meta` as a plain
`Dict{String,Any}`, and `payload` -- the very text or vector that was indexed, read from the
engine via [`IndexEngine.stored_payload`](@ref).

Both halves of that are changes from what this used to return. It used to hand back a
`Dict{String,Any}` per item with the record's fields merged into the decoded `meta`, which
meant a `meta` key called `"doc_id"` silently overwrote the record's own, and it could not
return the payload at all: `"text"` was a reserved key stripped on the way in, so a text
project's fetch answered with everything about a paragraph except the paragraph.
"""
function fetch_items(handle::EmbeddedEngine, ids)
    project = handle.project
    results = Schema.StoredItem[]
    for raw_id in ids
        record = _resolve_record(project, raw_id)
        record === nothing && continue
        meta = something(get_meta(project, record._id; lazy=false), Dict{String,Any}())
        push!(results, Schema.StoredItem(record,
                                         IndexEngine.stored_payload(handle.engine, record._id),
                                         meta))
    end
    return results
end

"""
    exists(handle::EmbeddedEngine, ids) -> Vector{ExistsResult}

For each id in `ids` (internal `_id` or caller-supplied `doc_id`, see
[`_resolve_record`](@ref)), reports whether a record exists and, if so, whether it's been
soft-deleted. Mirrors `Server.handle_exists`. One result per queried id, in order, including
the ones that were not found -- unlike [`fetch_items`](@ref), which drops those.
"""
function exists(handle::EmbeddedEngine, ids)
    project = handle.project
    engine = handle.engine
    results = ExistsResult[]
    for raw_id in ids
        record = _resolve_record(project, raw_id)
        found = record !== nothing
        deleted = found && (record._id in engine.deleted_ids)
        push!(results, ExistsResult(string(raw_id), found, deleted))
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
    allknn(handle::EmbeddedEngine; k=10) -> Vector{KnnRow}

Runs `SimilaritySearch.allknn` synchronously against the project's dense index -- no
Job/spool machinery at all, unlike the HTTP API's `POST /api/v1/jobs/allknn` (PLAN.md
§8.5's "runs in-process" framing for the embedded surface). Errors if the project is a text
index or empty (mirrors `cli_handlers.jl`'s `_require_dense`). Returns one [`KnnRow`](@ref)
per item, its `_id` being the item's 1-based position in insertion order (matching
`execute_allknn`'s own output convention), not its `doc_id`.

A row can be shorter than `k`: the library pads unfilled neighbour slots with id `0`, and this
stops at the first one rather than reporting a neighbour that does not exist.
"""
function allknn(handle::EmbeddedEngine; k::Int=10)
    engine = handle.engine
    IndexEngine.is_text_index(engine) && error("allknn requires a dense (vector) index, but this project is a text index")
    length(engine.index) == 0 && error("allknn requires a non-empty dense index")

    ids, dists = SimilaritySearch.allknn(engine.index, engine.ctx, k)
    n = size(ids, 2)
    results = KnnRow[]
    for i in 1:n
        neighbor_ids = Int32[]
        neighbor_dists = Float32[]
        for j in 1:k
            ids[j, i] == 0 && break
            push!(neighbor_ids, Int32(ids[j, i]))
            push!(neighbor_dists, Float32(dists[j, i]))
        end
        push!(results, KnnRow(Int32(i), neighbor_ids, neighbor_dists))
    end
    return results
end

"""
    fft(handle::EmbeddedEngine, k::Integer; start::Int=0, verbose::Bool=false) -> FFTResult

Runs `SimilaritySearch.fft`'s Farthest-First-Traversal synchronously against the
project's dense index -- picks `k` well-separated items (e.g. as diverse candidate
cluster centers), the same "runs in-process, no job queue" framing as [`allknn`](@ref).
Errors if the project is a text index or empty (same guard as `allknn`). Unlike
`allknn`/`closestpairs`/`bichromatic_kclosestpairs`, this is a brute-force sweep over the
raw vectors (`SimilaritySearch.distance`/`database` of the engine's index) rather than a
graph search, so it works the same regardless of how well-tuned (or untuned) the
project's `BeamSearch` is.

Returns an [`FFTResult`](@ref) -- every field `SimilaritySearch.fft` itself reports (see its
docstring), restated as a named type with `ε` spelled `epsilon` and ids as `Int32`, matching
every other id this package hands back. `centers`/`nn` are internal `_id`s (1-based position in
the index), not hydrated with metadata -- look them up yourself (e.g. via
[`fetch_items`](@ref)) if you need it.
"""
function fft(handle::EmbeddedEngine, k::Integer; start::Int=0, verbose::Bool=false)
    engine = handle.engine
    IndexEngine.is_text_index(engine) && error("fft requires a dense (vector) index, but this project is a text index")
    length(engine.index) == 0 && error("fft requires a non-empty dense index")
    r = SimilaritySearch.fft(SimilaritySearch.distance(engine.index), SimilaritySearch.database(engine.index), k; start, verbose)
    FFTResult(Int32.(r.centers), Int32.(r.nn), Float32.(r.dists), Float32(r.ε),
              Int(r.costdists), Int(r.costblocks))
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
