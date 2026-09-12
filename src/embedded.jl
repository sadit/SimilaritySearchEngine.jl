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
    CenterSelectionResult

What [`fft`](@ref) and [`dnet`](@ref) return: the chosen `centers`, which center each item was
assigned to (`assign`) and its distance to it (`assigndist`), the two radii describing the
selection (`covering` and `separation`), and the two cost counters.

A faithful, typed restatement of `SimilaritySearch.CenterSelection`, with ids as `Int32` to
match every other id this package hands back.

- `centers` are internal `_id`s (1-based position in the index), not hydrated with metadata.
- `assign[i]` is a **position in `centers`**, from `1` to `length(centers)` -- not an `_id`.
  The `_id` of item `i`'s center is `centers[assign[i]]`.
- `covering` is the largest `assigndist`: the radius the centers need to cover everything.
- `separation` is the smallest distance between two centers. These two are different numbers
  and used to be conflated under one name; see `SimilaritySearch.CenterSelection`.
"""
struct CenterSelectionResult
    centers::Vector{Int32}
    assign::Vector{Int32}
    assigndist::Vector{Float32}
    covering::Float32
    separation::Float32
    costdists::Int
    costblocks::Int
end

const FFTResult = CenterSelectionResult

"""
    NearDupResult

What [`neardup`](@ref) returns: the selected `centers` (the ϵ-net of surviving non-duplicate
objects), which center covers each object (`assign`) and its distance to it (`assigndist`),
the threshold radius `epsilon` and the maximum covering radius `covering`, plus the cost counters.

A faithful, typed restatement of `SimilaritySearch.NearDupSelection`, with ids as `Int32` to
match every other id this package hands back.
"""
struct NearDupResult
    centers::Vector{Int32}
    assign::Vector{Int32}
    assigndist::Vector{Float32}
    covering::Float32
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
into field-by-field, and `pending_flush` -- a flag a `DenseEngine{<:ExactBackend}`'s
`SimilaritySearch.CallbackLog` callback sets (see [`create_project`](@ref)) that this module
checks and clears from *outside* the engine's own insertion call, once it's known safe to
do so (see [`_maybe_flush_index!`](@ref); unused by every backend that persists its index
incrementally by itself, which is all of them except the exact ones -- see
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
    read_only::Bool
    wrote::Base.RefValue{Bool}
end

"""
    _maybe_flush_index!(handle::EmbeddedEngine)

Persists `handle.engine.backend.index` if (and only if) `handle.pending_flush[]` is set, then
clears the flag. Callers must only call this from a point where the just-finished mutation
is fully done -- in particular, *not* from inside an `SimilaritySearch.CallbackLog` callback
itself. Right after an `IndexEngine.add_item!`/`index!` call returns to this
module is always safe. A no-op for `DenseEngine{GraphBackend}`/`FullTextEngine`:
each has its own dedicated incremental `on_change` (see [`_searchgraph_on_change`](@ref)/
[`_invertedfile_on_change`](@ref)) that never touches `pending_flush`; only `DenseEngine{<:ExactBackend}`
(`ExhaustiveSearch`/`ParallelExhaustiveSearch`) still uses this whole-index path.
"""
function _maybe_flush_index!(handle::EmbeddedEngine)
    if handle.pending_flush[]
        handle.pending_flush[] = false
        Persistence.save_field!(handle.store, :index, handle.engine.backend.index)
    end
end

"""
    _searchgraph_on_change(store::Persistence.EngineStore, adj_store::Persistence.AdjacencyStore) -> Function

The `on_change` callback for a `DenseEngine{GraphBackend}`: fires from *inside*
`IndexEngine.index!(engine::IndexEngine.DenseEngine{GraphBackend})` (never from `add_item!`/
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
`SimilaritySearch.CallbackLog` hands over at that point, and reconstruction
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

The `on_change` callback for a `FullTextEngine`: on every
`push_item!`/`append_items!` report for range `sp:ep`, saves *only* that range's raw
indexed objects (bags-of-words or `SparseVector`s -- see `IndexEngine.invertedfile_objects`)
as a new, never-again-rewritten block in `obj_store` (`Persistence.append_objects!`) --
never the whole (growing) index as one value. Safe to do synchronously, right inside the
callback (unlike the whole-index save `DenseEngine{<:ExactBackend}` uses, see
[`_maybe_flush_index!`](@ref)): an inverted file's `LOG` only fires after a call's
mutation is fully done, so there's no partial-state hazard here the way there is for a
`SearchGraph` (see `SimilaritySearch.CallbackLog`'s docstring). Reconstruction
(`IndexEngine.build_sparseinvertedfile`, used by [`open_project`](@ref)) rebuilds the whole
index by replaying every saved object back through the library's own insertion -- see
`Persistence.InvertedFileObjectStore`'s docstring for the scaling trade-off that implies.

Only a sparse project is wired to this now: a text project persists its index
(`Persistence.InvertedIndexStore`) and has no observer of its own.
"""
_invertedfile_on_change(obj_store::Persistence.InvertedFileObjectStore) =
    (index, sp, ep) -> Persistence.append_objects!(obj_store, sp, IndexEngine.invertedfile_objects(index, sp, ep))

"""
    _invfile_store(project) -> Persistence.InvertedIndexStore

The project's persisted inverted index (posting lists and per-document vectors). See
`Persistence.INVFILE_POSTINGS_CF`.
"""
_invfile_store(project::Project.ProjectManager) = Persistence.open_inverted_index_store(project.db)

"""
    _invfile_adj_factory(store; maxlists, baselists) -> Function

The `vocsize -> AbstractAdjList` an inverted file is built around when its posting lists live
in `store` instead of in memory. `IndexEngine` calls it when it constructs the index and never
looks inside the result -- see `IndexEngine._text_index`.
"""
_invfile_adj_factory(store::Persistence.InvertedIndexStore; maxlists::Int, baselists::Int) =
    vocsize -> Persistence.LazyPostings(store, vocsize, maxlists, baselists)

"""
    _assemble_text_index(profile, kind, distance, store, adj_factory) -> AbstractInvertedFile

Rebuilds a queryable text index from what was persisted, without reindexing anything.

The cheap, small half is reconstructed from the profile -- vocabulary, scorer or weighting
model, and the query pipeline all come from the library's own profile-taking constructor, which
is also what derives the spelling-variant map once instead of per query. The large half comes
from storage: document vectors read into memory, posting lists left on disk behind
`adj_factory`'s adjacency list, and the per-document bookkeeping each backend keeps
(`doclens` for BM25, `sizes` for the weighted one) recomputed from those vectors rather than
stored a second time where the two could drift.

Measured on 265k Gutenberg paragraphs, 2026-09-11: 1.1s against 20.4s to recompute a BM25
index from the raw objects, and that is *with* the document vectors loaded eagerly.
"""
function _assemble_text_index(profile::TextProfile, kind::Type, distance,
                              store::Persistence.InvertedIndexStore, adj_factory::Function)
    docvecs = Persistence.load_invfile_docvecs(store)
    if kind === BM25InvertedFile
        template = BM25InvertedFile(profile)
        doclens = Int32[Int32(sum(v.nzval; init=UInt32(0))) for v in docvecs]
        return BM25InvertedFile(template.voc, template.bm25, adj_factory(length(template.adj)),
                                doclens, VectorDatabase(docvecs), Ref(Int64(length(docvecs))),
                                template.query)
    end
    template = TextInvertedFile(profile; dist=distance)
    inner = template.invfile
    sizes = UInt32[UInt32(Persistence.docvec_nnz(v)) for v in docvecs]
    rebuilt = InvertedFile(inner.dist, adj_factory(length(inner.adj)), sizes,
                                            VectorDatabase(docvecs), Ref(Int64(length(docvecs))))
    TextInvertedFile(template.model, rebuilt, template.query)
end

"""
    create_project(workdir, dataset; engine=DenseEngine, backend=nothing, distance=nothing,
                   minrecall=0.9, dimension=nothing, textmodel=nothing, schema_version=1) -> EmbeddedEngine

Creates a brand-new project directly on disk at `<workdir>/<dataset>`, with no HTTP server or
CLI subprocess involved -- the same on-disk layout `similarity-search build` already produces,
so a project created this way can later be inspected/rebuilt/dumped by the CLI, or reopened by
[`open_project`](@ref).

Two keywords say what the project *is*, and they are separate on purpose:

- `engine` is what it holds -- `DenseEngine` (dense vectors, the default), `SparseEngine`
  (sparse vectors you encoded) or `FullTextEngine` (text this package encodes). It also fixes
  the item type [`append_items!`](@ref) will accept: `DenseItem`, `SparseItem`, `TextItem`.
- `backend` is the index that holds it, defaulting to that engine's own
  (`IndexEngine.default_backend`): a `SearchGraph`, an `InvertedFile`, a `BM25InvertedFile`.
  `IndexEngine.BACKENDS` is the table of legal pairings, and an illegal one is refused here
  with a message naming the alternatives rather than failing later on a `MethodError`.

So `create_project(workdir, "ds")` is a dense project on a self-tuning `SearchGraph`, and every
other shape is one or two keywords away:

```julia
create_project(w, ds; engine=DenseEngine,    backend=ExhaustiveSearch)
create_project(w, ds; engine=SparseEngine,   dimension=50_000)
create_project(w, ds; engine=FullTextEngine, backend=TextInvertedFile,
                      textmodel=DefaultProfile(:es))
```

`dimension` is required for a sparse project and refused for any other. An `InvertedFile` is a
fixed array of posting lists, so its size is structural rather than descriptive: it has to be
known before the first item, and every `SparseItem` appended has to agree with it.

`distance`, left at its default `nothing`, defers to whichever default
`IndexEngine.default_distance` names for the chosen backend (`SqL2()` for the dense ones,
`NormCosine()` for an `InvertedFile`) instead of this function imposing one blanket default
across every index kind. `minrecall` is the target recall a `DenseEngine{GraphBackend}` autotunes
`BeamSearch` toward as it grows (see `IndexEngine.DenseEngine{GraphBackend}`); it must be given here,
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

Passing a `textmodel` to a *dense or sparse* project is an error rather than an ignored keyword:
there is no reading under which it does anything, and swallowing it silently is how a project
ends up not being the kind its author thought it was. That check is the engine's rather than the
backend's, because `InvertedFile` is a legal backend for both a sparse and a text project, so the
backend alone cannot answer it. The spec is persisted with the project and restored verbatim by
[`open_project`](@ref).

`schema_version` is stamped onto every `Schema.MetadataRecord` [`append_items!`](@ref)
writes for this project's lifetime (not persisted/restored itself -- like `minrecall`
before it was carried on the engine, a caller reopening this same project later must pass
the same `schema_version` again to keep tagging new records consistently with old ones);
it's never validated against `meta`'s actual shape, purely a version tag for the caller's
own interpretation of it.

The engine's index itself is persisted incrementally as it grows, not rewritten wholesale
on every `append_items!` call: for a `SearchGraph`, every [`index!`](@ref index!(::EmbeddedEngine))
call saves just the direct-links-only block it just built (see
[`_searchgraph_on_change`](@ref)); for the inverted-file backends -- a text project's and a
sparse project's alike -- every insertion likewise saves just the newly-encoded objects it just
indexed (see [`_invertedfile_on_change`](@ref)). Only an exact dense backend
(`ExhaustiveSearch`/`ParallelExhaustiveSearch`) instead flags a pending whole-index save that
happens right after each `add_item!` call returns (see [`_maybe_flush_index!`](@ref));
[`close_project!`](@ref) forces one final flush of that so nothing recent is lost -- every other
backend never has anything left to flush there, since each block is already durable the instant
it is saved.
"""
function create_project(workdir::String, dataset::String;
                        engine::Type=DenseEngine, backend::Union{Nothing,Type}=nothing,
                        distance=nothing, minrecall::Union{Nothing,Real}=0.9,
                        dimension::Union{Nothing,Integer}=nothing,
                        textmodel::Union{Nothing,IndexEngine.AbstractTextModelSpec}=nothing,
                        index_type=nothing, schema_version::Int=1,
                        postings_cache_max::Int=4096, postings_cache_base::Int=2048)
    index_type === nothing || invalid_option(:index_type, """
        `index_type` is gone: a project now names the kind of data it holds and, separately, the
        index that holds it.
            create_project(w, ds; engine=DenseEngine,    backend=SearchGraph, distance=SqL2())
            create_project(w, ds; engine=SparseEngine,   backend=InvertedFile, dimension=50_000)
            create_project(w, ds; engine=FullTextEngine, backend=BM25InvertedFile, textmodel=...)
        `engine` defaults to DenseEngine and `backend` to that engine's own default.""")

    # Everything checkable is checked before anything exists on disk. `create_engine` repeats
    # some of it, but only after the project directory has been made and RocksDB's write lock
    # taken, so a call that fails there leaves both behind.
    kind = IndexEngine.engine_kind(engine)
    back = backend === nothing ? IndexEngine.default_backend(engine) :
                                 IndexEngine.validate_backend(engine, backend)
    if kind === :text
        IndexEngine.validate_textmodel(back, textmodel)
    elseif textmodel !== nothing
        # Not delegated to `validate_textmodel`, which checks a *backend*: `InvertedFile` is a
        # legal backend for both a sparse and a text project, so the backend alone cannot answer
        # whether a text model belongs here. The engine can, and it is the thing the caller named.
        invalid_option(:textmodel, "`textmodel` only applies to a text project; $(nameof(engine)) indexes $kind " *
              "vectors, which have no text to tokenize and no vocabulary to fit")
    end
    if kind === :sparse
        dimension === nothing &&
            invalid_option(:dimension, "a sparse project needs `dimension`: an InvertedFile is a fixed array of " *
                  "posting lists, so it has to be sized before the first item, and every " *
                  "SparseItem appended has to agree with it")
    elseif dimension !== nothing
        invalid_option(:dimension, "`dimension` only applies to a sparse project; $(nameof(engine)) does not take one")
    end

    dir = joinpath(workdir, dataset)
    mkpath(dir)
    project = Project.open_project(dir, dataset; extra_cf_names=[Persistence.ENGINE_CF, Persistence.ADJACENCY_CF, Persistence.INVFILE_DB_CF,
                                                 Persistence.STAGED_TEXT_CF, Persistence.INVFILE_POSTINGS_CF,
                                                 Persistence.INVFILE_DOCVECS_CF])
    store = Persistence.open_engine_store(project.db)
    pending_flush = Ref(false)
    dense_vectors = Ref{Union{Nothing,MMapMatrixDatabase}}(nothing)

    # Which storage a project persists incrementally into follows from what its backend
    # produces per insertion: a graph reports link blocks, an inverted file reports encoded
    # objects, and an exact index has nothing to report so its whole index is flushed instead.
    on_change = if kind === :dense && back === SearchGraph
        _searchgraph_on_change(store, Persistence.open_adjacency_store(project.db))
    elseif kind === :text
        # A text project persists its *index* (see `Persistence.InvertedIndexStore`), written a
        # block at a time by `IndexEngine.index!` itself, so there is nothing for an observer to
        # save on its behalf and no reason to keep a second copy of every object as well.
        nothing
    elseif kind === :sparse
        _invertedfile_on_change(Persistence.open_invertedfile_object_store(project.db))
    else
        (_, __, ___) -> (pending_flush[] = true)
    end
    adj_factory = kind === :text ?
        _invfile_adj_factory(_invfile_store(project); maxlists=postings_cache_max, baselists=postings_cache_base) :
        nothing

    # The sentinel is resolved here, once, and everything below this line receives a real
    # value -- the whole of the no-defaults-below-the-surface policy in one statement.
    dist = distance === nothing ? IndexEngine.default_distance(back) : distance
    eng = kind === :sparse ?
        IndexEngine.create_sparse_engine(; distance=dist, dimension, on_change, log_io=nothing) :
        IndexEngine.create_engine(back; distance=dist, minrecall, textmodel, on_change, log_io=nothing, adj_factory)
    Persistence.save_fields!(store, IndexEngine.snapshot_state(eng))
    return EmbeddedEngine(workdir, dataset, dir, project, eng, store, pending_flush, schema_version,
                          dense_vectors, false, Ref(true))
end

"""
    open_project(workdir, dataset; read_only=false) -> EmbeddedEngine

Reopens a project previously created by [`create_project`](@ref), restoring its search
engine -- including the `minrecall` target and any calibrated `opt_beamsearch` it was
created/calibrated with -- field-by-field from its `Persistence.EngineStore` (a
`DenseEngine{GraphBackend}`'s index specifically via `IndexEngine.build_searchgraph` replaying its
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
function open_project(workdir::String, dataset::String; read_only::Bool=false, schema_version::Int=1,
                      postings_cache_max::Int=4096, postings_cache_base::Int=2048)
    dir = joinpath(workdir, dataset)
    project = Project.open_project(dir, dataset; read_only, extra_cf_names=[Persistence.ENGINE_CF, Persistence.ADJACENCY_CF, Persistence.INVFILE_DB_CF,
                                                 Persistence.STAGED_TEXT_CF, Persistence.INVFILE_POSTINGS_CF,
                                                 Persistence.INVFILE_DOCVECS_CF])
    store = Persistence.open_engine_store(project.db)
    pending_flush = Ref(false)
    dense_vectors = Ref{Union{Nothing,MMapMatrixDatabase}}(nothing)

    # `kind` and `backend` are symbols, not Julia types (see `IndexEngine.backend_tag`): the
    # on-disk format no longer names the types in `index_engine.jl`, so renaming one is a rename
    # rather than a migration.
    kind = Persistence.load_field(store, :kind, nothing)
    backend = Persistence.load_field(store, :backend, nothing)
    engine = if kind === nothing
        # Never saved: create the same default project `create_project` would have, and save it,
        # mirroring `Server._reload_one_dataset!`'s own create-if-absent fallback.
        on_change = _searchgraph_on_change(store, Persistence.open_adjacency_store(project.db))
        engine = IndexEngine.create_engine(SearchGraph;
            distance=IndexEngine.default_distance(SearchGraph), minrecall=0.9,
            textmodel=nothing, on_change, log_io=nothing)
        Persistence.save_fields!(store, IndexEngine.snapshot_state(engine))
        engine
    elseif kind === :dense && backend === :graph
        dense_vectors[] = Persistence.open_dense_vectors(dir; read_only)
        adj_store = Persistence.open_adjacency_store(project.db)
        state = (
            kind=kind, backend=backend,
            distance=Persistence.load_field(store, :distance, nothing),
            vector_blocks=Persistence.load_dense_vector_blocks(dense_vectors[]),
            load_neighbors=(i -> Persistence.load_neighbors(adj_store, i)),
            graph_len=Persistence.load_field(store, :graph_len, 0),
            minrecall=Persistence.load_field(store, :minrecall, nothing),
            opt_beamsearch=Persistence.load_field(store, :opt_beamsearch, nothing),
            deleted_ids=Persistence.load_field(store, :deleted_ids, nothing),
        )
        IndexEngine.restore_engine(state; on_change=_searchgraph_on_change(store, adj_store), log_io=nothing)
    elseif kind === :dense
        # The exact backends are the one kind whose whole index is a single saved value, so they
        # are also the one kind that still needs `pending_flush` (see `_maybe_flush_index!`).
        on_change = (_, __, ___) -> (pending_flush[] = true)
        state = (
            kind=kind, backend=backend,
            index=Persistence.load_field(store, :index, nothing),
            deleted_ids=Persistence.load_field(store, :deleted_ids, nothing),
        )
        IndexEngine.restore_engine(state; on_change, log_io=nothing)
    elseif kind === :sparse
        obj_store = Persistence.open_invertedfile_object_store(project.db)
        state = (
            kind=kind, backend=backend,
            distance=Persistence.load_field(store, :distance, nothing),
            dimension=Persistence.load_field(store, :dimension, nothing),
            object_blocks=Persistence.load_object_blocks(obj_store),
            deleted_ids=Persistence.load_field(store, :deleted_ids, nothing),
        )
        IndexEngine.restore_engine(state; on_change=_invertedfile_on_change(obj_store), log_io=nothing)
    elseif kind === :text
        # One branch for both text backends: which inverted file to rebuild is `state.backend`'s
        # to say, and `restore_engine` says it. That is the same collapse `FullTextEngine` is.
        # One branch for both text backends: which inverted file to assemble is `state.backend`'s
        # to say, and `restore_engine` says it. That is the same collapse `FullTextEngine` is.
        staged_store = Persistence.open_staged_text_store(project.db)
        profile = Persistence.load_field(store, :profile, nothing)
        invfile_store = _invfile_store(project)
        distance = Persistence.load_field(store, :distance, nothing)
        adj_factory = _invfile_adj_factory(invfile_store; maxlists=postings_cache_max,
                                           baselists=postings_cache_base)
        kindtype = backend === :bm25 ? BM25InvertedFile : TextInvertedFile
        # No index on disk means no index: a text project is assembled from what was persisted or
        # it has none, and `index!` is what builds one out of the staged text. Nothing here
        # reconstructs an index from the raw objects a pre-persistence project saved -- see
        # `index!`'s own docstring for what to do with one of those.
        prebuilt = profile !== nothing && Persistence.has_inverted_index(invfile_store) ?
            _assemble_text_index(profile, kindtype, distance, invfile_store, adj_factory) : nothing
        state = (
            kind=kind, backend=backend,
            profile=profile,
            fitspec=Persistence.load_field(store, :fitspec, nothing),
            distance=distance,
            prebuilt_index=prebuilt,
            adj_factory=adj_factory,
            staged=vcat(String[], Persistence.load_staged_text_blocks(staged_store)...),
            deleted_ids=Persistence.load_field(store, :deleted_ids, nothing),
        )
        IndexEngine.restore_engine(state; on_change=nothing, log_io=nothing)
    else
        corrupted_storage("""
            this project records kind $(repr(kind)), which this version does not know how to \
            restore -- it reads :dense, :sparse and :text. A project written before the engine \
            kinds were restructured stored a Julia type there instead of a symbol, and has to be \
            rebuilt rather than reopened.""")
    end
    return EmbeddedEngine(workdir, dataset, dir, project, engine, store, pending_flush, schema_version,
                          dense_vectors, read_only, Ref(false))
end

"""
    close_project!(handle::EmbeddedEngine; compact::Bool=true)

Flushes the engine's index one final time and closes the project's RocksDB connection
(and, for a `DenseEngine{GraphBackend}` that ever had a vector appended, its `MMapMatrixDatabase`
file too -- see [`_searchgraph_on_change`](@ref)). For a graph-backed dense project and for the
inverted-file backends (sparse and text) there is nothing left to flush -- every block was already
saved the instant it was reported -- so this only does the final `:index` save for
`DenseEngine{<:ExactBackend}`. Every other field (`deleted_ids`, `opt_beamsearch`, ...) is, for every
kind, already persisted immediately by whichever call changed it (`delete_item!`,
`calibrate!`, ...).

Then, unless `compact=false`, it compacts the project's storage through
[`compact_project!`](@ref) -- a no-op for a read-only handle or a session that only read. That
is what keeps the *next* open fast (2.97s versus 0.28s on a 50k-vector project, measured
2026-09-10); pass `compact=false` when closing is on a latency path and the next open is not.
"""
function close_project!(handle::EmbeddedEngine; compact::Bool=true)
    handle.pending_flush[] = false
    if !(handle.engine isa Union{IndexEngine.DenseEngine{IndexEngine.GraphBackend},
                                 IndexEngine.SparseEngine, IndexEngine.FullTextEngine})
        Persistence.save_field!(handle.store, :index, handle.engine.backend.index)
    end
    handle.dense_vectors[] === nothing || close(handle.dense_vectors[])
    compact && compact_project!(handle)
    Project.close_project(handle.project)
    return nothing
end

"""
    compact_project!(handle::EmbeddedEngine; force::Bool=false) -> Bool

Compacts the project's RocksDB storage now, and reports whether it actually ran.

Skipped, and `false` returned, when the handle is read-only (it cannot write) or when this
session has not written anything -- opening a project to run queries and closing it again
leaves nothing to compact, and compaction is not free. `force=true` runs it regardless, for
a caller that knows another process did the writing.

Worth understanding rather than treating as housekeeping: what a bulk write session leaves
behind is a write-ahead log the *next* open has to replay, and that shows up as open
latency, not write latency. Measured 2026-09-10 on a 50k-vector dense project: 2.97s to
reopen without this, 0.28s with it, for 0.03s of compaction. It runs automatically from
[`close_project!`](@ref) (pass `compact=false` there to skip it), so an explicit call is for
compacting mid-session -- after a big ingest that the process will keep serving from, say.
"""
function compact_project!(handle::EmbeddedEngine; force::Bool=false)
    (handle.read_only || !(force || handle.wrote[])) && return false
    Project.compact_all!(handle.project)
    handle.wrote[] = false
    return true
end

"""
    index!(handle::EmbeddedEngine)

Catches up encoding/indexing over whatever's been staged via [`append_items!`](@ref)
since the last call (or since creation) -- one uniform entry point for every project kind:
idempotent, safe to call repeatedly, only ever processes the backlog (see
`IndexEngine.index!(engine::IndexEngine.DenseEngine{IndexEngine.GraphBackend})`'s docstring for
the exact contract, shared verbatim by the text engines). Nothing staged via `append_items!` becomes
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

A no-op for the backends that have no staging split -- a sparse project's `InvertedFile` and a
dense project's `ExhaustiveSearch`/`ParallelExhaustiveSearch` both index on insertion, so there
is never a backlog here to catch up. It is deliberately a no-op and not an error: a caller
writing `create_project` / `append_items!` / `index!` / `search` gets four lines that mean the
same thing for every kind of project, and does not have to know which backends need the third
one. The cost of that is a call that does nothing, which is the cheaper mistake -- the
alternative made switching a project from a graph to a brute-force scan a change to every script
that fed it.
"""
function index!(handle::EmbeddedEngine)
    handle.wrote[] = true
    engine = handle.engine
    if engine isa IndexEngine.DenseEngine{IndexEngine.GraphBackend}
        IndexEngine.index!(engine)
    elseif IndexEngine.payload_kind(engine) === :text
        # `:fitspec`/`:distance` were already written by create_project's full snapshot_state
        # save, and neither ever changes afterwards -- only `:profile` can go from nothing to a
        # fitted model, and only once, so that is the only field this has to write back.
        just_fitted = IndexEngine.text_profile(engine) === nothing
        IndexEngine.index!(engine)
        just_fitted && Persistence.save_field!(handle.store, :profile, IndexEngine.text_profile(engine))
    else
        IndexEngine.index!(engine)   # sparse and exact: already indexed on append, a no-op
    end
    return handle
end

"""
    append_items!(handle::EmbeddedEngine, items) -> Int

Appends a batch of typed items -- [`DenseItem`](@ref)s for a dense project,
[`TextItem`](@ref)s for a text one, each already carrying its own `doc_id`,
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
The backends with no staging split -- the exact dense ones
(`ExhaustiveSearch`/`ParallelExhaustiveSearch`) and a sparse project's `InvertedFile` -- index
synchronously on every `add_item!` instead (see
`IndexEngine.index!(engine::IndexEngine.DenseEngine{GraphBackend})`'s docstring for why
`SimilaritySearch.jl`/`TextSearch.jl` support the split where they do).

Persistence: for a dense project, every batch's raw vectors are appended directly to the
project's `MMapMatrixDatabase` (see [`index!`](@ref index!(::EmbeddedEngine))'s docstring
and the "Performance" note below); for a text project, every batch's raw text is appended
directly to its own `Persistence.StagedTextStore` -- both happen right here, at stage
time, not via `on_change`, since an item is already durable-worthy the moment it's staged,
long before any graph-linking/encoding happens. Once an [`index!`](@ref index!(::EmbeddedEngine))
call actually encodes/indexes a text project's backlog, `SimilaritySearch.CallbackLog` persists
each newly-encoded object incrementally (see [`_invertedfile_on_change`](@ref)).
`DenseEngine{<:ExactBackend}` instead flags a pending whole-index save that happens right after each
`add_item!` call returns (see [`_maybe_flush_index!`](@ref)).

# Performance: batch size matters a lot for a `SearchGraph` (dense) project

Every call here does *one* `SimilaritySearch.append_items!` call into the project's
`MMapMatrixDatabase` for its whole batch of staged vectors, followed by *one*
`SimilaritySearch.flush` of it (durability is opt-in on that type now -- see its docstring's
"Durability" section -- and staying durable right away, as documented above, is this
function's job to arrange, not the database's). So the number of calls to *this* function a
caller makes (not the total vector count) is what drives wall-clock time, exactly as before:
confirmed empirically, 200,000 128-dim vectors written as one project's
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
# Which item a project takes is `payload_kind` spelled the other way round, so it is derived
# rather than restated: a fourth engine kind would otherwise need remembering here too.
const _ITEM_KIND = Dict(Schema.DenseItem => :dense, Schema.SparseItem => :sparse,
                        Schema.TextItem => :text)
const _KIND_ITEM = Dict(v => k for (k, v) in _ITEM_KIND)

_check_item(engine, item::Schema.AbstractItem) =
    _ITEM_KIND[typeof(item)] === IndexEngine.payload_kind(engine) || payload_mismatch(
        "this project indexes $(IndexEngine.payload_kind(engine)), so it takes " *
        "$(nameof(_KIND_ITEM[IndexEngine.payload_kind(engine)]))s; got a $(typeof(item))" *
        (item.doc_id === nothing ? "" : " (doc_id $(item.doc_id))"))

function append_items!(handle::EmbeddedEngine, items::AbstractVector{<:Schema.AbstractItem})
    handle.wrote[] = true
    engine = handle.engine
    project = handle.project
    is_text = IndexEngine.payload_kind(engine) === :text
    is_dense_graph = engine isa IndexEngine.DenseEngine{IndexEngine.GraphBackend}
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
        # `MMapMatrixDatabase` no longer msyncs/fsyncs on its own (see its docstring's
        # "Durability" section) -- making a staged batch durable right away, which is the
        # contract this function documents, is this module's job now.
        SimilaritySearch.flush(handle.dense_vectors[])
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

The count that determines the next item's `_id`, which has to be the *staged* count wherever
staging and indexing are separate: for a `DenseEngine{GraphBackend}`, the number of staged vectors
(`length(database(engine.backend.index))`, i.e. `engine.backend.index.db`'s own count), and for a
`FullTextEngine`, the number of staged texts (`length(engine.staged)`). Since
[`append_items!`](@ref) only stages on those two, `length(engine.backend.index)` (the
encoded/indexed count) would lag behind and hand out `_id`s already taken.

The fallback method is `length(engine.backend.index)`, which is right for every backend that
indexes on insertion -- the exact dense ones and a sparse project's `InvertedFile` -- because
there the index already reflects the item just added.
"""
_current_size(engine::IndexEngine.DenseEngine{IndexEngine.GraphBackend}) = length(SimilaritySearch.database(engine.backend.index))
_current_size(engine::IndexEngine.FullTextEngine) = length(engine.staged)
_current_size(engine::IndexEngine.AbstractSearchEngine) = length(engine.backend.index)

"""
    _search_with_filter(engine, project, query, k, predicate; minrecall, candidates)

Top-`k` under a metadata `predicate`: asks the index for `candidates` hits, keeps the ones
the predicate accepts, stops at `k`.

One pass, not a widening loop. The candidate budget is what the caller pays for and what
the caller therefore names: every candidate costs two RocksDB reads and two decodes (the
record and its `meta`), so a budget that grows itself until `k` is satisfied would let a
selective predicate turn one query into a full scan without anybody asking for it. See
[`search`](@ref) for what to set it to.
"""
function _search_with_filter(engine::IndexEngine.AbstractSearchEngine, project::Project.ProjectManager, query, k::Int, predicate; minrecall, candidates::Int)
    raw = IndexEngine.search_live(engine, query, max(candidates, k); bs_override=nothing, minrecall, policy=nothing)

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
    _resolve_records(project, raw_id) -> Vector{Schema.MetadataRecord}

The records `raw_id` names, whether it is an internal `_id` or a caller-supplied `doc_id`.

Tries the numeric reading first, because it is a direct key lookup, and it can only ever
name one record. A numeric-looking `doc_id` therefore resolves as an `_id` if one exists
with that value -- an ambiguity inherent to accepting both in one argument, and the reason
[`fetch_items`](@ref) and [`exists`](@ref) share this function rather than each inventing its
own precedence.

Everything else is a `doc_id` lookup through `Project.find_all_by_doc_id`, which returns
however many records carry exactly that `doc_id` -- zero, one, or several. A vector, not a
record, because `doc_id` is a caller-chosen external id that nothing here requires to be
unique: appending two items under the same one is allowed, and the scan this replaced
answered such a lookup with whichever of them it reached first.
"""
function _resolve_records(project::Project.ProjectManager, raw_id)
    id_str = string(raw_id)
    maybe_int = tryparse(Int, id_str)
    if maybe_int !== nothing
        record = get_metadata(project, maybe_int)
        record === nothing || return [record]
    end
    find_all_by_doc_id(project, id_str)
end

"""
    _search_query(engine, vector) -> Vector{Float32} | SparseVector{Float32,Int32}

`vector` in the form this project's backend searches with.

A dense project's index evaluates a distance against a contiguous vector; a sparse project's
`InvertedFile` walks the query's nonzero positions to pick posting lists, and reads its
`nzind`/`nzval` directly. So the conversion is the project's, not the caller's -- and the sparse
case cannot fall back on the dense one: `convert(Vector{Float32}, sparse_query)` type-checks,
densifies, and then dies inside `select_posting_lists` on a `Float32` used as an index.

A dense vector handed to a sparse project is rejected rather than sparsified. It would usually
be a mistake worth naming (a project's items are `SparseItem`s, so its queries are sparse too),
and sparsifying silently would turn a dimension mismatch into posting lists selected from
whatever the values happened to be.
"""
function _search_query(engine::IndexEngine.AbstractSearchEngine, vector)
    kind = IndexEngine.payload_kind(engine)
    kind === :dense && return convert(Vector{Float32}, vector)
    kind === :sparse || payload_mismatch("search(handle, vector) needs a dense or sparse project; this one holds $kind")
    vector isa AbstractSparseVector ||
        payload_mismatch("a sparse project searches with a SparseVector{Float32,Int32}, not a $(typeof(vector)); " *
              "build one with `sparsevec(indices, values, dimension)` over the same dimension the " *
              "project was created with ($(engine.backend.dimension))")
    length(vector) == engine.backend.dimension ||
        wrong_dimension(engine.backend.dimension, length(vector),
              "query dimension $(length(vector)) does not match the project's $(engine.backend.dimension)")
    convert(SparseVector{Float32,Int32}, vector)
end

"""
    search(handle::EmbeddedEngine, vector, k::Int=10; filter=nothing) -> Vector{SearchResult}

Dense vector search, hydrated with each hit's original id (mirrors `Server.handle_search`
minus the HTTP/telemetry/pagination machinery). `filter`, if given, is a
`(record::Schema.MetadataRecord, meta) -> Bool` predicate function called on each
candidate's record and (lazily decoded, see `Schema.decode_meta`) `meta` -- e.g.
`(record, meta) -> meta !== nothing && meta["year"] >= 2020`. A caller here already has real
Julia functions to work with, not a JSON wire format to encode a filter into, so `filter`
is just that function, not a name-keyed spec for some interpreter to replay. `meta` is
fetched (one extra `Project.get_meta` read per candidate) even if `record` alone would've
been enough for a given predicate -- the common case (filtering on a `meta` field) needs it,
and there's no way to know a predicate won't from here.

`candidates` is how many hits the index is asked for before filtering, and it matters
whenever `filter` is given: **the search returns fewer than `k` results when fewer than `k`
of those candidates satisfy the predicate**, quietly, because the alternative -- growing the
budget until `k` is met -- turns a selective predicate into a full scan of the collection at
two RocksDB reads per item. The default of `2k` suits a predicate that most items pass (a
language, a type, a recent date). Match it to how selective yours is: a predicate that
roughly one item in `n` satisfies needs about `n * k` candidates to fill a page of `k`, so
`search(h, q, 10; filter=f, candidates=8192)` is the shape of a query filtering on something
rare, and the cost of that is ~8k candidate reads. Without `filter`, `candidates` is unused.

Without `filter`, this is a plain top-`k` search: soft-deleted candidates are not hidden
or backfilled -- they're returned with `deleted=true` and no hydrated metadata (see
[`IndexEngine.search_live`](@ref)), so `doc_id` is `nothing` for them.
Getting `k` *live* results back is a paging concern for a layer above this one (e.g. a
server walking successive windows via a cursor), not something this function does itself.

`minrecall`, for a `DenseEngine{GraphBackend}`, searches at (approximately) that target recall
using a calibrated `BeamSearch` from `engine.backend.opt_beamsearch` instead of its current
default (see [`IndexEngine.search_live`](@ref)) -- if that table is still empty, this
triggers a one-off `calibrate!` over `IndexEngine.DEFAULT_MINRECALL_LEVELS` and persists
the resulting `opt_beamsearch` so that calibration isn't silently repeated on every future
search. Ignored for any other engine kind.

Returns a `Vector{`[`SearchResult`](@ref)`}` -- `_id` is the internal id, `doc_id` the
caller's own, and note that those two field names sit the opposite way round from the named
tuple this replaced.
"""
function search(handle::EmbeddedEngine, vector, k::Int=10; filter=nothing, minrecall=nothing, candidates::Int=2k)
    query = _search_query(handle.engine, vector)
    needs_save = minrecall !== nothing && handle.engine isa IndexEngine.DenseEngine{IndexEngine.GraphBackend} && isempty(handle.engine.backend.opt_beamsearch)
    res_knn = filter === nothing ?
        IndexEngine.search_live(handle.engine, query, k; bs_override=nothing, minrecall, policy=nothing) :
        _search_with_filter(handle.engine, handle.project, query, k, filter; minrecall, candidates)
    needs_save && Persistence.save_field!(handle.store, :opt_beamsearch, handle.engine.backend.opt_beamsearch)
    return _hydrate_results(handle.project, res_knn)
end

"""
    _batch_query_database(engine, queries) -> SimilaritySearch.AbstractDatabase

`queries` in the form this project's index searches a whole batch with: a `MatrixDatabase`
of `Float32` columns for a dense project, a `VectorDatabase` of `SparseVector{Float32,Int32}`
for a sparse one.

Each query goes through [`_search_query`](@ref) first, so a batch rejects exactly what a
single query rejects (wrong payload kind, wrong dimension) and says the same thing when it
does -- the batch entry point is not a second, laxer door into the same index.
"""
function _batch_query_database(engine::IndexEngine.AbstractSearchEngine, queries)
    isempty(queries) && invalid_option(:queries, "searchbatch needs at least one query")
    kind = IndexEngine.payload_kind(engine)
    kind === :text &&
        unsupported_operation(:searchbatch, "searchbatch is for dense and sparse projects; a text project answers strings, " *
              "so batch it with `[ftsearch(h, q, k) for q in queries]` (one query at a time)")
    prepared = [_search_query(engine, q) for q in queries]
    if kind === :dense
        dim = length(first(prepared))
        all(q -> length(q) == dim, prepared) ||
            wrong_dimension(dim, length(first(prepared)),
                  "every query in a batch must have the project's dimension ($dim)")
        M = Matrix{Float32}(undef, dim, length(prepared))
        for (j, q) in enumerate(prepared)
            M[:, j] = q
        end
        return MatrixDatabase(M)
    end
    return VectorDatabase(prepared)
end

"""
    searchbatch(handle::EmbeddedEngine, queries, k::Int=10) -> (ids, dists)

`k` nearest neighbours for every query at once: two `(k, length(queries))` matrices, of
`UInt32` internal ids and `Float32` distances, exactly as the library returns them. A column
with fewer than `k` hits is padded with `0` ids and `typemax(Float32)` distances.

This is the raw, unhydrated form -- no `doc_id`, no soft-delete flags, no RocksDB reads at
all -- and it exists because that is what the operations built on a batch of queries
(evaluation loops, offline scoring, feeding another index) actually want.
`search(handle, queries, k)` is the hydrated form.

**A whole-dataset operation, in the same family as [`allknn`](@ref)/[`fft`](@ref)**, not a
concurrent search: it takes the project's write lock (full exclusivity, including from other
batch calls and from `index!`) and refuses to run while items are staged but not yet indexed.
See `IndexEngine.searchbatch_live` for why that is not a choice.

Worth it whenever there are many queries: measured 2026-09-10 over 50k dense vectors on 8
threads, 39k queries/s here against 3.6k/s calling [`search`](@ref) one query at a time.

A text project has no batch entry point here -- its queries are strings the library resolves
one at a time -- so call [`ftsearch`](@ref) in a loop.
"""
function searchbatch(handle::EmbeddedEngine, queries, k::Int=10)
    Q = _batch_query_database(handle.engine, queries)
    IndexEngine.searchbatch_live(handle.engine, Q, k)
end

"""
    search(handle::EmbeddedEngine, queries::AbstractVector{<:AbstractVector{<:Real}}, k::Int=10) -> Vector{Vector{SearchResult}}

Hydrated batch search: one inner vector of [`SearchResult`](@ref) per query, in the order
`queries` gave, each ordered by ascending distance.

The same results [`search`](@ref) returns for one query, computed for all of them in one
parallel pass ([`searchbatch`](@ref)) and hydrated the same way -- one `doc_id` lookup and one
soft-delete check per hit. It carries `searchbatch`'s constraints with it (write lock, no
staged backlog, dense or sparse only), because it is that call plus hydration.

No `filter` keyword here: filtering is a per-query candidate budget (see [`search`](@ref)'s
`candidates`), and mixing that with a batch whose whole point is one uniform `k` for every
query would make the cost of a call impossible to read off it. Ask for a larger `k` and apply
the predicate yourself, or loop over the single-query [`search`](@ref) when the predicate is
what matters.
"""
function search(handle::EmbeddedEngine, queries::AbstractVector{<:AbstractVector{<:Real}}, k::Int=10)
    ids, dists = searchbatch(handle, queries, k)
    engine = handle.engine
    out = Vector{Vector{SearchResult}}(undef, size(ids, 2))
    for j in axes(ids, 2)
        hits = SearchResult[]
        for i in axes(ids, 1)
            id = ids[i, j]
            id == 0 && continue
            _id = Int32(id)
            if _id in engine.deleted_ids
                push!(hits, SearchResult(_id, nothing, Float32(dists[i, j]), true))
            else
                record = get_metadata(handle.project, _id)
                push!(hits, SearchResult(_id, record === nothing ? nothing : record.doc_id,
                                         Float32(dists[i, j]), false))
            end
        end
        out[j] = hits
    end
    return out
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
    res_knn = IndexEngine.search_live(handle.engine, text, k; bs_override=nothing, minrecall, policy)
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
    IndexEngine.payload_kind(engine) === :text ||
        unsupported_operation(:ftexplain, "ftexplain is only meaningful for a text project (BM25InvertedFile/InvertedFile); this one is $(typeof(engine))")
    return TextSearch.explain(IndexEngine.resolve_query(engine, text, policy).resolution)
end

"""
    delete_item!(handle::EmbeddedEngine, _id::Integer)

Soft-deletes `_id` (future searches exclude it, the underlying index is untouched) and
immediately persists just the `deleted_ids` field -- mirrors `Server.handle_delete_item`.
"""
function delete_item!(handle::EmbeddedEngine, _id::Integer)
    handle.wrote[] = true
    IndexEngine.mark_deleted!(handle.engine, _id)
    Persistence.save_field!(handle.store, :deleted_ids, handle.engine.deleted_ids)
    return nothing
end

"""
    fetch_items(handle::EmbeddedEngine, ids) -> Vector{Schema.StoredItem}

Batch retrieval by id -- each element of `ids` may be the internal `_id` or a `doc_id` (see
[`_resolve_records`](@ref)). An id that resolves to nothing is skipped and a `doc_id` shared
by several items brings back all of them, so the result can be shorter *or longer* than
`ids`: it is a flat vector of whatever matched, in the order `ids` gave, ties broken by
ascending `_id`.

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
        for record in _resolve_records(project, raw_id)
            meta = something(get_meta(project, record._id; lazy=false), Dict{String,Any}())
            push!(results, Schema.StoredItem(record,
                                             IndexEngine.stored_payload(handle.engine, record._id),
                                             meta))
        end
    end
    return results
end

"""
    exists(handle::EmbeddedEngine, ids) -> Vector{ExistsResult}

For each id in `ids` (internal `_id` or a `doc_id`, see [`_resolve_records`](@ref)), reports
whether a record exists and, if so, whether it's been soft-deleted. One result per *queried
id*, in order, including the ones that were not found -- unlike [`fetch_items`](@ref), which
drops those and returns one entry per matching record.

Since a `doc_id` may be carried by several items, the two flags summarize them: `exists` is
true when at least one record matched, and `deleted` is true only when *every* match is
soft-deleted, i.e. when asking for that id leaves nothing live to retrieve.
"""
function exists(handle::EmbeddedEngine, ids)
    project = handle.project
    engine = handle.engine
    results = ExistsResult[]
    for raw_id in ids
        records = _resolve_records(project, raw_id)
        found = !isempty(records)
        deleted = found && all(r -> r._id in engine.deleted_ids, records)
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
    handle.wrote[] = true
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
    IndexEngine.payload_kind(engine) === :dense ||
        unsupported_operation(:allknn, "allknn requires a dense (vector) project; this one indexes $(IndexEngine.payload_kind(engine))")
    length(engine.backend.index) == 0 && empty_project(:allknn, "allknn requires a non-empty dense index")

    ids, dists = IndexEngine.allknn_live(engine, k)
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
docstring and `SimilaritySearch.CenterSelection`), with ids as `Int32` to match every other id
this package hands back. `centers` are internal `_id`s (1-based position in the index), not
hydrated with metadata -- look them up yourself (e.g. via [`fetch_items`](@ref)) if you need
it. `assign[i]` is a position in `centers`, so item `i`'s center is `centers[assign[i]]`.
"""
function fft(handle::EmbeddedEngine, k::Integer; start::Int=0, verbose::Bool=false)
    engine = handle.engine
    IndexEngine.payload_kind(engine) === :dense ||
        unsupported_operation(:fft, "fft requires a dense (vector) project; this one indexes $(IndexEngine.payload_kind(engine))")
    length(engine.backend.index) == 0 && empty_project(:fft, "fft requires a non-empty dense index")
    r = IndexEngine.fft_live(engine, k, start, verbose)
    CenterSelectionResult(Int32.(r.centers), Int32.(r.assign), Float32.(r.assigndist),
                          Float32(r.covering), Float32(r.separation),
                          Int(r.costdists), Int(r.costblocks))
end

"""
    dnet(handle::EmbeddedEngine, k::Integer; verbose::Bool=false) -> CenterSelectionResult

Runs `SimilaritySearch.dnet` (density-based ball partitioning) synchronously against the
project's dense index -- selects representative centers by carving the database into balls of
approximately `length(db) ÷ k` elements. Errors if the project is not a dense index or is empty.

Returns a [`CenterSelectionResult`](@ref).
"""
function dnet(handle::EmbeddedEngine, k::Integer; verbose::Bool=false)
    engine = handle.engine
    IndexEngine.payload_kind(engine) === :dense ||
        unsupported_operation(:dnet, "dnet requires a dense (vector) project; this one indexes $(IndexEngine.payload_kind(engine))")
    length(engine.backend.index) == 0 && empty_project(:dnet, "dnet requires a non-empty dense index")
    r = IndexEngine.dnet_live(engine, k, verbose)
    CenterSelectionResult(Int32.(r.centers), Int32.(r.assign), Float32.(r.assigndist),
                          Float32(r.covering), Float32(r.separation),
                          Int(r.costdists), Int(r.costblocks))
end

"""
    neardup(handle::EmbeddedEngine, epsilon::Real; verbose::Bool=false, recall::Real=1.0) -> NearDupResult

Runs `SimilaritySearch.neardup` synchronously against the project's dense index --
finds the ``ϵ``-net of non-duplicate objects whose mutual distances are at least `epsilon`.
Objects closer than `epsilon` to an existing center are considered near-duplicates and assigned to it.
Errors if the project is not a dense index or is empty.

Returns a [`NearDupResult`](@ref).
"""
function neardup(handle::EmbeddedEngine, epsilon::Real; verbose::Bool=false, recall::Real=1.0)
    engine = handle.engine
    IndexEngine.payload_kind(engine) === :dense ||
        unsupported_operation(:neardup, "neardup requires a dense (vector) project; this one indexes $(IndexEngine.payload_kind(engine))")
    length(engine.backend.index) == 0 && empty_project(:neardup, "neardup requires a non-empty dense index")
    r = IndexEngine.neardup_live(engine, Float32(epsilon), verbose, Float32(recall))
    NearDupResult(Int32.(r.centers), Int32.(r.assign), Float32.(r.assigndist),
                  Float32(r.covering), Float32(r.epsilon),
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
    IndexEngine.payload_kind(engine) === :dense ||
        unsupported_operation(:closestpairs, "closestpairs requires a dense (vector) project; this one indexes $(IndexEngine.payload_kind(engine))")
    length(engine.backend.index) == 0 && empty_project(:closestpairs, "closestpairs requires a non-empty dense index")
    IndexEngine.closestpairs_live(engine, k, min_k)
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
    IndexEngine.payload_kind(engine) === :dense ||
        unsupported_operation(:bichromatic_kclosestpairs, "bichromatic_kclosestpairs requires a dense (vector) project; this one indexes $(IndexEngine.payload_kind(engine))")
    length(engine.backend.index) == 0 && empty_project(:bichromatic_kclosestpairs, "bichromatic_kclosestpairs requires a non-empty dense index")
    IndexEngine.bichromatic_kclosestpairs_live(engine, B, k, min_k)
end
