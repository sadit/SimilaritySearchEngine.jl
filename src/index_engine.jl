module IndexEngine

using SimilaritySearch
# Not re-exported at SimilaritySearch's top level (only ExhaustiveSearch is) -- it lives in
# the Exact submodule, brought into SimilaritySearch's own namespace via `using .Exact`.
import SimilaritySearch: ParallelExhaustiveSearch
using TextSearch
using Base.Threads

export SearchEngineWrapper, create_engine, restore_engine, add_item!, ensure_trained!, search_live, mark_deleted!, is_text_index
export calibrate!, current_beamsearch, DEFAULT_BEAMSEARCH

# Distances we support mapped by string
const DISTANCE_MAP = Dict{String, Any}(
    "L2" => SimilaritySearch.Dist.SqL2(), # often optimized internally to SqL2
    "Cosine" => SimilaritySearch.Dist.Cosine(),
    "Angle" => SimilaritySearch.Dist.Angle(),
    "NormalizedCosine" => SimilaritySearch.Dist.NormCosine()
)

function get_distance(name::String)
    return get(DISTANCE_MAP, name, SimilaritySearch.Dist.SqL2())
end

"""
    TextKind

Tags which (if any) TextSearch.jl inverted-file family a `SearchEngineWrapper` is
backing. `BM25Kind` and `WeightedKind` indices can't be constructed until a
`Vocabulary` has been trained from a real corpus (see [`ensure_trained!`](@ref)),
so `NoText` engines (dense `SearchGraph`/`ExhaustiveSearch`/`ParallelExhaustiveSearch`)
are built eagerly while text engines start with `index === nothing`.
"""
@enum TextKind NoText BM25Kind WeightedKind

"""
    SearchEngineWrapper{IndexType}

A wrapper for SimilaritySearch and TextSearch indices
that handles basic concurrency and logical deletions (soft deletes).

# Fields
- `index::IndexType`: The underlying search index. `nothing` for a text engine
  whose vocabulary hasn't been trained yet.
- `text_kind::TextKind`: `NoText` for dense indices, otherwise which text-index family to build once trained.
- `voc::Union{Nothing, TextSearch.Vocabulary}`: The trained vocabulary for text engines (`nothing` until trained).
- `model::Union{Nothing, TextSearch.VectorModel}`: for `WeightedKind` engines only, the
  trained tf-idf `VectorModel` used to turn text into the `SparseVector`s a
  `NormCosine`-distance `InvertedFile` needs (`nothing` for `NoText`/`BM25Kind` engines,
  which vectorize via `bagofwords`/a raw `BOW` instead).
- `ctx::Any`: A cached, reusable search/insertion context (avoids allocating one per request).
- `deleted_ids::Set{Int}`: A set of logically deleted document IDs.
- `lock::ReentrantLock`: A lock for concurrent access.
"""
mutable struct SearchEngineWrapper{IndexType}
    index::IndexType
    text_kind::TextKind
    voc::Union{Nothing, TextSearch.Vocabulary}
    model::Union{Nothing, TextSearch.VectorModel}
    ctx::Any
    deleted_ids::Set{Int}
    lock::ReentrantLock
end

"""
    DEFAULT_BEAMSEARCH

`SimilaritySearch.jl`'s own `BeamSearch()` defaults (`bsize=4, Δ=1.0, maxvisits=10^6`),
used as the bootstrap "baseline" for the PLAN.md §3 safety-multiplier barrier on a
`SearchGraph` that has never been explicitly calibrated (`calibrate!`) yet — a real
per-library-default baseline rather than the arbitrary global constants (`bsize<=20`,
`Δ<1.7`) PLAN.md's own §3 warning flags as unjustified.
"""
const DEFAULT_BEAMSEARCH = BeamSearch()

# Context helpers
#
# `hyperparameters_callback=nothing` disables `SearchGraphContext`'s own default
# (`OptimizeParameters()`, i.e. `MinRecall(0.9)`), which would otherwise silently
# autotune `BeamSearch` during index construction/growth -- PLAN.md §5.6 explicitly warns
# against letting that mechanism and an explicit `calibrate!` call both write to the same
# stored baseline independently. Resolution: `calibrate!` (below) is the *sole* source of
# truth for a `SearchGraph`'s `BeamSearch` config in this pass; construction never
# autotunes on its own.
get_context_for_index(index::SearchGraph) = SearchGraphContext(; hyperparameters_callback=nothing)
get_context_for_index(index::Union{ExhaustiveSearch,ParallelExhaustiveSearch}) = GenericContext()
get_context_for_index(index) = GenericContext()

"""
    create_engine(index::IndexType) -> SearchEngineWrapper{IndexType}

Wraps an already-built dense index (used for CLI `searchbatch`, which loads a JLD2 snapshot).
"""
function create_engine(index::IndexType) where {IndexType}
    return SearchEngineWrapper{IndexType}(index, NoText, nothing, nothing, get_context_for_index(index), Set{Int}(), ReentrantLock())
end

"""
    create_engine(index_type::String, distance_name::String) -> SearchEngineWrapper

Creates a new, empty search engine of the specified type. Dense index kinds are built
immediately; text index kinds (`invfile`, `bm25_invfile`/`bm25`) are left untrained
(`index === nothing`) until [`ensure_trained!`](@ref) is called with a real corpus,
since `BM25InvertedFile`/`WeightedInvertedFile` need a `Vocabulary` to be constructed at all.
"""
function create_engine(index_type::String, distance_name::String)
    dist = get_distance(distance_name)

    if index_type == "searchgraph"
        return create_engine(SearchGraph(dist, VectorDatabase()))
    elseif index_type == "exhaustive_search"
        return create_engine(ExhaustiveSearch(dist, VectorDatabase()))
    elseif index_type == "parallel_exhaustive_search"
        return create_engine(ParallelExhaustiveSearch(dist, VectorDatabase()))
    elseif index_type == "bm25_invfile" || index_type == "bm25"
        return SearchEngineWrapper{Any}(nothing, BM25Kind, nothing, nothing, nothing, Set{Int}(), ReentrantLock())
    elseif index_type == "invfile" || index_type == "weighted_invfile"
        return SearchEngineWrapper{Any}(nothing, WeightedKind, nothing, nothing, nothing, Set{Int}(), ReentrantLock())
    else
        # Default fallback
        return create_engine(SearchGraph(dist, VectorDatabase()))
    end
end

"""
    restore_engine(index, text_kind::TextKind, voc, model, deleted_ids::Set{Int}) -> SearchEngineWrapper

Rebuilds a `SearchEngineWrapper` from a persisted `(index, text_kind, voc, model, deleted_ids)`
tuple (see `Persistence.save_snapshot`/`load_snapshot`) — the counterpart of
`create_engine` used when loading a JLD2 snapshot back, since a bare `index` alone isn't
enough to know whether it's a text engine, to recompute bags-of-words/vectors for future
queries, or which of its documents are soft-deleted (tombstones live only in the snapshot;
nothing about them is stored inside `index`/`voc`/`model` themselves).

`model` is the trained `TextSearch.VectorModel` used by `WeightedKind` engines to vectorize
text into the `SparseVector`s a `NormCosine`-distance `InvertedFile` needs (`nothing` for
`NoText`/`BM25Kind` engines, which never use it).
"""
function restore_engine(index, text_kind::TextKind, voc, model, deleted_ids::Set{Int})
    if text_kind === NoText
        engine = create_engine(index)
        engine.deleted_ids = deleted_ids
        return engine
    end
    ctx = index === nothing ? nothing : InvertedFileContext()
    return SearchEngineWrapper{Any}(index, text_kind, voc, model, ctx, deleted_ids, ReentrantLock())
end

is_text_index(engine::SearchEngineWrapper) = engine.text_kind !== NoText

"""
    ensure_trained!(engine::SearchEngineWrapper, corpus::AbstractVector{<:AbstractString})

Trains the engine's `Vocabulary` (and builds the real `BM25InvertedFile`/`WeightedInvertedFile`)
from `corpus` if this is a text engine that hasn't been trained yet. No-op for dense engines,
for already-trained text engines, and for an empty `corpus`.

Must be called once with the *first* batch of text a text-index dataset receives (e.g. the
first `append`/`build` batch) — `TextSearch.jl`'s `Vocabulary` is trained once from a corpus;
tokens not seen at training time are treated as out-of-vocabulary and silently dropped on
every later append (a library limitation, not a bug — see `PLAN.md` §1's note on vocabulary drift).
"""
function ensure_trained!(engine::SearchEngineWrapper, corpus::AbstractVector)
    engine.text_kind === NoText && return
    engine.voc !== nothing && return
    isempty(corpus) && return

    lock(engine.lock) do
        engine.voc !== nothing && return
        text_corpus = String.(corpus)
        voc = Vocabulary(TextConfig(), text_corpus)
        engine.voc = voc
        if engine.text_kind === BM25Kind
            engine.index = BM25InvertedFile(voc)
        else
            engine.index = WeightedInvertedFile(max(vocsize(voc), 1))
            engine.model = VectorModel(IdfWeighting(), TfWeighting(), voc)
        end
        engine.ctx = InvertedFileContext()
    end
end

"""
    current_beamsearch(engine::SearchEngineWrapper) -> Union{Nothing, BeamSearch}

The `BeamSearch` configuration currently installed on `engine.index`, if it's a
`SearchGraph` (`nothing` for anything else — exact indices have no beam to configure,
text indices aren't dense at all). This is `DEFAULT_BEAMSEARCH` until `calibrate!` has
been called at least once, since construction no longer autotunes on its own.
"""
current_beamsearch(engine::SearchEngineWrapper) = engine.index isa SearchGraph ? engine.index.algo[] : nothing

"""
    calibrate!(engine::SearchEngineWrapper; minrecall=0.9, numqueries=64, ksearch=10, queries=nothing) -> BeamSearch

Runs `SimilaritySearch.jl`'s real `optimize_index!` hyperparameter sweep (PLAN.md §5.6 —
a `SearchModels`-driven stochastic search over `BeamSearchSpace`, not a hand-rolled one)
against `engine.index`, installing the best-found `BeamSearch` as `engine.index.algo[]`
and returning it. Only meaningful for a `SearchGraph`; errors otherwise.

# Keyword Arguments
- `minrecall`: target recall (0-1) the calibration optimizes for — see `MinRecall`.
- `numqueries`: size of the query sample drawn from the already-indexed database when
  `queries` isn't given.
- `ksearch`: neighbors retrieved per query while evaluating candidate configurations.
- `queries::Union{Nothing, AbstractVector{<:AbstractVector{Float32}}}`: an explicit
  ground-truth query set (PLAN.md §5.6's "supplied ... set") instead of an internally
  sampled one.
"""
function calibrate!(engine::SearchEngineWrapper; minrecall::Real=0.9, numqueries::Int=64, ksearch::Int=10, queries=nothing)
    engine.index isa SearchGraph || error("calibrate! only applies to a searchgraph (approximate dense) index")

    Q = queries === nothing ? nothing : VectorDatabase([convert(Vector{Float32}, q) for q in queries])

    lock(engine.lock) do
        optimize_index!(engine.index, engine.ctx, MinRecall(Float32(minrecall)); queries=Q, numqueries, ksearch)
    end

    return engine.index.algo[]
end

"""
    mark_deleted!(engine::SearchEngineWrapper, doc_id::Int)

Marks a document as logically deleted.
"""
function mark_deleted!(engine::SearchEngineWrapper, doc_id::Int)
    lock(engine.lock) do
        push!(engine.deleted_ids, doc_id)
    end
end

# Insertion helpers for dense indices
insert_dense!(index::SearchGraph, ctx, item) = append_items!(index, ctx, VectorDatabase([item]))
insert_dense!(index::Union{ExhaustiveSearch,ParallelExhaustiveSearch}, ctx, item) = push_item!(index, ctx, item)
insert_dense!(index, ctx, item) = push_item!(index, ctx, item)

"""
    add_item!(engine::SearchEngineWrapper, item)

Adds a single item to the index. Thread-safe wrapper.
For text engines, `item` is raw text and `ensure_trained!` must already have been called
(with the batch `item` belongs to) or the item is silently dropped (untrained engine has no index yet).

# Arguments
- `engine::SearchEngineWrapper`: The search engine wrapper.
- `item`: The item (vector, text, etc.) to add to the index.
"""
function add_item!(engine::SearchEngineWrapper, item)
    lock(engine.lock) do
        if engine.text_kind === NoText
            insert_dense!(engine.index, engine.ctx, item)
        else
            engine.voc === nothing && return # untrained text engine: nothing to index into yet
            # BM25Kind wants raw term-frequency pairs (a BOW); WeightedKind's `InvertedFile`
            # is NormCosine-distanced and needs a real weighted SparseVector -- TextSearch.jl
            # dropped Dict-based dot/norm/evaluate, so a raw BOW no longer scores correctly
            # there (see PLAN.md's note on this fix).
            obj = engine.text_kind === BM25Kind ? bagofwords(engine.voc, item) : vectorize(engine.model, item)
            push_item!(engine.index, engine.ctx, obj)
        end
    end
end

"""
    search_live(engine::SearchEngineWrapper, query, k::Int; bs_override=nothing) -> (id=..., dist=...)

Performs a search, post-filtering deleted items.
If there are many deleted items in the top-k, it might return fewer than `k` results.

# Arguments
- `engine::SearchEngineWrapper`: The search engine wrapper.
- `query`: The search query (vector, text, etc.).
- `k::Int`: The number of neighbors to retrieve.

# Keyword Arguments
- `bs_override::Union{Nothing, BeamSearch}`: an ad hoc `BeamSearch` configuration to
  search with for this call only (PLAN.md §3/§7's per-request `beamsearch_overrides`),
  instead of `engine.index.algo[]`'s calibrated default. Only honored for a `SearchGraph`
  — ignored (falls back to the normal path) for anything else, since no other index kind
  has a `BeamSearch` to override in the first place.
"""
function search_live(engine::SearchEngineWrapper, query, k::Int; bs_override=nothing)
    # Heuristic: request slightly more than k if there are deleted items
    k_request = max(k + length(engine.deleted_ids), 1)

    if engine.text_kind === NoText
        ctx = engine.ctx
        if bs_override !== nothing && engine.index isa SearchGraph
            res = knnqueue(KnnSorted, k_request)
            vstate = SimilaritySearch.getvstate(length(engine.index), ctx)
            search(bs_override, engine.index, ctx, query, res, engine.index.hints, vstate)
            return _collect_live(engine, IdView(res), DistView(res), k)
        end
        Q = VectorDatabase([query])
        raw_ids, raw_dists = SimilaritySearch.searchbatch(engine.index, ctx, Q, k_request; sorted=true)
        return _collect_live(engine, view(raw_ids, :, 1), view(raw_dists, :, 1), k)
    else
        engine.voc === nothing && return (id=Int32[], dist=Float32[])
        obj = engine.text_kind === BM25Kind ? bagofwords(engine.voc, query) : vectorize(engine.model, query)
        res = knnqueue(KnnSorted, k_request)
        search(engine.index, engine.ctx, obj, res)
        return _collect_live(engine, IdView(res), DistView(res), k)
    end
end

function _collect_live(engine::SearchEngineWrapper, cand_ids, cand_dists, k::Int)
    final_ids = Int32[]
    final_dists = Float32[]

    for (cand_id, cand_dist) in zip(cand_ids, cand_dists)
        cand_id == 0 && continue # 0 means empty slot
        if !in(cand_id, engine.deleted_ids)
            push!(final_ids, cand_id)
            push!(final_dists, cand_dist)
            if length(final_ids) == k
                break
            end
        end
    end

    return (id=final_ids, dist=final_dists)
end

end # module
