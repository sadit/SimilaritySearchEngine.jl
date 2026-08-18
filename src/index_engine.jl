module IndexEngine

using SimilaritySearch
# Not re-exported at SimilaritySearch's top level (only ExhaustiveSearch is) -- it lives in
# the Exact submodule, brought into SimilaritySearch's own namespace via `using .Exact`.
import SimilaritySearch: ParallelExhaustiveSearch
using TextSearch
using Base.Threads
using SparseArrays: SparseVector
using Dates: Dates

export AbstractSearchEngine, SearchGraphEngine, GenericEngine, BM25Engine, InvertedFileEngine
export ReadWriteLock, read_lock, write_lock
export ContextPool, checkout!, checkin!
export create_engine, restore_engine, snapshot_state, extra_state_fields, add_item!, ensure_trained!, search_live, mark_deleted!, is_text_index
export calibrate!, current_beamsearch, OptBeamSearch, DEFAULT_MINRECALL_LEVELS, CallbackLog, FileLog
export searchgraph_vectors, direct_neighbors, apply_searchgraph_vectors!, build_searchgraph
export invertedfile_objects, build_bm25invertedfile, build_invertedfile

"""
    AbstractSearchEngine

Common supertype for every concrete engine kind ([`SearchGraphEngine`](@ref),
[`GenericEngine`](@ref), [`BM25Engine`](@ref), [`InvertedFileEngine`](@ref)). Each kind
carries only the state it actually needs -- e.g. `minrecall` only exists on
`SearchGraphEngine`, `voc`/`model` only on the text engines -- and the concrete Julia type
itself is what tells `add_item!`/`search_live`/etc. and `restore_engine` which behavior to
run, instead of a separate enum tag every method would have to branch on.

Every concrete engine shares three fields: `deleted_ids::Set{UInt32}` (logically deleted
document ids), `lock::ReadWriteLock` (guards concurrent access -- see its docstring for
why this needs to be a reader/writer lock, not a plain mutex), and `search_ctx_pool`
(each concurrent [`search_live`](@ref) call's own private context -- see
[`ContextPool`](@ref) for why that's separate from `ctx`, which insertion keeps to itself).
"""
abstract type AbstractSearchEngine end

"""
    ReadWriteLock

Any number of readers may hold this concurrently, but a writer needs it exclusively (no
readers, no other writer) -- unlike a plain `ReentrantLock`, which only ever allows one
holder at a time regardless of what it's used for.

This distinction matters here specifically because every underlying index kind
(`SearchGraph`, `ExhaustiveSearch`/`ParallelExhaustiveSearch`, `BM25InvertedFile`,
`InvertedFile`) safely supports running multiple searches concurrently against the same
index -- that's exactly what their own `searchbatch`/`allknn`/`@BATCHES`-driven parallel
search paths already rely on internally -- but *none* of them tolerates a search running
concurrently with an insertion or deletion into the same index (a search reading
`index.adj`/`index.db` while `add_item!` resizes/appends to those same arrays is exactly
the kind of concurrent read-during-mutation none of them are built for). So
[`search_live`](@ref)/[`current_beamsearch`](@ref) take a *read* lock (concurrent with
each other, exclusive of any write), while `add_item!`/`ensure_trained!`/
`mark_deleted!`/`calibrate!` take a *write* lock (exclusive of everything).

Implemented with a `Threads.Condition` guarding a plain reader count: [`read_lock`](@ref)
briefly locks it only to bump/drop that count (so the actual read work runs *without*
holding the condition's own lock, letting readers overlap), while [`write_lock`](@ref)
holds the condition's lock for its whole body -- once acquired, no new reader can even
start, and the writer waits for any in-flight readers to finish first.

!!! warning "Never take a write lock from inside a read-locked section"
    A read-to-write "upgrade" on the *same* logical operation deadlocks: the write lock
    would wait for the active reader count to drop to zero, including the one this same
    call is itself holding. [`search_live`](@ref)'s `minrecall`-driven auto-`calibrate!`
    (a write) is therefore resolved *before* its own read lock is acquired, never nested
    inside it -- see that method's implementation for the pattern to follow for anything
    similar in the future.
"""
mutable struct ReadWriteLock
    readers::Int
    cond::Threads.Condition
end
ReadWriteLock() = ReadWriteLock(0, Threads.Condition())

"""
    read_lock(f::Function, rw::ReadWriteLock)

Runs `f()` with `rw` held as a reader -- concurrently with any other number of readers,
but never while a writer holds it (see [`ReadWriteLock`](@ref)).
"""
function read_lock(f::Function, rw::ReadWriteLock)
    lock(rw.cond) do
        rw.readers += 1
    end
    try
        return f()
    finally
        lock(rw.cond) do
            rw.readers -= 1
            rw.readers == 0 && notify(rw.cond)
        end
    end
end

"""
    write_lock(f::Function, rw::ReadWriteLock)

Runs `f()` with `rw` held exclusively -- no concurrent readers, no concurrent writer (see
[`ReadWriteLock`](@ref)). Never call this from code already running inside a
[`read_lock`](@ref) on the *same* `rw` -- see [`ReadWriteLock`](@ref)'s deadlock warning.
"""
function write_lock(f::Function, rw::ReadWriteLock)
    lock(rw.cond) do
        while rw.readers > 0
            wait(rw.cond)
        end
        return f()
    end
end

"""
    ContextPool{T}

A pool of `T` search contexts, each safe to hand to exactly one concurrent
[`search_live`](@ref) call at a time -- avoiding both of the two bad alternatives: sharing
one `T` across concurrent searches (unsafe -- its scratch buffers get corrupted by
concurrent mutation, confirmed directly by a segfault) and rebuilding a fresh `T` from
scratch on *every* call (wasteful -- `T`'s own scratch buffers, beam state,
visited-vertices, per-batch cost counters, are sized by `maxbatches`/similar and
non-trivial to allocate, so a high query rate would reallocate them constantly).

`template` is never mutated -- it's read-only, existing only to be [`deepcopy`](@ref)'d
into a new private worker when [`checkout!`](@ref) finds `available` empty. Every
returned worker goes back into `available` via [`checkin!`](@ref) for the next caller to
reuse instead of allocating again, so the pool grows lazily to (and then stays at) however
many *concurrent* searches this engine has actually seen at once -- never pre-allocated
speculatively, never shrunk either, on the assumption that peak concurrency is roughly
stable over an engine's lifetime.
"""
mutable struct ContextPool{T}
    template::T
    available::Vector{T}
    lock::ReentrantLock
end
ContextPool(template::T) where {T} = ContextPool{T}(template, T[], ReentrantLock())

"""
    checkout!(pool::ContextPool{T}) -> T

Hands back a `T` context exclusively owned by the caller until returned via
[`checkin!`](@ref) -- an existing idle one if `pool.available` has one, otherwise a fresh
[`deepcopy`](@ref) of `pool.template`.
"""
function checkout!(pool::ContextPool{T}) where {T}
    lock(pool.lock) do
        isempty(pool.available) ? deepcopy(pool.template) : pop!(pool.available)
    end
end

"""
    checkin!(pool::ContextPool{T}, ctx::T)

Returns a context obtained from [`checkout!`](@ref) to the pool once the caller is done
with it, for the next `checkout!` to reuse instead of allocating a new one.
"""
checkin!(pool::ContextPool{T}, ctx::T) where {T} = lock(pool.lock) do
    push!(pool.available, ctx)
end

"""
    OptBeamSearch

`minrecall::Float32 => BeamSearch` map of hyperparameters calibrated for each expected
recall quality (see [`calibrate!`](@ref) and [`SearchGraphEngine`](@ref)).
"""
const OptBeamSearch = Dict{Float32, BeamSearch}

"""
    DEFAULT_MINRECALL_LEVELS

The recall levels [`calibrate!`](@ref) populates a `SearchGraphEngine`'s
[`OptBeamSearch`](@ref) with when no explicit `levels` are given.
"""
const DEFAULT_MINRECALL_LEVELS = Float32[0.8, 0.9, 0.95, 0.97]

"""
    SearchGraphEngine

Dense engine backed by a `SearchGraph`. The only engine kind with a `BeamSearch` to
autotune, hence the only one carrying a `minrecall` target and an [`OptBeamSearch`](@ref)
table.

# Fields
- `index::SearchGraph`
- `ctx::SearchGraphContext`: reusable search/insertion context.
- `minrecall::Union{Nothing, Float32}`: the `MinRecall` target `BeamSearch` autotunes
  toward as the graph grows (`nothing` disables that autotuning; given to `ctx` itself, via
  `OptimizeParameters`, at construction time). Set once at `create_engine` time and carried
  through `restore_engine` so a reopened project keeps growing toward the same target
  instead of silently drifting to a different one.
- `opt_beamsearch::OptBeamSearch`: per-recall-level calibrated `BeamSearch`es, populated by
  [`calibrate!`](@ref) and consulted by [`search_live`](@ref) when a caller asks for a
  specific `minrecall` at query time -- refreshed by `calibrate!` since the right
  hyperparameters for a given recall level drift as the indexed data itself changes.
- `search_ctx_pool::ContextPool`: one private, plain `SearchGraphContext`
  per concurrent [`search_live`](@ref) call (see [`ContextPool`](@ref)) -- never `ctx`
  itself, which is reserved for insertion. Left unparameterized since `SearchGraphContext`
  itself is parametric (`SearchGraphContext{KnnType,VSType}`) and the concrete
  instantiation is an internal detail, not something worth pinning down in the field type.
- `deleted_ids::Set{UInt32}`: `UInt32` across every engine kind (see [`AbstractSearchEngine`](@ref))
  so the type doesn't depend on which concrete index kind is behind it.
- `lock::ReadWriteLock`
"""
mutable struct SearchGraphEngine <: AbstractSearchEngine
    index::SearchGraph
    ctx::SearchGraphContext
    minrecall::Union{Nothing, Float32}
    opt_beamsearch::OptBeamSearch
    search_ctx_pool::ContextPool
    deleted_ids::Set{UInt32}
    lock::ReadWriteLock
end

"""
    GenericEngine{IndexType}

Dense engine for an index kind with no engine-specific state of its own --
`ExhaustiveSearch`/`ParallelExhaustiveSearch` today, and any future dense index kind that
doesn't warrant a dedicated engine type.

# Fields
- `index::IndexType`
- `ctx::Any`: reusable insertion context.
- `search_ctx_pool::ContextPool`: one private context per concurrent [`search_live`](@ref)
  call (see [`ContextPool`](@ref)) -- never `ctx` itself, which is reserved for insertion.
- `deleted_ids::Set{UInt32}`: `UInt32` across every engine kind (see [`AbstractSearchEngine`](@ref)).
- `lock::ReadWriteLock`
"""
mutable struct GenericEngine{IndexType} <: AbstractSearchEngine
    index::IndexType
    ctx::Any
    search_ctx_pool::ContextPool
    deleted_ids::Set{UInt32}
    lock::ReadWriteLock
end

"""
    BM25Engine

Text engine backed by a `BM25InvertedFile`, scored via raw bags-of-words
(`bagofwords`/`bm25score`), so it never needs a vectorizing `model`. Left untrained
(`index === nothing`, `voc === nothing`) until [`ensure_trained!`](@ref) is called with a
real corpus, since a `BM25InvertedFile` needs a `Vocabulary` to be constructed at all.

# Fields
- `index::Union{Nothing, BM25InvertedFile}`
- `voc::Union{Nothing, TextSearch.Vocabulary}`
- `model::Union{Nothing, TextSearch.VectorModel}`: always `nothing` -- BM25 scores from
  raw bags-of-words, not a vectorized `model`; the field exists so every text engine
  shares the same shape.
- `ctx::InvertedFileContext`: built eagerly at `create_engine` time -- `InvertedFileContext`
  needs no vocabulary/index to exist, so there's no reason to leave this `nothing` until
  [`ensure_trained!`](@ref) the way `index`/`voc` must be. Reserved for insertion; never
  shared with a concurrent [`search_live`](@ref) call, which gets its own from
  `search_ctx_pool` instead.
- `search_ctx_pool::ContextPool`: one private context per concurrent
  [`search_live`](@ref) call (see [`ContextPool`](@ref)). Left unparameterized since
  `InvertedFileContext` itself is parametric (`InvertedFileContext{A,B}`).
- `deleted_ids::Set{UInt32}`: `UInt32` across every engine kind (see [`AbstractSearchEngine`](@ref)).
- `lock::ReadWriteLock`
"""
mutable struct BM25Engine <: AbstractSearchEngine
    index::Union{Nothing, BM25InvertedFile}
    voc::Union{Nothing, TextSearch.Vocabulary}
    model::Union{Nothing, TextSearch.VectorModel}
    ctx::InvertedFileContext
    search_ctx_pool::ContextPool
    deleted_ids::Set{UInt32}
    lock::ReadWriteLock
end

"""
    InvertedFileEngine

Text engine backed by a `SimilaritySearch.InvertedFile` (the general-purpose inverted
index `TextSearch.jl`'s `WeightedInvertedFile` convenience constructor itself just calls
with `Dist.NormCosine()`) -- vectorizing text into `SparseVector`s via a trained `model`
before indexing/querying, so it can be built against any `PreMetric` `distance`, not just
the cosine default. Left untrained (`index === nothing`, `voc === nothing`) until
[`ensure_trained!`](@ref) is called with a real corpus.

# Fields
- `index::Union{Nothing, InvertedFile}`
- `voc::Union{Nothing, TextSearch.Vocabulary}`
- `model::Union{Nothing, TextSearch.VectorModel}`: the trained tf-idf `VectorModel` used
  to turn text into the `SparseVector`s `index` needs.
- `distance::SimilaritySearch.PreMetric`: the distance chosen at `create_engine` time --
  remembered here since the real `InvertedFile` can't be built until
  [`ensure_trained!`](@ref) knows the vocabulary size.
- `ctx::InvertedFileContext`: built eagerly at `create_engine` time -- `InvertedFileContext`
  needs no vocabulary/index to exist, so there's no reason to leave this `nothing` until
  [`ensure_trained!`](@ref) the way `index`/`voc` must be. Reserved for insertion; never
  shared with a concurrent [`search_live`](@ref) call, which gets its own from
  `search_ctx_pool` instead.
- `search_ctx_pool::ContextPool`: one private context per concurrent
  [`search_live`](@ref) call (see [`ContextPool`](@ref)). Left unparameterized since
  `InvertedFileContext` itself is parametric (`InvertedFileContext{A,B}`).
- `deleted_ids::Set{UInt32}`: `UInt32` across every engine kind (see [`AbstractSearchEngine`](@ref))
  -- matches `InvertedFile`'s own posting lists (`AdjList{UInt32}`), which key ids by
  `UInt32` too.
- `lock::ReadWriteLock`
"""
mutable struct InvertedFileEngine <: AbstractSearchEngine
    index::Union{Nothing, InvertedFile}
    voc::Union{Nothing, TextSearch.Vocabulary}
    model::Union{Nothing, TextSearch.VectorModel}
    distance::SimilaritySearch.PreMetric
    ctx::InvertedFileContext
    search_ctx_pool::ContextPool
    deleted_ids::Set{UInt32}
    lock::ReadWriteLock
end

"""
    CallbackLog(flush::Function)

An `AbstractLog` backend (see `SimilaritySearch.jl`'s `log.jl`) that calls
`flush(index, sp, ep)` -- instead of printing, the way `InformativeLog` does -- on every
single `push_item!`/`append_items!`/`add!` report, forwarding exactly the range `sp:ep`
the library itself reports, with no batching/throttling of its own. Knows nothing about
RocksDB or any other storage backend; `flush` is supplied by the caller.

!!! warning "what `index` looks like inside `flush`, for a `SearchGraph`"
    `LOG` for a `SearchGraph` fires *before* `connect_reverse_links!` runs for `sp:ep`
    (`searchgraph/insertions.jl`) -- so `index.adj` for that exact range holds only the
    *direct* links just computed, none of the reverse links other nodes will later add
    into it. This is by design, not a bug to route around: [`searchgraph_vectors`](@ref)/
    [`direct_neighbors`](@ref) capture exactly that direct-links-only slice, and
    [`build_searchgraph`](@ref) reconnects every reverse link *once*, only after every
    saved vectors block and adjacency entry has been replayed -- reconnecting on an
    already-complete graph isn't safe (it isn't idempotent; it would duplicate reverse
    edges), which is exactly why this format never saves them in the first place.
    `InvertedFile`/`BM25InvertedFile` have no such hazard -- their own `LOG` calls happen
    only after all of a call's mutation is done, so [`invertedfile_objects`](@ref) can
    read `sp:ep`'s objects straight out of `index.db` with nothing left pending.
"""
mutable struct CallbackLog <: SimilaritySearch.AbstractLog
    flush::Function
end

function SimilaritySearch.LOG(log::CallbackLog, event::Symbol, index::SimilaritySearch.AbstractSearchIndex,
                               ctx::SimilaritySearch.AbstractContext, sp::Integer, ep::Integer)
    log.flush(index, sp, ep)
end

"""
    FileLog(io::IO; dt::Float64=1.0, prompt::String="LOG")

An `AbstractLog` backend that prints the exact same throttled status line
`SimilaritySearch.jl`'s own `InformativeLog` does, but to a caller-given `io` instead of a
hardcoded `stderr` -- an open file handle (e.g. `open(path, "a")`) or `stdout`/`stderr`
work identically, since both are just `IO`. Purely informative, like `InformativeLog`
itself: it never triggers any persistence of its own, unlike [`CallbackLog`](@ref), which
is the mechanism responsible for incrementally saving the index/vectors/adjacency to
RocksDB. The two are independent loggers fanned out together via `LogList` (see
[`_engine_logger`](@ref)) -- adding a `FileLog` changes nothing about what gets persisted.
"""
mutable struct FileLog <: SimilaritySearch.AbstractLog
    io::IO
    dt::Float64
    prompt::String
    last::Ref{Float64}
    lock::Threads.SpinLock
end
FileLog(io::IO; dt::Float64=1.0, prompt::String="LOG") = FileLog(io, dt, prompt, Ref(0.0), Threads.SpinLock())

function SimilaritySearch.LOG(log::FileLog, event::Symbol, index::SimilaritySearch.AbstractSearchIndex,
                               ctx::SimilaritySearch.AbstractContext, sp::Integer, ep::Integer)
    trylock(log.lock) || return
    try
        now = time()
        if log.last[] + log.dt < now
            n = length(index)
            mem = ceil(Int, Sys.total_memory() / 2^20)
            maxrss = ceil(Int, Sys.maxrss() / 2^20)
            println(log.io, log.prompt, " $event $(typeof(index)) sp=$sp ep=$ep n=$n mem=$(mem) max-rss=$(maxrss) $(Dates.now())")
            flush(log.io)
            log.last[] = now
        end
    finally
        unlock(log.lock)
    end
end

"""
    searchgraph_vectors(index::SearchGraph, sp::Integer, ep::Integer) -> NamedTuple

Range `sp:ep`'s raw vectors (`index.db`), read at the same moment `LOG` reports this
range -- the counterpart of [`direct_neighbors`](@ref), which captures that same range's
adjacency separately (each stored under its own object id, in a dedicated column family
-- see `Persistence.AdjacencyStore` -- rather than bundled here with the vectors, since
the two live in different RocksDB column families).
"""
searchgraph_vectors(index::SearchGraph, sp::Integer, ep::Integer) =
    (sp=Int(sp), ep=Int(ep), vectors=[database(index, i) for i in sp:ep])

"""
    direct_neighbors(index::SearchGraph, i::Integer) -> Vector{UInt32}

Object `i`'s direct (not yet reverse-connected) neighbor list, read at exactly the
moment `LOG` reports the range containing `i` -- i.e. before `connect_reverse_links!` has
added anything else into it (see [`CallbackLog`](@ref)'s warning).
"""
direct_neighbors(index::SearchGraph, i::Integer) = copy(neighbors(index.adj, i))

"""
    apply_searchgraph_vectors!(index::SearchGraph, block)

Replays one [`searchgraph_vectors`](@ref) block into `index`: appends its vectors to
`index.db`. Does *not* touch `index.adj`/`index.len[]` -- [`build_searchgraph`](@ref)
handles adjacency (via a separate per-object lookup) and finalizing length itself, once,
after every vectors block has been applied.
"""
function apply_searchgraph_vectors!(index::SearchGraph, block)
    for v in block.vectors
        push_item!(index.db, v)
    end
end

"""
    build_searchgraph(distance, vector_blocks, load_neighbors::Function) -> SearchGraph

Rebuilds a `SearchGraph` against `distance`: replays each of `vector_blocks` (as produced
incrementally during insertion via [`searchgraph_vectors`](@ref) and read back via
`Persistence.load_blocks`) in order via [`apply_searchgraph_vectors!`](@ref) to
reconstruct `index.db` and the object count `n`, then, for every object id `1:n`, calls
`load_neighbors(i)` (typically `i -> Persistence.load_neighbors(adjacency_store, i)`) to
get back its saved direct neighbor list and `add!`s it -- and only then connects every
reverse link exactly once over the whole result, safe (unlike calling it a second time on
an already-connected graph) precisely because a freshly rebuilt graph has none yet.
"""
function build_searchgraph(distance, vector_blocks, load_neighbors::Function)
    index = SearchGraph(distance, VectorDatabase())
    n = 0
    for block in vector_blocks
        apply_searchgraph_vectors!(index, block)
        n = block.ep
    end
    for i in 1:n
        add!(index.adj, i, load_neighbors(i))
    end
    index.len[] = n
    n > 0 && SimilaritySearch.connect_reverse_links!(index.adj, 1, n)
    return index
end

"""
    invertedfile_objects(index::AbstractInvertedFile, sp::Integer, ep::Integer) -> Vector

Range `sp:ep`'s raw indexed objects (`index.db` -- bags-of-words for a `BM25InvertedFile`,
`SparseVector`s for an `InvertedFile`), read at the moment `LOG` reports this range.
Unlike [`searchgraph_vectors`](@ref)'s `SearchGraph` counterpart, this can be read at any
point after the report -- `push_item!`/`append_items!` for an inverted file fully finalize
an object's contribution to the posting lists *before* `LOG` fires (see
[`CallbackLog`](@ref)'s docstring), so there's no direct/reverse-link ordering hazard here.

`database(index, i)` for a sparse-vector-backed index hands back a `Special.Sparse.
SparseVecView` -- a *view* into `index.db`'s own packed storage, fine for reading/scoring
but not directly re-insertable (confirmed empirically, two different ways it breaks):
- Materializing it into a plain `SparseVector` and pushing that into a fresh
  `InvertedFile` works (that's the same shape `add_item!`'s own `vectorize` already
  produces for a live insert).
- Pushing that *same* `SparseVector` into a fresh `BM25InvertedFile` does not:
  `push_item!(idx::BM25InvertedFile, ctx, doc::T) where T<:Union{AbstractString,
  AbstractVector,TokenizedText}` (`TextSearch.BM25`'s tokenizing overload) is more
  specific than the generic `push_item!(idx, ctx, obj)` for any `SparseVector` (it *is* an
  `AbstractVector`), so it gets (mis)treated as literal pre-tokenized input instead of an
  already-computed bag. `BM25InvertedFile` needs a genuine `BOW` (`Dict{UInt32,Int32}`)
  instead, which the generic method's `pairiterator(::Dict) = d` accepts directly. Hence
  [`_materialize`](@ref) dispatches on the *target index type*, not just the view.
"""
invertedfile_objects(index::AbstractInvertedFile, sp::Integer, ep::Integer) = [_materialize(index, database(index, i)) for i in sp:ep]

_materialize(::AbstractInvertedFile, obj) = obj
_materialize(::AbstractInvertedFile, v::SimilaritySearch.Special.Sparse.SparseVecView) = SparseVector(v.n, collect(v.nzind), collect(v.nzval))
_materialize(::BM25InvertedFile, v::SimilaritySearch.Special.Sparse.SparseVecView) =
    Dict{UInt32,Int32}(UInt32(id) => Int32(freq) for (id, freq) in zip(v.nzind, v.nzval))

"""
    build_bm25invertedfile(voc::TextSearch.Vocabulary, object_blocks) -> BM25InvertedFile

Rebuilds a `BM25InvertedFile` against a trained `voc` by replaying every saved raw object
(as produced incrementally via [`invertedfile_objects`](@ref) and read back via
`Persistence.load_object_blocks`) through the library's own `push_item!`, in order -- a
full rebuild-by-reinsertion, not an incremental deserialize (see
`Persistence.InvertedFileObjectStore`'s docstring for why, and its documented scaling
limits).
"""
function build_bm25invertedfile(voc, object_blocks)
    index = BM25InvertedFile(voc)
    ctx = InvertedFileContext()
    for block in object_blocks, obj in block
        push_item!(index, ctx, obj)
    end
    return index
end

"""
    build_invertedfile(distance, voc::TextSearch.Vocabulary, object_blocks) -> InvertedFile

Rebuilds an `InvertedFile` against `distance` (with room for `voc`'s vocabulary size) by
replaying every saved raw object through the library's own `push_item!`, in order -- see
[`build_bm25invertedfile`](@ref) (same rebuild-by-reinsertion approach and scaling caveat).
"""
function build_invertedfile(distance, voc, object_blocks)
    index = InvertedFile(max(vocsize(voc), 1), distance)
    ctx = InvertedFileContext()
    for block in object_blocks, obj in block
        push_item!(index, ctx, obj)
    end
    return index
end

# `on_change`, given to every `create_engine`/`restore_engine` method below, is an
# optional `(index, sp, ep) -> nothing` callback -- typically a closure over an on-disk
# store -- fired via a `CallbackLog` fanned out alongside the library's own default
# `InformativeLog` (via `LogList`, so the existing console progress printing isn't lost).
# `log_io`, also given to every `create_engine`/`restore_engine` method, is an optional
# extra `IO` (an open file handle, or `stdout`/`stderr`) to *additionally* print the same
# informative status line to via `FileLog` -- purely informative, like the default
# `InformativeLog` it's fanned out alongside; it never changes what gets persisted.
function _engine_logger(on_change::Union{Nothing,Function}, log_io::Union{Nothing,IO}=nothing)
    loggers = SimilaritySearch.AbstractLog[InformativeLog()]
    log_io === nothing || push!(loggers, FileLog(log_io))
    on_change === nothing || push!(loggers, CallbackLog(on_change))
    return length(loggers) == 1 ? loggers[1] : SimilaritySearch.LogList(loggers)
end

# `hyperparameters_callback` on a `SearchGraphContext` drives `OptimizeParameters`'
# in-band autotuning of `BeamSearch` toward a `MinRecall` target during index
# construction/growth -- every `SearchGraphEngine` gets one from an explicit `minrecall`
# given at `create_engine` time (default 0.9) rather than running on the library's own
# untuned defaults while it grows; `calibrate!` is a separate, explicit re-optimization
# pass layered on top, not the sole writer of `engine.index.algo[]`.
_searchgraph_context(minrecall::Nothing, logger) = SearchGraphContext(; hyperparameters_callback=nothing, logger)
_searchgraph_context(minrecall::Real, logger) = SearchGraphContext(; hyperparameters_callback=OptimizeParameters(MinRecall(Float32(minrecall))), logger)

"""
    create_engine(::Type{SearchGraph}; distance=SimilaritySearch.Dist.SqL2(), minrecall::Union{Nothing,Real}=0.9, on_change=nothing, log_io=nothing) -> SearchGraphEngine
    create_engine(::Type{ExhaustiveSearch}; distance=SimilaritySearch.Dist.SqL2(), on_change=nothing, log_io=nothing) -> GenericEngine{ExhaustiveSearch}
    create_engine(::Type{ParallelExhaustiveSearch}; distance=SimilaritySearch.Dist.SqL2(), on_change=nothing, log_io=nothing) -> GenericEngine{ParallelExhaustiveSearch}

Creates a new, empty dense search engine of the given index type, built immediately
against `distance`. `minrecall` only means anything for a `SearchGraph`; it's accepted
(and ignored) on the other two so a caller can pass the same keyword set uniformly
regardless of index type. `on_change::Union{Nothing,Function}`, if given, is installed
(via [`CallbackLog`](@ref)) as an `(index, sp, ep) -> nothing` callback fired on every
`push_item!`/`append_items!`/`add!` report -- e.g. to persist the range `sp:ep` that was
just inserted. `log_io::Union{Nothing,IO}`, if given, additionally prints the same
throttled informative status line [`InformativeLog`](@ref) already prints to `stderr`
to this `IO` too (via [`FileLog`](@ref)) -- an open file handle or `stdout`/`stderr`
both work; purely informative, changes nothing about what gets persisted.
"""
function create_engine(::Type{SearchGraph}; distance=SimilaritySearch.Dist.SqL2(), minrecall::Union{Nothing,Real}=0.9, on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing)
    mr = minrecall === nothing ? nothing : Float32(minrecall)
    return SearchGraphEngine(SearchGraph(distance, VectorDatabase()), _searchgraph_context(mr, _engine_logger(on_change, log_io)), mr, OptBeamSearch(), ContextPool(SearchGraphContext()), Set{UInt32}(), ReadWriteLock())
end
create_engine(::Type{ExhaustiveSearch}; distance=SimilaritySearch.Dist.SqL2(), minrecall=nothing, on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing) =
    GenericEngine{ExhaustiveSearch}(ExhaustiveSearch(distance, VectorDatabase()), GenericContext(; logger=_engine_logger(on_change, log_io)), ContextPool(GenericContext()), Set{UInt32}(), ReadWriteLock())
create_engine(::Type{ParallelExhaustiveSearch}; distance=SimilaritySearch.Dist.SqL2(), minrecall=nothing, on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing) =
    GenericEngine{ParallelExhaustiveSearch}(ParallelExhaustiveSearch(distance, VectorDatabase()), GenericContext(; logger=_engine_logger(on_change, log_io)), ContextPool(GenericContext()), Set{UInt32}(), ReadWriteLock())

"""
    create_engine(::Type{BM25InvertedFile}; distance=nothing, minrecall=nothing, on_change=nothing, log_io=nothing) -> BM25Engine
    create_engine(::Type{InvertedFile}; distance=SimilaritySearch.Dist.NormCosine(), minrecall=nothing, on_change=nothing, log_io=nothing) -> InvertedFileEngine

Creates a new, untrained text search engine of the given index type (`index === nothing`
until [`ensure_trained!`](@ref) is called with a real corpus, since both need a
`Vocabulary` to be constructed at all). `minrecall` is accepted and ignored on both, and
`distance` is accepted and ignored on `BM25InvertedFile` (BM25 always scores via its own
`bm25score`), so a caller can pass the same keyword set uniformly regardless of index type.
`on_change`/`log_io` are as in the dense `create_engine` methods above.
"""
create_engine(::Type{BM25InvertedFile}; distance=nothing, minrecall=nothing, on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing) =
    BM25Engine(nothing, nothing, nothing, InvertedFileContext(; logger=_engine_logger(on_change, log_io)), ContextPool(InvertedFileContext()), Set{UInt32}(), ReadWriteLock())
create_engine(::Type{InvertedFile}; distance=SimilaritySearch.Dist.NormCosine(), minrecall=nothing, on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing) =
    InvertedFileEngine(nothing, nothing, nothing, distance, InvertedFileContext(; logger=_engine_logger(on_change, log_io)), ContextPool(InvertedFileContext()), Set{UInt32}(), ReadWriteLock())

"""
    snapshot_state(engine::AbstractSearchEngine) -> NamedTuple

The full set of `engine`'s state that must survive a save (see `Persistence.EngineStore`)
-- `ctx`/`lock` are transient and rebuilt fresh by [`restore_engine`](@ref). The returned
`kind` field records which concrete type to reconstruct: `SearchGraphEngine`/`BM25Engine`/
`InvertedFileEngine` themselves for those three, or the wrapped index's own type for a
`GenericEngine` -- the type value is the discriminator `restore_engine` dispatches on,
standing in for what a separate enum tag would otherwise have to do. Used for a one-shot
full save (e.g. right after `create_engine`, when there's nothing indexed yet); routine
mutations instead persist just the one or two fields they touched (see
[`extra_state_fields`](@ref) for the field lists a caller needs to read back *before* an
engine instance exists to call this on).

None of `SearchGraphEngine`/`BM25Engine`/`InvertedFileEngine` has a plain `:index` field
here -- unlike `GenericEngine`, none of the three ever saves its index as a single value.
`SearchGraphEngine` includes `distance` instead (see [`CallbackLog`](@ref)/
[`searchgraph_vectors`](@ref)/[`direct_neighbors`](@ref)/[`build_searchgraph`](@ref)); the
two text engines include `voc`/(`model`/`distance` for `InvertedFileEngine`) instead (see
[`invertedfile_objects`](@ref)/[`build_bm25invertedfile`](@ref)/
[`build_invertedfile`](@ref)) -- all three need their saved insertion blocks replayed
through a dedicated `build_*` function to get `index` back, not a plain field read.
"""
snapshot_state(engine::SearchGraphEngine) =
    (kind=SearchGraphEngine, distance=engine.index.dist, minrecall=engine.minrecall, opt_beamsearch=engine.opt_beamsearch, deleted_ids=engine.deleted_ids)
snapshot_state(engine::GenericEngine{IndexType}) where {IndexType} =
    (kind=IndexType, index=engine.index, deleted_ids=engine.deleted_ids)
snapshot_state(engine::BM25Engine) =
    (kind=BM25Engine, voc=engine.voc, deleted_ids=engine.deleted_ids)
snapshot_state(engine::InvertedFileEngine) =
    (kind=InvertedFileEngine, voc=engine.voc, model=engine.model, distance=engine.distance, deleted_ids=engine.deleted_ids)

"""
    extra_state_fields(kind::Type) -> Tuple{Vararg{Symbol}}

The field names [`snapshot_state`](@ref) includes for `kind` *beyond* `:deleted_ids`
(every kind has that one) and `:index` (only `GenericEngine`'s wrapped index types have
that as a plain field) -- the static, per-kind list a loader needs before an engine
instance exists to call `snapshot_state` on. The default (empty) applies to
`GenericEngine`'s wrapped index types (`ExhaustiveSearch`, `ParallelExhaustiveSearch`,
...). `SearchGraphEngine`/`BM25Engine`/`InvertedFileEngine` are deliberately not covered
here -- none of them has a plain `:index` field to read back this way at all (see
`snapshot_state`'s docstring); loading any of them always needs its own dedicated
distance/vocabulary + insertion-block-sequence path instead.
"""
extra_state_fields(::Type) = ()

"""
    restore_engine(state; on_change=nothing, log_io=nothing) -> AbstractSearchEngine

Rebuilds a concrete engine from `state` (as produced by [`snapshot_state`](@ref), or
assembled field-by-field via [`extra_state_fields`](@ref) from an `EngineStore`),
dispatching on `state.kind` to reconstruct exactly the engine type that was saved --
including, for a `SearchGraphEngine`, the `minrecall` target it was growing toward, so a
reopened project keeps autotuning toward the same target instead of reverting to some
other default. `on_change`/`log_io` are as in [`create_engine`](@ref).

None of `SearchGraphEngine`/`BM25Engine`/`InvertedFileEngine`'s `state` carries a plain
`index` -- each carries what its own `build_*` function needs instead:
- `SearchGraphEngine`: `distance`, `vector_blocks` (`Persistence.load_blocks(store, :index)`),
  `load_neighbors` (typically `i -> Persistence.load_neighbors(adjacency_store, i)`) -- see
  [`build_searchgraph`](@ref).
- `BM25Engine`: `voc`, `object_blocks` (`Persistence.load_object_blocks(obj_store)`, or
  `nothing`/empty if `voc === nothing`, i.e. never trained) -- see
  [`build_bm25invertedfile`](@ref).
- `InvertedFileEngine`: `voc`, `distance`, `object_blocks` likewise -- see
  [`build_invertedfile`](@ref).
"""
restore_engine(state; on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing) = restore_engine(state.kind, state; on_change, log_io)

function restore_engine(::Type{SearchGraphEngine}, state; on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing)
    index = build_searchgraph(state.distance, state.vector_blocks, state.load_neighbors)
    return SearchGraphEngine(index, _searchgraph_context(state.minrecall, _engine_logger(on_change, log_io)), state.minrecall, state.opt_beamsearch, ContextPool(SearchGraphContext()), state.deleted_ids, ReadWriteLock())
end

function restore_engine(::Type{IndexType}, state; on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing) where {IndexType}
    return GenericEngine{IndexType}(state.index, GenericContext(; logger=_engine_logger(on_change, log_io)), ContextPool(GenericContext()), state.deleted_ids, ReadWriteLock())
end

function restore_engine(::Type{BM25Engine}, state; on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing)
    index = state.voc === nothing ? nothing : build_bm25invertedfile(state.voc, state.object_blocks)
    return BM25Engine(index, state.voc, nothing, InvertedFileContext(; logger=_engine_logger(on_change, log_io)), ContextPool(InvertedFileContext()), state.deleted_ids, ReadWriteLock())
end

function restore_engine(::Type{InvertedFileEngine}, state; on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing)
    index = state.voc === nothing ? nothing : build_invertedfile(state.distance, state.voc, state.object_blocks)
    return InvertedFileEngine(index, state.voc, state.model, state.distance, InvertedFileContext(; logger=_engine_logger(on_change, log_io)), ContextPool(InvertedFileContext()), state.deleted_ids, ReadWriteLock())
end

is_text_index(::AbstractSearchEngine) = false
is_text_index(::Union{BM25Engine, InvertedFileEngine}) = true

"""
    ensure_trained!(engine::AbstractSearchEngine, corpus::AbstractVector{<:AbstractString})

Trains a text engine's `Vocabulary` (and builds its real `BM25InvertedFile`/`InvertedFile`)
from `corpus` if it hasn't been trained yet. No-op for dense engines (`SearchGraphEngine`/
`GenericEngine`), for an already-trained text engine, and for an empty `corpus`.

Must be called once with the *first* batch of text a text-index project receives (e.g. the
first `append`/`build` batch) — `TextSearch.jl`'s `Vocabulary` is trained once from a
corpus; tokens not seen at training time are treated as out-of-vocabulary and silently
dropped on every later append (a library limitation, not a bug).
"""
ensure_trained!(::Union{SearchGraphEngine, GenericEngine}, corpus::AbstractVector) = nothing

function ensure_trained!(engine::BM25Engine, corpus::AbstractVector)
    engine.voc !== nothing && return
    isempty(corpus) && return

    write_lock(engine.lock) do
        engine.voc !== nothing && return
        voc = Vocabulary(TextConfig(), String.(corpus))
        engine.voc = voc
        engine.index = BM25InvertedFile(voc)
    end
end

function ensure_trained!(engine::InvertedFileEngine, corpus::AbstractVector)
    engine.voc !== nothing && return
    isempty(corpus) && return

    write_lock(engine.lock) do
        engine.voc !== nothing && return
        voc = Vocabulary(TextConfig(), String.(corpus))
        engine.voc = voc
        engine.index = InvertedFile(max(vocsize(voc), 1), engine.distance)
        engine.model = VectorModel(IdfWeighting(), TfWeighting(), voc)
    end
end

"""
    current_beamsearch(engine::AbstractSearchEngine) -> Union{Nothing, BeamSearch}

The `BeamSearch` configuration currently installed on a `SearchGraphEngine`'s index
(`nothing` for anything else — exact indices have no beam to configure, text indices
aren't dense at all). This is `SimilaritySearch.jl`'s own `BeamSearch()` default until
`calibrate!` has been called at least once (or until the growth autotuner has had a chance
to run), since construction no longer installs a calibrated `BeamSearch` synchronously.
"""
current_beamsearch(engine::SearchGraphEngine) = read_lock(() -> engine.index.algo[], engine.lock)
current_beamsearch(::AbstractSearchEngine) = nothing

"""
    calibrate!(engine::SearchGraphEngine; levels=DEFAULT_MINRECALL_LEVELS, numqueries=64, ksearch=10, queries=nothing) -> OptBeamSearch

Runs `SimilaritySearch.jl`'s real `optimize_index!` hyperparameter sweep (PLAN.md §5.6 —
a `SearchModels`-driven stochastic search over `BeamSearchSpace`, not a hand-rolled one)
against `engine.index` once per recall level in `levels`, storing each resulting
`BeamSearch` into `engine.opt_beamsearch[level]` (and returning that table) so
[`search_live`](@ref) can later serve a per-request `minrecall` from calibrated
hyperparameters instead of a single one-size-fits-all default. Only defined for a
`SearchGraphEngine`; errors for any other engine kind, which has no `BeamSearch` to
calibrate. Re-running this periodically as the indexed data changes keeps `opt_beamsearch`
accurate -- the right hyperparameters for a given recall level drift as the index grows.

# Keyword Arguments
- `levels`: a single target recall or a collection of them (0-1) to calibrate for — see
  `MinRecall`. Defaults to [`DEFAULT_MINRECALL_LEVELS`](@ref).
- `numqueries`: size of the query sample drawn from the already-indexed database when
  `queries` isn't given.
- `ksearch`: neighbors retrieved per query while evaluating candidate configurations.
- `queries::Union{Nothing, AbstractVector{<:AbstractVector{Float32}}}`: an explicit
  ground-truth query set (PLAN.md §5.6's "supplied ... set") instead of an internally
  sampled one.
"""
function calibrate!(engine::SearchGraphEngine; levels=DEFAULT_MINRECALL_LEVELS, numqueries::Int=64, ksearch::Int=10, queries=nothing)
    Q = queries === nothing ? nothing : VectorDatabase([convert(Vector{Float32}, q) for q in queries])
    target_levels = levels isa Real ? Float32[levels] : Float32.(levels)

    write_lock(engine.lock) do
        for r in target_levels
            optimize_index!(engine.index, engine.ctx, MinRecall(r); queries=Q, numqueries, ksearch)
            engine.opt_beamsearch[r] = engine.index.algo[]
        end
    end

    return engine.opt_beamsearch
end
calibrate!(::AbstractSearchEngine; kwargs...) = error("calibrate! only applies to a SearchGraphEngine (approximate dense) index")

"""
    _nearest_beamsearch(engine::SearchGraphEngine, minrecall::Real) -> BeamSearch

The `BeamSearch` calibrated for `minrecall` (see [`calibrate!`](@ref)), auto-calibrating
[`DEFAULT_MINRECALL_LEVELS`](@ref) first if `engine.opt_beamsearch` is still empty. An
exact match is used if present; otherwise the closest calibrated level *at or below*
`minrecall` is used, falling back to the lowest calibrated level available if `minrecall`
sits below all of them.
"""
function _nearest_beamsearch(engine::SearchGraphEngine, minrecall::Real)
    isempty(engine.opt_beamsearch) && calibrate!(engine)
    target = Float32(minrecall)
    haskey(engine.opt_beamsearch, target) && return engine.opt_beamsearch[target]
    levels = keys(engine.opt_beamsearch)
    below = Iterators.filter(<=(target), levels)
    key = isempty(below) ? minimum(levels) : maximum(below)
    return engine.opt_beamsearch[key]
end

"""
    mark_deleted!(engine::AbstractSearchEngine, doc_id::Integer)

Marks a document as logically deleted. `doc_id` is converted to `UInt32`, matching
`deleted_ids`'s element type across every engine kind (see [`AbstractSearchEngine`](@ref)).
"""
function mark_deleted!(engine::AbstractSearchEngine, doc_id::Integer)
    write_lock(engine.lock) do
        push!(engine.deleted_ids, UInt32(doc_id))
    end
end

# Insertion helpers for dense indices
insert_dense!(index::SearchGraph, ctx, item) = append_items!(index, ctx, VectorDatabase([item]))
insert_dense!(index::Union{ExhaustiveSearch,ParallelExhaustiveSearch}, ctx, item) = push_item!(index, ctx, item)
insert_dense!(index, ctx, item) = push_item!(index, ctx, item)

"""
    add_item!(engine::AbstractSearchEngine, item)

Adds a single item to the index. Thread-safe wrapper.
For text engines, `item` is raw text and `ensure_trained!` must already have been called
(with the batch `item` belongs to) or the item is silently dropped (untrained engine has no index yet).
"""
function add_item!(engine::Union{SearchGraphEngine, GenericEngine}, item)
    write_lock(engine.lock) do
        insert_dense!(engine.index, engine.ctx, item)
    end
end

function add_item!(engine::BM25Engine, item)
    write_lock(engine.lock) do
        engine.voc === nothing && return # untrained text engine: nothing to index into yet
        push_item!(engine.index, engine.ctx, bagofwords(engine.voc, item))
    end
end

function add_item!(engine::InvertedFileEngine, item)
    write_lock(engine.lock) do
        engine.voc === nothing && return # untrained text engine: nothing to index into yet
        # TextSearch.jl dropped Dict-based dot/norm/evaluate, so a raw BOW no longer
        # scores correctly against a NormCosine (or other) InvertedFile -- it needs a
        # real weighted SparseVector, hence `vectorize` via the trained `model`.
        push_item!(engine.index, engine.ctx, vectorize(engine.model, item))
    end
end

"""
    search_live(engine::AbstractSearchEngine, query, k::Int; bs_override=nothing) -> (id=..., dist=..., deleted=...)

Performs a plain top-`k` search and reports each candidate's soft-delete status instead of
hiding deleted candidates and silently backfilling behind them: `deleted[i]` tells the
caller whether `id[i]` is a soft-deleted document, rather than that document being dropped
and a live one from further down the ranking taking its place under the same `k` slots.
Compensating for deletions by overfetching here doesn't scale (the overfetch would have to
grow with the total number of deletions) and it hides from the caller which ids were even
considered; a caller that wants `k` *live* results after filtering out the deleted ones is
expected to page for more (e.g. via a cursor over successive top-`k'` windows) at a layer
above this one, which is also where such a policy belongs.

# Keyword Arguments
- `bs_override::Union{Nothing, BeamSearch}`: an ad hoc `BeamSearch` configuration to
  search with for this call only (PLAN.md §3/§7's per-request `beamsearch_overrides`),
  instead of `engine.index.algo[]`'s calibrated default. Only meaningful for a
  `SearchGraphEngine` — accepted and ignored on every other engine kind, which has no
  `BeamSearch` to override in the first place. Takes priority over `minrecall` below.
- `minrecall::Union{Nothing, Real}`: search at (approximately) this target recall instead
  of `engine.index.algo[]`'s current default, by looking up the matching calibrated
  `BeamSearch` in `engine.opt_beamsearch` (see [`calibrate!`](@ref)/[`_nearest_beamsearch`](@ref)
  — auto-calibrating [`DEFAULT_MINRECALL_LEVELS`](@ref) first if that table is still empty).
  Only meaningful for a `SearchGraphEngine`; accepted and ignored on every other engine
  kind, which has no calibrated `BeamSearch` table to consult.

!!! warning "Concurrency: a *read* lock, not exclusive -- multiple searches run concurrently"
    None of the underlying index kinds (`SearchGraph`, `ExhaustiveSearch`/
    `ParallelExhaustiveSearch`, `BM25InvertedFile`, `InvertedFile`) support searching
    concurrently with an insertion into the *same* index -- a search reading `index.adj`/
    `index.db` while `add_item!` is resizing/appending to those same arrays is exactly the
    kind of concurrent read-during-mutation none of them are built to tolerate. But all of
    them *do* support running multiple searches concurrently against the same index --
    that's exactly what their own `searchbatch`/`allknn`/`@BATCHES`-driven parallel search
    paths already rely on internally. So `search_live` takes `engine.lock` as a
    [`read_lock`](@ref): concurrent with any other number of searches, exclusive of
    `add_item!`/`ensure_trained!`/`mark_deleted!`/`calibrate!` (each a [`write_lock`](@ref))
    and of each other. See [`ReadWriteLock`](@ref) for the full contract, including why a
    `SearchGraphEngine`'s own `minrecall`-driven auto-`calibrate!` below has to be resolved
    *before* the read lock is acquired, not nested inside it.

    That guarantees the *index data* (`index.adj`/`index.db`) tolerates concurrent reads,
    but not by itself enough: `engine.ctx` is a *mutable, reused* scratch context (beam
    buffers, visited-vertices state, per-batch cost counters, ...), and two concurrent
    searches sharing that same object corrupt each other's scratch space -- confirmed
    directly, a segfault inside `SearchGraph`'s own `beamsearch_inner_beam` from two
    threads racing on shared visited-vertices state via a shared `ctx`. So every
    `search_live` method borrows a private context from [`ContextPool`](@ref) via
    [`checkout!`](@ref)/[`checkin!`](@ref) instead of reusing `engine.ctx` (which stays
    reserved for insertion, itself already exclusive via `write_lock`, so sharing it
    *there* is fine) or allocating a brand-new context on every single call.
"""
function search_live(engine::SearchGraphEngine, query, k::Int; bs_override=nothing, minrecall=nothing)
    # Resolved (and, if needed, `calibrate!`'s own write lock acquired) *before* taking our
    # own read lock below -- see `ReadWriteLock`'s deadlock warning for why this can't be
    # nested inside the `read_lock` block instead.
    bs = bs_override !== nothing ? bs_override :
         minrecall !== nothing ? _nearest_beamsearch(engine, minrecall) : nothing
    read_lock(engine.lock) do
        ctx = checkout!(engine.search_ctx_pool)
        try
            res = knnqueue(KnnSorted, max(k, 1))
            if bs !== nothing
                vstate = SimilaritySearch.getvstate(length(engine.index), ctx)
                search(bs, engine.index, ctx, query, res, engine.index.hints, vstate)
            else
                search(engine.index, ctx, query, res)
            end
            return _collect_live(engine, IdView(res), DistView(res))
        finally
            checkin!(engine.search_ctx_pool, ctx)
        end
    end
end

function search_live(engine::GenericEngine, query, k::Int; bs_override=nothing, minrecall=nothing)
    read_lock(engine.lock) do
        ctx = checkout!(engine.search_ctx_pool)
        try
            res = knnqueue(KnnSorted, max(k, 1))
            search(engine.index, ctx, query, res)
            return _collect_live(engine, IdView(res), DistView(res))
        finally
            checkin!(engine.search_ctx_pool, ctx)
        end
    end
end

function search_live(engine::BM25Engine, query, k::Int; bs_override=nothing, minrecall=nothing)
    read_lock(engine.lock) do
        engine.voc === nothing && return (id=Int32[], dist=Float32[], deleted=Bool[])
        ctx = checkout!(engine.search_ctx_pool)
        try
            res = knnqueue(KnnSorted, max(k, 1))
            search(engine.index, ctx, bagofwords(engine.voc, query), res)
            return _collect_live(engine, IdView(res), DistView(res))
        finally
            checkin!(engine.search_ctx_pool, ctx)
        end
    end
end

function search_live(engine::InvertedFileEngine, query, k::Int; bs_override=nothing, minrecall=nothing)
    read_lock(engine.lock) do
        engine.voc === nothing && return (id=Int32[], dist=Float32[], deleted=Bool[])
        ctx = checkout!(engine.search_ctx_pool)
        try
            res = knnqueue(KnnSorted, max(k, 1))
            search(engine.index, ctx, vectorize(engine.model, query), res)
            return _collect_live(engine, IdView(res), DistView(res))
        finally
            checkin!(engine.search_ctx_pool, ctx)
        end
    end
end

function _collect_live(engine::AbstractSearchEngine, cand_ids, cand_dists)
    final_ids = Int32[]
    final_dists = Float32[]
    final_deleted = Bool[]

    for (cand_id, cand_dist) in zip(cand_ids, cand_dists)
        cand_id == 0 && continue # 0 means empty slot
        push!(final_ids, cand_id)
        push!(final_dists, cand_dist)
        push!(final_deleted, cand_id in engine.deleted_ids)
    end

    return (id=final_ids, dist=final_dists, deleted=final_deleted)
end

end # module
