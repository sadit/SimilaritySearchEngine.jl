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
export create_engine, restore_engine, snapshot_state, extra_state_fields, add_item!, index!, search_live, mark_deleted!, is_text_index
export calibrate!, current_beamsearch, OptBeamSearch, DEFAULT_MINRECALL_LEVELS, CallbackLog, FileLog
export direct_neighbors, apply_searchgraph_vectors!, build_searchgraph
export invertedfile_objects, build_bm25invertedfile, build_textinvertedfile
export text_profile, text_vocabulary, resolve_query, fit_profile
export AbstractTextModelSpec, BaseProfile, FitFromCorpus, is_text_index_type, validate_textmodel

"""
    AbstractSearchEngine

Common supertype for every concrete engine kind ([`SearchGraphEngine`](@ref),
[`GenericEngine`](@ref), [`BM25Engine`](@ref), [`InvertedFileEngine`](@ref)). Each kind
carries only the state it actually needs -- e.g. `minrecall` only exists on
`SearchGraphEngine`, `profile` only on the text engines -- and the concrete Julia type
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
each other, exclusive of any write), while `add_item!`/`index!`/
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
    AbstractTextModelSpec

The text-model decision a text project is created with, as a value: either
[`BaseProfile`](@ref) (index against a model fitted elsewhere) or [`FitFromCorpus`](@ref)
(fit one from this project's own corpus).

There is no default and no `nothing`. The two are not variations on one setting, they are
opposite answers to "where does the vocabulary come from", and the difference outlives the
call: a fitted-here vocabulary is frozen at the first [`index!`](@ref index!(::BM25Engine))
call, so every term a later batch introduces is out-of-vocabulary and silently dropped from
then on. That is a fine trade for a closed corpus and a bad surprise for a growing one, which
is why [`create_engine`](@ref) makes a text project name which one it wants rather than
picking the cheap one by default.

Each is a type rather than a flag, and carries its *own* related options, so an option that
only means something on one path cannot be handed to the other -- the same reason the engine
hierarchy discriminates on concrete types instead of an enum tag.
"""
abstract type AbstractTextModelSpec end

"""
    BaseProfile(profile::TextProfile)

Index against `profile` -- a `TextSearch.TextProfile` fitted elsewhere, typically
`load_profile("wiki20231101-es.zip")` over one of `TextSearch.jl`'s corpus profiles.

The project is trained from the moment it is created: nothing is inferred from the data
appended later, the vocabulary covers the language rather than whichever batch arrived first,
and the project gains whatever stopword set, lemma map and query-expansion network the profile
carries.

Which of those artifacts are *applied* is a property of the profile, not of this wrapper, and
it stays that way: change it before handing it over, with `TextSearch`'s own
`with_applied(p; lemmas=false)`. Re-exposing the same switches here would be a second place to
say the same thing, and the profile is what gets persisted.
"""
struct BaseProfile <: AbstractTextModelSpec
    profile::TextProfile
end

"""
    FitFromCorpus(textconfig::TextConfig=TextConfig();
                  local_weighting=TfWeighting(), global_weighting=IdfWeighting(),
                  min_ndocs=1, min_occs=1, stopwords=nothing)

Fit this project's text model from its own staged corpus, under `textconfig` -- the
corpus-independent policy (normalization, tokenization, `language`), with no base profile.
This is the explicit form of "I have no pre-fitted model": see [`AbstractTextModelSpec`](@ref)
for the out-of-vocabulary consequence it accepts.

The fit is deferred to the first [`index!`](@ref index!(::BM25Engine)) call, which runs it
over everything staged by then -- whatever you have appended before that call *is* the
training corpus.

# Options
- `local_weighting`/`global_weighting`: the `VectorModel` weighting scheme (`TfWeighting`,
  `TpWeighting`, `FreqWeighting`, `BinaryLocalWeighting` × `IdfWeighting`,
  `BinaryGlobalWeighting`, `EntropyWeighting`). Only the weighted index kind scores through
  it; a `BM25InvertedFile` scores from raw bags, and carries the model for the query side.
- `min_ndocs`/`min_occs`: drop a token seen in fewer than `min_ndocs` documents, or fewer than
  `min_occs` times overall, before the model is built. Both default to `1` (keep everything).
  Worth raising on real corpora: the long tail of hapaxes is most of a vocabulary's size and
  almost none of its retrieval value, and it is where unaccented misspellings and
  foreign-language fragments live.
- `stopwords`: a document-frequency threshold in `(0, 1]`, or `nothing` (the default) not to
  detect any. When given, tokens above the threshold are flagged by
  `TextSearch.stopword_candidates` and the vocabulary is rebuilt with them filtered out --
  which costs a **second pass over the corpus**, because the counts have to be recomputed
  under the pipeline that drops them. A frequency heuristic only, and per spelling rather than
  per word (`stopword_candidates` explains what that costs); it is the one artifact estimable
  from an indexing corpus at all -- a lemma map needs word embeddings and an expansion network
  needs an LSI, which is what a [`BaseProfile`](@ref) brings instead.
"""
struct FitFromCorpus <: AbstractTextModelSpec
    textconfig::TextConfig
    local_weighting::LocalWeighting
    global_weighting::GlobalWeighting
    min_ndocs::Int
    min_occs::Int
    stopwords::Union{Nothing,Float64}

    function FitFromCorpus(textconfig::TextConfig=TextConfig();
                           local_weighting::LocalWeighting=TfWeighting(),
                           global_weighting::GlobalWeighting=IdfWeighting(),
                           min_ndocs::Integer=1, min_occs::Integer=1,
                           stopwords::Union{Nothing,Real}=nothing)
        min_ndocs >= 1 || throw(ArgumentError("min_ndocs must be at least 1; got $min_ndocs"))
        min_occs >= 1 || throw(ArgumentError("min_occs must be at least 1; got $min_occs"))
        stopwords === nothing || 0 < stopwords <= 1 ||
            throw(ArgumentError("stopwords must be a document-frequency threshold in (0, 1], or nothing; got $stopwords"))
        new(textconfig, local_weighting, global_weighting, Int(min_ndocs), Int(min_occs),
            stopwords === nothing ? nothing : Float64(stopwords))
    end
end

Base.show(io::IO, s::BaseProfile) = print(io, "BaseProfile(", s.profile.model.voc |> vocsize, " tokens)")

function Base.show(io::IO, s::FitFromCorpus)
    print(io, "FitFromCorpus(", s.textconfig.language)
    print(io, ", ", nameof(typeof(s.local_weighting)), "/", nameof(typeof(s.global_weighting)))
    s.min_ndocs == 1 || print(io, ", min_ndocs=", s.min_ndocs)
    s.min_occs == 1 || print(io, ", min_occs=", s.min_occs)
    s.stopwords === nothing || print(io, ", stopwords=", s.stopwords)
    print(io, ")")
end

"""
    is_text_index_type(::Type) -> Bool

Whether an `index_type` names one of the text index kinds -- the ones that take a
[`AbstractTextModelSpec`](@ref) and reject nothing else. `InvertedFile` and
`TextInvertedFile` both name the weighted engine (see [`InvertedFileEngine`](@ref)).
"""
is_text_index_type(::Type{BM25InvertedFile}) = true
is_text_index_type(::Type{InvertedFile}) = true
is_text_index_type(::Type{TextInvertedFile}) = true
is_text_index_type(::Type) = false

# The two halves a spec resolves into: a profile to start trained from, and a recipe for the
# deferred fit. Exactly one of them is ever non-`nothing` at creation, which is what makes the
# choice unambiguous downstream -- `index!` fits if and only if it finds no profile.
_initial_profile(spec::BaseProfile) = spec.profile
_initial_profile(::FitFromCorpus) = nothing
_deferred_fit(::BaseProfile) = nothing
_deferred_fit(spec::FitFromCorpus) = spec

function _require_textmodel(IndexType::Type, textmodel)
    textmodel isa AbstractTextModelSpec && return textmodel
    textmodel === nothing && error("""
        a text project ($(nameof(IndexType))) needs an explicit `textmodel`, because the two ways to get \
        a vocabulary are not interchangeable and one of them cannot be undone later:
          textmodel=BaseProfile(load_profile("wiki20231101-es.zip"))  -- index against a model fitted \
        elsewhere; the vocabulary covers the language, so terms a later batch introduces stay searchable
          textmodel=FitFromCorpus(TextConfig())  -- fit one from this project's own corpus at the first \
        index! call, which freezes the vocabulary there: every term appended afterwards that it never saw \
        is out-of-vocabulary and silently dropped""")
    error("textmodel must be a BaseProfile or a FitFromCorpus; got $(typeof(textmodel))")
end

function _reject_textmodel(IndexType::Type, textmodel)
    textmodel === nothing && return nothing
    error("`textmodel` only applies to a text index kind (BM25InvertedFile/TextInvertedFile/InvertedFile); " *
          "$(nameof(IndexType)) is a dense index, with no text to tokenize and no vocabulary to fit")
end

"""
    validate_textmodel(index_type::Type, textmodel) -> Union{Nothing, AbstractTextModelSpec}

Checks `textmodel` against `index_type` -- required for a text index kind, refused for a dense
one -- and hands back the spec (or `nothing` for a dense kind). Raises the same errors
[`create_engine`](@ref) would.

Split out so a caller can run the check *before* committing to anything: `create_project`
opens the project's RocksDB directory before it ever reaches `create_engine`, so letting the
error surface there would leave a created directory and a held write lock behind on a call
that failed. `create_engine` still checks for itself, for callers that reach it directly.
"""
validate_textmodel(IndexType::Type, textmodel) =
    is_text_index_type(IndexType) ? _require_textmodel(IndexType, textmodel) :
                                    _reject_textmodel(IndexType, textmodel)

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
(`bagofwords`/`bm25score`). Created from a [`BaseProfile`](@ref) -- trained immediately --
or from a [`FitFromCorpus`](@ref), in which case it stays untrained (`index === nothing`,
`profile === nothing`) until the first [`index!`](@ref index!(::BM25Engine)) call fits a
profile from what's staged; a `BM25InvertedFile` needs a `Vocabulary` to be constructed at
all. Exactly like `SearchGraphEngine`, `add_item!`/`append_items!` only ever *stage* raw text
into `staged`; [`index!`](@ref index!(::BM25Engine)) is the explicit step that trains (when
it has to) and encodes/indexes the backlog.

# Fields
- `index::Union{Nothing, BM25InvertedFile}`
- `profile::Union{Nothing, TextProfile}`: this engine's whole text model -- vocabulary and
  weights (`profile.model`) plus the corpus-produced artifacts (stopword set, lemma map,
  query-expansion network) and the lineage saying how it was produced. One field replaces
  the `voc`/`model` pair the pre-`TextSearch` 1.1 shape carried, which is the point of the
  type: an artifact has exactly one home, and the `TextConfig` the tokenizer runs
  (`gettextconfig(profile)`) is *derived* from it, so an index cannot end up tokenizing
  documents through a different lemma map than the one it saves. `index === nothing` if and
  only if this is `nothing`.
- `fitspec::Union{Nothing, FitFromCorpus}`: the recipe for the deferred fit (policy,
  weighting, pruning, stopword threshold) -- consulted only by the [`index!`](@ref
  index!(::BM25Engine)) call that has to fit `profile`, and never again once one exists;
  `gettextconfig(profile)` is the authority from that point on, since it additionally carries
  whichever artifacts the profile applies. `nothing` for an engine created from a
  [`BaseProfile`](@ref), which has no fit to defer. Kept after the fit rather than cleared:
  it is what the profile was made from, and a reopened project should be able to say so
  without re-deriving it from the lineage.
- `variants::Union{Nothing, Dict{String,Vector{String}}}`: the orthographic variant map
  [`resolve_query`](@ref) bridges a query with, derived from the vocabulary the moment
  `profile` is set rather than per query -- it is a pure function of that (frozen)
  vocabulary, and deriving it costs ~0.24s over a half-million-token vocabulary, which is
  not a per-search cost worth paying. Deriving it up front also keeps [`search_live`](@ref)
  free of any write to the engine, so it stays safe under a plain read lock. `nothing`
  exactly while `profile` is.
- `staged::Vector{String}`: every raw text ever staged via `add_item!`/`append_items!`, in
  insertion order, whether or not it has been encoded into `index` yet -- the text-engine
  counterpart of `SearchGraphEngine.index.db`. `index!` catches up the range
  `(engine.index === nothing ? 0 : length(engine.index))+1:length(staged)`.
- `ctx::InvertedFileContext`: built eagerly at `create_engine` time -- `InvertedFileContext`
  needs no vocabulary/index to exist, so there's no reason to leave this `nothing` until
  [`index!`](@ref index!(::BM25Engine)) the way `index`/`profile` must be. Reserved for
  insertion; never shared with a concurrent [`search_live`](@ref) call, which gets its own
  from `search_ctx_pool` instead.
- `search_ctx_pool::ContextPool`: one private context per concurrent
  [`search_live`](@ref) call (see [`ContextPool`](@ref)). Left unparameterized since
  `InvertedFileContext` itself is parametric (`InvertedFileContext{A,B}`).
- `deleted_ids::Set{UInt32}`: `UInt32` across every engine kind (see [`AbstractSearchEngine`](@ref)).
- `lock::ReadWriteLock`
"""
mutable struct BM25Engine <: AbstractSearchEngine
    index::Union{Nothing, BM25InvertedFile}
    profile::Union{Nothing, TextProfile}
    fitspec::Union{Nothing, FitFromCorpus}
    variants::Union{Nothing, Dict{String,Vector{String}}}
    staged::Vector{String}
    ctx::InvertedFileContext
    search_ctx_pool::ContextPool
    deleted_ids::Set{UInt32}
    lock::ReadWriteLock
end

"""
    InvertedFileEngine

Text engine backed by a `TextSearch.TextInvertedFile` -- a `VectorModel` paired with a
`SimilaritySearch.InvertedFile`, so text is vectorized into `SparseVector`s on the way in
and on the way out by the library rather than by this package, and it can be built against
any `PreMetric` `distance`, not just the cosine default. It replaces the hand-rolled
`InvertedFile` + separate `model` field this engine carried before `TextSearch` 1.1
published the pairing as a type; the two are the same construction, and letting the library
own it is what keeps document and query vectorization from drifting apart.

Created from a [`BaseProfile`](@ref) -- trained immediately -- or from a
[`FitFromCorpus`](@ref), in which case it stays untrained (`index === nothing`,
`profile === nothing`) until the first [`index!`](@ref index!(::InvertedFileEngine)) call fits
a profile from what's staged. Exactly like `SearchGraphEngine`, `add_item!`/`append_items!`
only ever *stage* raw text into `staged`.

# Fields
- `index::Union{Nothing, TextInvertedFile}`
- `profile::Union{Nothing, TextProfile}`: as on [`BM25Engine`](@ref) -- vocabulary,
  weights, artifacts and lineage in one value. `index.model` is this profile's own
  `VectorModel`, not a second copy of it.
- `fitspec::Union{Nothing, FitFromCorpus}`: as on [`BM25Engine`](@ref) -- the recipe for the
  deferred fit, unused once `profile` exists, `nothing` when there was never one to defer.
- `variants::Union{Nothing, Dict{String,Vector{String}}}`: as on [`BM25Engine`](@ref).
- `distance::SimilaritySearch.PreMetric`: the distance chosen at `create_engine` time --
  remembered here since the real index can't be built until a vocabulary exists to size
  its posting-list array.
- `staged::Vector{String}`: every raw text ever staged via `add_item!`/`append_items!`, in
  insertion order, whether or not it has been encoded into `index` yet -- the text-engine
  counterpart of `SearchGraphEngine.index.db`. `index!` catches up the range
  `(engine.index === nothing ? 0 : length(engine.index))+1:length(staged)`.
- `ctx::InvertedFileContext`: built eagerly at `create_engine` time -- `InvertedFileContext`
  needs no vocabulary/index to exist, so there's no reason to leave this `nothing` until
  [`index!`](@ref index!(::InvertedFileEngine)) the way `index`/`profile` must be. Reserved
  for insertion; never shared with a concurrent [`search_live`](@ref) call, which gets its
  own from `search_ctx_pool` instead.
- `search_ctx_pool::ContextPool`: one private context per concurrent
  [`search_live`](@ref) call (see [`ContextPool`](@ref)). Left unparameterized since
  `InvertedFileContext` itself is parametric (`InvertedFileContext{A,B}`).
- `deleted_ids::Set{UInt32}`: `UInt32` across every engine kind (see [`AbstractSearchEngine`](@ref))
  -- matches `InvertedFile`'s own posting lists (`AdjList{UInt32}`), which key ids by
  `UInt32` too.
- `lock::ReadWriteLock`
"""
mutable struct InvertedFileEngine <: AbstractSearchEngine
    index::Union{Nothing, TextInvertedFile}
    profile::Union{Nothing, TextProfile}
    fitspec::Union{Nothing, FitFromCorpus}
    variants::Union{Nothing, Dict{String,Vector{String}}}
    distance::SimilaritySearch.PreMetric
    staged::Vector{String}
    ctx::InvertedFileContext
    search_ctx_pool::ContextPool
    deleted_ids::Set{UInt32}
    lock::ReadWriteLock
end

"""
    CallbackLog(flush::Function)

An `AbstractLog` backend (see `SimilaritySearch.jl`'s `log.jl`) that calls
`flush(index, sp, ep)` -- instead of printing, the way `InformativeLog` does -- on every
`:add!` event a `push_item!`/`append_items!` call reports, forwarding exactly the range
`sp:ep` the library itself reports, with no batching/throttling of its own. Knows nothing
about RocksDB or any other storage backend; `flush` is supplied by the caller.

`SimilaritySearch.jl`'s `log.jl` standardized every index kind's `LOG` calls onto two
canonical events regardless of which entry point (`push_item!`, `append_items!`, `index!`,
...) triggered them: `:add!` for a real, `sp:ep`-scoped structural mutation, and `:info`
for a no-op (only `ExhaustiveSearch`/`ParallelExhaustiveSearch`'s `index!` fires this,
since their database already *is* the index). This package's `IndexEngine.index!(engine::SearchGraphEngine)`
does call `SimilaritySearch.index!` directly (unlike `push_item!`/`append_items!`, which
only ever stage) -- but a genuine no-op call (nothing new staged since the last one) fires
no `LOG` at all, since `SimilaritySearch.index!`'s own backlog loop (`length(index)+1:n`)
is simply empty in that case, not a `:info` call to filter out. `BM25Engine`/
`InvertedFileEngine`'s own `index!` methods have the same staged-vs-indexed split (see
their docstrings) but no `SimilaritySearch.index!` of their own to delegate to -- they run
their backlog loop by calling `push_item!` once per staged-but-not-yet-indexed item
directly, which fires `LOG(:add!, ...)` per item exactly as an eager `add_item!` used to.
Either way `CallbackLog` never has to branch on `event` itself: every call it does receive
here is a real mutation.

!!! warning "what `index` looks like inside `flush`, for a `SearchGraph`"
    `LOG`'s `:add!` event for a `SearchGraph` fires *before* `connect_reverse_links!` runs
    for `sp:ep` (`searchgraph/insertions.jl`) -- so `index.adj` for that exact range holds only the
    *direct* links just computed, none of the reverse links other nodes will later add
    into it. This is by design, not a bug to route around: [`direct_neighbors`](@ref)
    captures exactly that direct-links-only slice, and [`build_searchgraph`](@ref)
    reconnects every reverse link *once*, only after every staged vector and graph-indexed
    object's adjacency entry has been replayed -- reconnecting on an already-complete
    graph isn't safe (it isn't idempotent; it would duplicate reverse edges), which is
    exactly why this format never saves reverse links in the first place.
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
    direct_neighbors(index::SearchGraph, i::Integer) -> Vector{UInt32}

Object `i`'s direct (not yet reverse-connected) neighbor list, read at exactly the
moment `LOG` reports the range containing `i` -- i.e. before `connect_reverse_links!` has
added anything else into it (see [`CallbackLog`](@ref)'s warning).
"""
direct_neighbors(index::SearchGraph, i::Integer) = copy(neighbors(index.adj, i))

"""
    apply_searchgraph_vectors!(index::SearchGraph, block)

Replays one vector block (as returned by `Persistence.load_dense_vector_blocks`, shaped
`(sp, ep, vectors)`) into `index`: appends its vectors to `index.db`. Does *not* touch
`index.adj`/`index.len[]` -- [`build_searchgraph`](@ref) handles adjacency (via a separate
per-object lookup) and finalizing length itself, once, after every vectors block has been
applied.
"""
function apply_searchgraph_vectors!(index::SearchGraph, block)
    for v in block.vectors
        push_item!(index.db, v)
    end
end

"""
    build_searchgraph(distance, vector_blocks, load_neighbors::Function, graph_len::Integer) -> SearchGraph

Rebuilds a `SearchGraph` against `distance`: replays each of `vector_blocks` (as returned
by `Persistence.load_dense_vector_blocks` from the project's `MMapMatrixDatabase` file) in
order via [`apply_searchgraph_vectors!`](@ref) to reconstruct `index.db` in full -- *every*
vector ever staged, whether or not it was ever actually graph-indexed (see [`index!`](@ref
index!(::SearchGraphEngine))'s stage-then-index split: `append_items!`/`add_item!` only
ever stage into `.db`, never graph-connect by themselves anymore).

`graph_len` (persisted separately, see `Persistence`'s `:graph_len` engine field) is the
count that actually matters for the *graph* structure: it can be `<= length(index.db)` if
the process closed (or crashed) after staging some vectors but before an explicit
[`index!`](@ref index!(::SearchGraphEngine)) call caught them up. Only object ids `1:graph_len`
get their saved direct neighbor list restored (`load_neighbors(i)`, typically
`i -> Persistence.load_neighbors(adjacency_store, i)`) and reverse-connected -- restoring
`index.len[]` to `graph_len`, not to the full staged count, so a later explicit `index!`
call picks up exactly where indexing left off, over the *same* `.db` (already fully
restored here) it would have seen pre-restart.
"""
function build_searchgraph(distance, vector_blocks, load_neighbors::Function, graph_len::Integer)
    index = SearchGraph(distance, VectorDatabase())
    for block in vector_blocks
        apply_searchgraph_vectors!(index, block)
    end
    for i in 1:graph_len
        add!(index.adj, i, load_neighbors(i))
    end
    index.len[] = graph_len
    graph_len > 0 && SimilaritySearch.connect_reverse_links!(index.adj, 1, graph_len)
    return index
end

"""
    invertedfile_objects(index::AbstractInvertedFile, sp::Integer, ep::Integer) -> Vector

Range `sp:ep`'s raw indexed objects (`index.db` -- bags-of-words for a `BM25InvertedFile`,
`SparseVector`s for an `InvertedFile`), read at the moment `LOG` reports this range.
Unlike [`direct_neighbors`](@ref)'s `SearchGraph` counterpart, there's no
before/after-`connect_reverse_links!` timing to worry about here -- `push_item!`/
`append_items!` for an inverted file fully finalize an object's contribution to the
posting lists *before* `LOG` fires at all (see [`CallbackLog`](@ref)'s docstring), so this
can be read at any point after the report, not just at exactly that moment.

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
    text_profile(engine::AbstractSearchEngine) -> Union{Nothing, TextProfile}
    text_vocabulary(engine::AbstractSearchEngine) -> Union{Nothing, TextSearch.Vocabulary}

This engine's text model, or `nothing` for a dense engine kind (and for a text engine that
has not been trained yet -- see [`BM25Engine`](@ref)). `text_vocabulary` is the shortcut for
the vocabulary inside it, which is what `bagofwords`/`token2id`/[`resolve_query`](@ref) all
work against.
"""
text_profile(engine::Union{BM25Engine, InvertedFileEngine}) = engine.profile
text_profile(::AbstractSearchEngine) = nothing
text_vocabulary(engine::AbstractSearchEngine) = (p = text_profile(engine); p === nothing ? nothing : p.model.voc)

# Drops tokens below `spec`'s frequency floors. Returns `voc` untouched when both floors are
# 1, so the no-pruning default costs nothing (`filter_tokens` rebuilds the whole vocabulary).
# `filter_tokens` carries `trainsize`/`numtokens` across unchanged, which is what BM25's
# average document length is computed from -- the pruned tokens still occurred in those
# documents, so those totals should not shrink with the vocabulary.
function _prune_vocabulary(voc::Vocabulary, spec::FitFromCorpus)
    (spec.min_ndocs <= 1 && spec.min_occs <= 1) && return voc
    filter_tokens(voc) do t
        t.ndocs >= spec.min_ndocs && t.occs >= spec.min_occs
    end
end

"""
    fit_profile(spec::FitFromCorpus, corpus; source="corpus") -> TextProfile

Fits a `TextSearch.TextProfile` over `corpus` as [`FitFromCorpus`](@ref) `spec` describes it:
a `Vocabulary` counted under `spec.textconfig` and pruned to its frequency floors, optional
stopword detection, a `VectorModel` under its weighting scheme, and a `:fit` lineage step
recording the corpus size and where it came from (`source` -- `"staged"` when the fit was the
deferred one an [`index!`](@ref index!(::BM25Engine)) call ran over a project's own staged
text, which is worth telling apart from a fit a caller ran deliberately over a corpus it
chose).

Stopword detection is what makes this two passes rather than one: the flags come from
document frequencies counted under the policy, and dropping the flagged tokens changes every
count the model is built from, so the vocabulary has to be recounted under the pipeline that
excludes them. `spec.stopwords === nothing` (the default) skips the second pass entirely.

Beyond stopwords the profile carries no artifacts of its own -- a lemma map needs word
embeddings and an expansion network needs an LSI, neither of which an indexing corpus yields,
which is what `TextSearch.jl`'s own `textsearch fit` pipeline exists to do and what a
[`BaseProfile`](@ref) brings instead. What this *does* preserve is any artifact the caller
already put in `spec.textconfig.pipeline`: those are lifted out into the profile's own fields
with `applied` set to match, so that `TextProfile`'s constructor rematerializes the identical
`TextConfig` rather than the bare policy. Skipping that step would silently index documents
through a lemma map the profile then claims not to have -- exactly the
saved-copy-versus-applied-copy drift the type was introduced to make impossible.
"""
function fit_profile(spec::FitFromCorpus, corpus; source::AbstractString="corpus")
    textconfig = spec.textconfig
    voc = _prune_vocabulary(Vocabulary(textconfig, corpus), spec)

    detected = 0
    if spec.stopwords !== nothing
        flagged = stopword_candidates(voc, spec.stopwords)
        if !isempty(flagged)
            detected = length(flagged)
            carried = textconfig.pipeline.stopwords
            stops = carried === nothing ? Set{String}(flagged) : union(carried, flagged)
            textconfig = TextConfig(textconfig;
                pipeline=TokenPipeline(lemmas=textconfig.pipeline.lemmas, stopwords=stops))
            voc = _prune_vocabulary(Vocabulary(textconfig, corpus), spec)
        end
    end

    model = VectorModel(spec.global_weighting, spec.local_weighting, voc)
    pipeline = textconfig.pipeline
    stopwords = pipeline.stopwords === nothing ? Set{String}() : pipeline.stopwords
    lemmas = pipeline.lemmas === nothing ? Dict{String,String}() : pipeline.lemmas
    applied = AppliedArtifacts(; stopwords=pipeline.stopwords !== nothing,
                                 lemmas=pipeline.lemmas !== nothing)
    return TextProfile(model; stopwords, lemmas, applied,
                       lineage=[LineageStep(:fit;
                                            trainsize=length(corpus),
                                            source=String(source),
                                            local_weighting=String(string(nameof(typeof(spec.local_weighting)))),
                                            global_weighting=String(string(nameof(typeof(spec.global_weighting)))),
                                            min_ndocs=spec.min_ndocs,
                                            min_occs=spec.min_occs,
                                            stopwords_detected=detected)])
end

# The variant map cached on a text engine (see `BM25Engine`'s `variants` field): derived once,
# the instant a profile becomes known, never per query. `derive_variants` returns an empty map
# immediately for any policy that already folds both case and diacritics -- the default
# `TextConfig()` among them -- so this is free unless a profile deliberately preserves them.
_derive_variants(profile::Nothing) = nothing
_derive_variants(profile::TextProfile) = derive_variants(profile.model.voc)

# `query_expansion=nothing` on the index deliberately: the network is a per-query choice
# (`QueryPolicy`'s `expansion`/`expansion_k`), and a network baked into the index would be
# all-or-nothing for every search against it. `search_live` applies the profile's own network
# per call instead. The `max(..., 1)` guard covers an empty vocabulary, which `InvertedFile`
# cannot be sized zero for.
_new_textinvertedfile(profile::TextProfile, distance) =
    TextInvertedFile(profile.model, InvertedFile(max(vocsize(profile.model.voc), 1), distance), nothing)

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
    build_textinvertedfile(distance, profile::TextProfile, object_blocks) -> TextInvertedFile

Rebuilds a `TextInvertedFile` against `distance` and `profile`'s `VectorModel` by replaying
every saved raw object through the library's own `push_item!`, in order -- see
[`build_bm25invertedfile`](@ref) (same rebuild-by-reinsertion approach and scaling caveat).

The saved objects are already-vectorized `SparseVector`s, not text, so they take
`TextInvertedFile`'s generic `push_item!` (which forwards straight to the wrapped
`InvertedFile`) rather than its vectorizing `AbstractString`/`TokenizedText` overload --
replaying them must not re-run a vectorization that already happened, and would not be able
to anyway.
"""
function build_textinvertedfile(distance, profile::TextProfile, object_blocks)
    index = _new_textinvertedfile(profile, distance)
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
(and ignored) on the other two, so a caller can pass the same keyword set uniformly
regardless of index type. A `textmodel`, by contrast, is *rejected* rather than ignored on all
three: a text model handed to an index of vectors is a caller mistake with no reading under
which it does anything, and swallowing it silently is how a project ends up not being the kind
its author thought it was. `on_change::Union{Nothing,Function}`, if given, is installed
(via [`CallbackLog`](@ref)) as an `(index, sp, ep) -> nothing` callback fired on every
`:add!` event a `push_item!`/`append_items!` call reports -- e.g. to persist the range
`sp:ep` that was just inserted. `log_io::Union{Nothing,IO}`, if given, additionally prints the same
throttled informative status line [`InformativeLog`](@ref) already prints to `stderr`
to this `IO` too (via [`FileLog`](@ref)) -- an open file handle or `stdout`/`stderr`
both work; purely informative, changes nothing about what gets persisted.
"""
function create_engine(::Type{SearchGraph}; distance=SimilaritySearch.Dist.SqL2(), minrecall::Union{Nothing,Real}=0.9, textmodel=nothing, on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing)
    _reject_textmodel(SearchGraph, textmodel)
    mr = minrecall === nothing ? nothing : Float32(minrecall)
    return SearchGraphEngine(SearchGraph(distance, VectorDatabase()), _searchgraph_context(mr, _engine_logger(on_change, log_io)), mr, OptBeamSearch(), ContextPool(SearchGraphContext()), Set{UInt32}(), ReadWriteLock())
end
function create_engine(::Type{ExhaustiveSearch}; distance=SimilaritySearch.Dist.SqL2(), minrecall=nothing, textmodel=nothing, on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing)
    _reject_textmodel(ExhaustiveSearch, textmodel)
    GenericEngine{ExhaustiveSearch}(ExhaustiveSearch(distance, VectorDatabase()), GenericContext(; logger=_engine_logger(on_change, log_io)), ContextPool(GenericContext()), Set{UInt32}(), ReadWriteLock())
end
function create_engine(::Type{ParallelExhaustiveSearch}; distance=SimilaritySearch.Dist.SqL2(), minrecall=nothing, textmodel=nothing, on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing)
    _reject_textmodel(ParallelExhaustiveSearch, textmodel)
    GenericEngine{ParallelExhaustiveSearch}(ParallelExhaustiveSearch(distance, VectorDatabase()), GenericContext(; logger=_engine_logger(on_change, log_io)), ContextPool(GenericContext()), Set{UInt32}(), ReadWriteLock())
end

"""
    create_engine(::Type{BM25InvertedFile}; textmodel, distance=nothing, minrecall=nothing, on_change=nothing, log_io=nothing) -> BM25Engine
    create_engine(::Type{InvertedFile}; textmodel, distance=SimilaritySearch.Dist.NormCosine(), minrecall=nothing, on_change=nothing, log_io=nothing) -> InvertedFileEngine
    create_engine(::Type{TextInvertedFile}; ...) -> InvertedFileEngine

Creates a new, empty text search engine of the given index type. `minrecall` is accepted and
ignored on both, and `distance` is accepted and ignored on `BM25InvertedFile` (BM25 always
scores via its own `bm25score`), so a caller can pass the same keyword set uniformly
regardless of index type. `on_change`/`log_io` are as in the dense `create_engine` methods
above. `TextInvertedFile` and `InvertedFile` select the same engine and are interchangeable
here; `TextInvertedFile` is the name of what actually gets built (see
[`InvertedFileEngine`](@ref)).

`textmodel::`[`AbstractTextModelSpec`](@ref) is **required** -- there is no default:

- [`BaseProfile`](@ref)`(profile)` indexes against a model fitted elsewhere. The engine is
  trained from this moment: the real index is built immediately, nothing is inferred from the
  data appended later, and the vocabulary covers the language rather than just this project's
  first batch.
- [`FitFromCorpus`](@ref)`(textconfig; ...)` fits one from this project's own staged corpus at
  the first [`index!`](@ref index!(::BM25Engine)) call, with no base profile.

Omitting it is an error rather than a default, and that is the point: the second form freezes
the vocabulary at the first `index!` call, so every term a later batch introduces is
out-of-vocabulary and silently dropped from then on. Nothing about a project's behaviour later
reveals that this choice was made by omission, which is exactly why it cannot be.
"""
function create_engine(::Type{BM25InvertedFile}; distance=nothing, minrecall=nothing,
                       textmodel::Union{Nothing,AbstractTextModelSpec}=nothing,
                       on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing)
    spec = _require_textmodel(BM25InvertedFile, textmodel)
    profile = _initial_profile(spec)
    index = profile === nothing ? nothing : BM25InvertedFile(profile.model.voc)
    BM25Engine(index, profile, _deferred_fit(spec), _derive_variants(profile), String[],
               InvertedFileContext(; logger=_engine_logger(on_change, log_io)),
               ContextPool(InvertedFileContext()), Set{UInt32}(), ReadWriteLock())
end

function create_engine(::Type{InvertedFile}; distance=SimilaritySearch.Dist.NormCosine(), minrecall=nothing,
                       textmodel::Union{Nothing,AbstractTextModelSpec}=nothing,
                       on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing)
    spec = _require_textmodel(InvertedFile, textmodel)
    profile = _initial_profile(spec)
    index = profile === nothing ? nothing : _new_textinvertedfile(profile, distance)
    InvertedFileEngine(index, profile, _deferred_fit(spec), _derive_variants(profile), distance, String[],
                       InvertedFileContext(; logger=_engine_logger(on_change, log_io)),
                       ContextPool(InvertedFileContext()), Set{UInt32}(), ReadWriteLock())
end

create_engine(::Type{TextInvertedFile}; kwargs...) = create_engine(InvertedFile; kwargs...)

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
[`direct_neighbors`](@ref)/[`build_searchgraph`](@ref)); the
two text engines include `profile`/`fitspec` (plus `distance` for `InvertedFileEngine`)
instead (see [`invertedfile_objects`](@ref)/[`build_bm25invertedfile`](@ref)/
[`build_textinvertedfile`](@ref)) -- all three need their saved insertion blocks replayed
through a dedicated `build_*` function to get `index` back, not a plain field read. The
text engines save one `profile` where they used to save a `voc`/`model` pair: a
`TextSearch.TextProfile` holds both, along with the artifacts and lineage neither of them
carried, so there is one value to write and no way for the two halves to be saved out of
step with each other.

None of the three includes `staged`/`.db`'s raw items either, for the same reason: those
live in their own dedicated, incrementally-persisted store (a `SearchGraph`'s
`MMapMatrixDatabase` file, a text engine's `Persistence.StagedTextStore`), not a plain
`EngineStore` field -- rewriting the whole (potentially large, ever-growing) staged
sequence into `EngineStore` on every mutation is exactly the "single ever-growing blob"
this design avoids. A caller restoring one of these three assembles `staged`/`vector_blocks`
into the `state` NamedTuple by hand from that dedicated store (see [`restore_engine`](@ref)'s
docstring) rather than getting it from `snapshot_state`.
"""
snapshot_state(engine::SearchGraphEngine) =
    (kind=SearchGraphEngine, distance=engine.index.dist, minrecall=engine.minrecall, opt_beamsearch=engine.opt_beamsearch, deleted_ids=engine.deleted_ids)
snapshot_state(engine::GenericEngine{IndexType}) where {IndexType} =
    (kind=IndexType, index=engine.index, deleted_ids=engine.deleted_ids)
snapshot_state(engine::BM25Engine) =
    (kind=BM25Engine, profile=engine.profile, fitspec=engine.fitspec, deleted_ids=engine.deleted_ids)
snapshot_state(engine::InvertedFileEngine) =
    (kind=InvertedFileEngine, profile=engine.profile, fitspec=engine.fitspec, distance=engine.distance, deleted_ids=engine.deleted_ids)

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
- `SearchGraphEngine`: `distance`, `vector_blocks` (`Persistence.load_dense_vector_blocks(dense_vectors_store)`),
  `load_neighbors` (typically `i -> Persistence.load_neighbors(adjacency_store, i)`),
  `graph_len` (`Persistence.load_field(store, :graph_len, 0)` -- may be `< length` of the
  restored `.db` if some staged vectors were never caught up by an explicit
  [`index!`](@ref index!(::SearchGraphEngine)) call before the project last closed) -- see
  [`build_searchgraph`](@ref).
- `BM25Engine`: `profile`, `fitspec`, `object_blocks`
  (`Persistence.load_object_blocks(obj_store)`, or `nothing`/empty if `profile === nothing`,
  i.e. never trained), `staged` (every raw text ever staged, flattened from
  `Persistence.load_staged_text_blocks`, which can be longer than `object_blocks`'s total
  count if a backlog was still pending an [`index!`](@ref index!(::BM25Engine)) call when the
  project last closed) -- see [`build_bm25invertedfile`](@ref).
- `InvertedFileEngine`: `profile`, `fitspec`, `distance`, `object_blocks`, `staged`
  likewise -- see [`build_textinvertedfile`](@ref).
"""
restore_engine(state; on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing) = restore_engine(state.kind, state; on_change, log_io)

function restore_engine(::Type{SearchGraphEngine}, state; on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing)
    index = build_searchgraph(state.distance, state.vector_blocks, state.load_neighbors, state.graph_len)
    return SearchGraphEngine(index, _searchgraph_context(state.minrecall, _engine_logger(on_change, log_io)), state.minrecall, state.opt_beamsearch, ContextPool(SearchGraphContext()), state.deleted_ids, ReadWriteLock())
end

function restore_engine(::Type{IndexType}, state; on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing) where {IndexType}
    return GenericEngine{IndexType}(state.index, GenericContext(; logger=_engine_logger(on_change, log_io)), ContextPool(GenericContext()), state.deleted_ids, ReadWriteLock())
end

function restore_engine(::Type{BM25Engine}, state; on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing)
    profile = state.profile
    index = profile === nothing ? nothing : build_bm25invertedfile(profile.model.voc, state.object_blocks)
    return BM25Engine(index, profile, state.fitspec, _derive_variants(profile), state.staged, InvertedFileContext(; logger=_engine_logger(on_change, log_io)), ContextPool(InvertedFileContext()), state.deleted_ids, ReadWriteLock())
end

function restore_engine(::Type{InvertedFileEngine}, state; on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing)
    profile = state.profile
    index = profile === nothing ? nothing : build_textinvertedfile(state.distance, profile, state.object_blocks)
    return InvertedFileEngine(index, profile, state.fitspec, _derive_variants(profile), state.distance, state.staged, InvertedFileContext(; logger=_engine_logger(on_change, log_io)), ContextPool(InvertedFileContext()), state.deleted_ids, ReadWriteLock())
end

is_text_index(::AbstractSearchEngine) = false
is_text_index(::Union{BM25Engine, InvertedFileEngine}) = true

"""
    index!(engine::BM25Engine)
    index!(engine::InvertedFileEngine)

Catches up encoding/indexing over whatever's been staged into `engine.staged` since the
last call (or since creation) -- the exact same contract as
[`index!`](@ref index!(::SearchGraphEngine)): idempotent, safe to call any number of
times, only ever processes the backlog (`(engine.index === nothing ? 0 :
length(engine.index))+1:length(engine.staged)`), and a no-op when nothing new is staged.

If this engine was created from a [`FitFromCorpus`](@ref), the first call additionally runs
that fit (see [`fit_profile`](@ref)) over *every* text staged so far (`engine.staged`, not
just what's staged in this particular call) and builds the real index against the resulting
profile -- `TextSearch.jl` needs a `Vocabulary` before either index type can be constructed at
all, and retraining isn't supported, so this only ever happens once per engine, whichever
`index!` call first finds `engine.profile === nothing`. Tokens never seen at that fit are
out-of-vocabulary on every later batch and silently dropped when encoded (a `TextSearch.jl`
limitation, not a bug in this package) -- if `engine.staged` at first-`index!` time isn't
representative of the text you'll keep appending, later recall will suffer.

An engine created from a [`BaseProfile`](@ref) skips all of that: it was trained before the
first item was ever staged, so every call here is a pure catch-up and the vocabulary never
depends on what happened to be appended first. That is the shape to prefer for anything but a
self-contained corpus -- see [`create_engine`](@ref), which makes the choice explicit.

`add_item!`/`append_items!` on a `BM25Engine`/`InvertedFileEngine` only ever stage raw
text into `engine.staged` -- exactly like `SearchGraphEngine`, they do *not* make new
items searchable by themselves; [`search_live`](@ref) only ever sees items this has
processed. `GenericEngine` (`ExhaustiveSearch`/`ParallelExhaustiveSearch`) is the one
engine kind with no such split at all -- it always evaluates directly against `db`, so it
has no `index!` method of its own.

Errors if `engine.staged` is completely empty (nothing has ever been staged) -- mirrors
`index!(engine::SearchGraphEngine)`'s empty-`.db` error.
"""
function index!(engine::BM25Engine)
    write_lock(engine.lock) do
        n = length(engine.staged)
        n == 0 && error("BM25Engine has nothing staged yet -- add_item!/append_items! at least one item before calling index!")
        already = engine.index === nothing ? 0 : length(engine.index)
        if engine.profile === nothing
            profile = fit_profile(engine.fitspec, engine.staged; source="staged")
            engine.profile = profile
            engine.variants = _derive_variants(profile)
            engine.index = BM25InvertedFile(profile.model.voc)
        end
        voc = engine.profile.model.voc
        for i in already+1:n
            push_item!(engine.index, engine.ctx, bagofwords(voc, engine.staged[i]))
        end
    end
    return engine
end

function index!(engine::InvertedFileEngine)
    write_lock(engine.lock) do
        n = length(engine.staged)
        n == 0 && error("InvertedFileEngine has nothing staged yet -- add_item!/append_items! at least one item before calling index!")
        already = engine.index === nothing ? 0 : length(engine.index)
        if engine.profile === nothing
            profile = fit_profile(engine.fitspec, engine.staged; source="staged")
            engine.profile = profile
            engine.variants = _derive_variants(profile)
            engine.index = _new_textinvertedfile(profile, engine.distance)
        end
        for i in already+1:n
            # A raw BOW does not score against a NormCosine (or any other) InvertedFile --
            # it needs a real weighted SparseVector. Handing the text straight to
            # `TextInvertedFile` is what produces one: its `AbstractString` overload of
            # `push_item!` vectorizes through its own `model`, which is this profile's, so
            # documents and queries can only ever be encoded by the same model.
            push_item!(engine.index, engine.ctx, engine.staged[i])
        end
    end
    return engine
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
#
# A SearchGraph deliberately only *stages* here -- pushing straight to `index.db`, never
# calling `SimilaritySearch.index!` (the real, expensive graph-linking step) inline. That
# used to happen on every single `add_item!` (via `append_items!(index, ctx, ...)`, which
# internally does exactly `append_items!(index.db, items); index!(index, ctx)`), forcing
# every insertion to pay graph-construction cost immediately and synchronously. Now
# staging and indexing are two separate, explicit steps -- see [`index!`](@ref
# index!(::SearchGraphEngine)) -- so a caller controls when the expensive step runs (a
# batch job, a cron, before the next round of searches) instead of paying it per item.
# `ExhaustiveSearch`/`ParallelExhaustiveSearch` (exact search, no graph to build) have no
# such split to make -- `push_item!` already *is* the whole insertion, cheap either way.
insert_dense!(index::SearchGraph, ctx, item) = push_item!(index.db, item)
insert_dense!(index::Union{ExhaustiveSearch,ParallelExhaustiveSearch}, ctx, item) = push_item!(index, ctx, item)
insert_dense!(index, ctx, item) = push_item!(index, ctx, item)

"""
    index!(engine::SearchGraphEngine)

Catches up the graph structure over whatever's been staged into `engine.index.db` since
the last call (or since creation) -- extends `SimilaritySearch.index!` with the same
meaning the text engines' [`index!`](@ref index!(::BM25Engine, ::AbstractVector)) methods
give it: "do the expensive part explicitly, now." Unlike those, this is *idempotent* and
can be called any number of times: `SimilaritySearch.index!(idx, ctx)` itself already
starts from `length(idx) + 1` (the current graph-indexed count) up through
`length(database(idx))` (everything staged so far, see [`insert_dense!`](@ref)), so
calling this repeatedly as more vectors accumulate only ever processes the backlog, never
redoing already-indexed work, and calling it with nothing new staged is a cheap no-op.

`add_item!`/`append_items!` on a `SearchGraphEngine` only ever stage raw vectors into
`engine.index.db` -- they do *not* make new items searchable by themselves anymore (unlike
`GenericEngine`/`BM25Engine`/`InvertedFileEngine`, which still index synchronously on every
`add_item!`). Call this explicitly -- interactively, on a schedule, whatever fits the
caller -- to actually build graph connections for the backlog; [`search_live`](@ref) only
ever sees items this has processed. This is also what unlocks batch-shaped insertion:
staging is cheap and safe to do many times in a row without paying graph-construction cost
until the caller actually wants it.

Errors if `engine.index.db` is completely empty (nothing has ever been staged) -- mirrors
`SimilaritySearch.index!`'s own `@assert n > 0`.
"""
function index!(engine::SearchGraphEngine)
    write_lock(engine.lock) do
        n = length(database(engine.index))
        n == 0 && error("SearchGraphEngine has nothing staged yet -- add_item!/append_items! at least one vector before calling index!")
        SimilaritySearch.index!(engine.index, engine.ctx)
    end
    return engine
end

"""
    add_item!(engine::AbstractSearchEngine, item)

Adds a single item to the index. Thread-safe wrapper.
For a `SearchGraphEngine`, this only *stages* `item` (see [`insert_dense!`](@ref)) -- it
does not become searchable until an explicit [`index!`](@ref index!(::SearchGraphEngine))
call. `BM25Engine`/`InvertedFileEngine` have the exact same split: `item` is raw text,
staged into `engine.staged` -- always allowed, whether or not the engine has been trained
yet -- and does not get encoded/indexed until an explicit
[`index!`](@ref index!(::BM25Engine)) call. `GenericEngine`
(`ExhaustiveSearch`/`ParallelExhaustiveSearch`) is the one engine kind that still indexes
synchronously, with no staging step at all.
"""
function add_item!(engine::Union{SearchGraphEngine, GenericEngine}, item)
    write_lock(engine.lock) do
        insert_dense!(engine.index, engine.ctx, item)
    end
end

function add_item!(engine::Union{BM25Engine, InvertedFileEngine}, item)
    write_lock(engine.lock) do
        push!(engine.staged, String(item))
    end
end

"""
    resolve_query(engine, text::AbstractString, policy::QueryPolicy=QueryPolicy()) -> TextSearch.QueryResolution

Runs `text` through this text engine's own `TextConfig` -- the same normalization,
tokenization, lemma and stopword stages every indexed document went through, since both come
from the one `profile` -- and then through the two steps only a query gets: orthographic
correction against the vocabulary, marked on `policy`.

The returned `QueryResolution` is what [`search_live`](@ref) searches with, and it is
returned rather than consumed silently because correcting a query is a substitution the
person who typed it is owed a report of: `TextSearch.explain(r)` renders one line per token
that gained or lost something ("musica appears in only 9 documents, searched as música
instead"), and `QueryPolicy(correction=:off)` is the escape that answers the query as typed.

Errors if `engine` has no profile yet (nothing has been indexed, so there is no vocabulary to
resolve against).
"""
function resolve_query(engine::Union{BM25Engine, InvertedFileEngine}, text::AbstractString, policy::QueryPolicy=QueryPolicy())
    profile = engine.profile
    profile === nothing && error("this text engine has not been trained yet -- index! at least one staged item, or create it with a profile, before resolving a query")
    tokens = collect(tokenize(gettextconfig(profile), text))
    # `variants` is only read when correction is on; passing `nothing` under `:off` keeps the
    # candidate group empty and makes the resolution a pure pass-through of what was typed.
    variants = policy.correction === :off ? nothing : engine.variants
    return resolve_query_tokens(profile.model.voc, tokens, variants, policy)
end

# The tokens to actually encode a query from, rebuilt from `r.resolved` -- one entry per typed
# token *occurrence* -- rather than taken from `r.tokens`.
#
# They differ in exactly one way that matters here: `r.tokens` is a de-duplicated set, which is
# the right answer for the set-intersection matcher `resolve_query_tokens` was written for, and
# the wrong one for an index that weights by term frequency. Reading it directly would silently
# collapse "casa casa casa" to a single occurrence and change every tf/BM25 score, including for
# queries where nothing was corrected at all. Rebuilding per occurrence keeps multiplicity and
# order, so a query nothing bridged encodes bit-for-bit as it did before any of this existed,
# while a bridged occurrence still replaces (`kept == false`) or enriches (`kept == true`) in
# place.
function _search_tokens(r::TextSearch.QueryResolution)
    out = String[]
    for t in r.resolved
        t.kept && push!(out, t.typed)
        for (form, _) in t.added
            push!(out, form)
        end
    end
    return out
end

# Whether `dist` makes `TextInvertedFile` index *bags* rather than weighted vectors. Its
# `push_item!` branches on exactly this (a set distance -- `Dist.Sets.Jaccard()`, `Dice()`,
# `Intersection()`, `CosineSet()` -- scores token membership, so a weighted vector is the wrong
# object to hand it), and a query has to be encoded the same way its documents were or the two
# sides stop being comparable. Mirrored here rather than called through the library's own
# unexported `FullText.is_set_distance`, and testing the same thing it does: which module the
# distance comes from.
_is_set_distance(dist) = parentmodule(typeof(dist)) === SimilaritySearch.Dist.Sets

# The slice of `profile`'s query-expansion network this particular query should be widened
# with, or `nothing` for "do not expand".
#
# Restricted to `expansion_sources(r)` -- one spelling per typed token, its group's commonest --
# and not to every token searched: bridging deliberately reaches spellings the corpus barely
# holds, and their neighbour lists come from a handful of documents, so expanding over the whole
# bridged set mixes senses (`TextSearch.jl` measured `musica` -> `libreto Puccini Verdi` against
# `música`'s actual topic). Trimming to `policy.expansion_k` here, rather than at expansion time,
# keeps the distances aligned with the neighbours they belong to.
function _query_expansion_network(profile::TextProfile, r::TextSearch.QueryResolution, policy::QueryPolicy)
    (policy.expansion && profile.applied.query_expansion && !isempty(profile.query_expansion)) || return (nothing, nothing)
    alldists = profile.query_expansion_distances
    net = Dict{String,Vector{String}}()
    dists = alldists === nothing ? nothing : Dict{String,Vector{Float32}}()
    for src in expansion_sources(r)
        neighbors = get(profile.query_expansion, src, nothing)
        neighbors === nothing && continue
        k = policy.expansion_k > 0 ? min(policy.expansion_k, length(neighbors)) : length(neighbors)
        net[src] = neighbors[1:k]
        if dists !== nothing
            d = get(alldists, src, nothing)
            d === nothing || (dists[src] = d[1:min(k, length(d))])
        end
    end
    return isempty(net) ? (nothing, nothing) : (net, dists)
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
- `policy::QueryPolicy`: how to treat a *text* query -- whether to correct its spelling
  against the vocabulary and whether to widen it with the profile's expansion network (see
  [`resolve_query`](@ref)). Only meaningful for `BM25Engine`/`InvertedFileEngine`; accepted
  and ignored on every dense engine kind, whose queries are vectors with nothing to resolve.
  The default `QueryPolicy()` is inert for a profile fitted under the default `TextConfig()`:
  that policy already folds case and diacritics, so there are no orthographic variants left
  to bridge, and a locally-fitted profile carries no expansion network to widen with.

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
    `add_item!`/`index!`/`mark_deleted!`/`calibrate!` (each a [`write_lock`](@ref))
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
function search_live(engine::SearchGraphEngine, query, k::Int; bs_override=nothing, minrecall=nothing, policy=nothing)
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
            return _collect_live(engine, res)
        finally
            checkin!(engine.search_ctx_pool, ctx)
        end
    end
end

function search_live(engine::GenericEngine, query, k::Int; bs_override=nothing, minrecall=nothing, policy=nothing)
    read_lock(engine.lock) do
        ctx = checkout!(engine.search_ctx_pool)
        try
            res = knnqueue(KnnSorted, max(k, 1))
            search(engine.index, ctx, query, res)
            return _collect_live(engine, res)
        finally
            checkin!(engine.search_ctx_pool, ctx)
        end
    end
end

function search_live(engine::BM25Engine, query, k::Int; bs_override=nothing, minrecall=nothing, policy::QueryPolicy=QueryPolicy())
    read_lock(engine.lock) do
        profile = engine.profile
        profile === nothing && return (id=Int32[], dist=Float32[], deleted=Bool[])
        voc = profile.model.voc
        r = resolve_query(engine, query, policy)
        bow = bagofwords(voc, TokenizedText(_search_tokens(r)))
        net, _ = _query_expansion_network(profile, r, policy)
        net === nothing || expand_query!(bow, voc, net)
        ctx = checkout!(engine.search_ctx_pool)
        try
            res = knnqueue(KnnSorted, max(k, 1))
            search(engine.index, ctx, bow, res)
            return _collect_live(engine, res)
        finally
            checkin!(engine.search_ctx_pool, ctx)
        end
    end
end

function search_live(engine::InvertedFileEngine, query, k::Int; bs_override=nothing, minrecall=nothing, policy::QueryPolicy=QueryPolicy())
    read_lock(engine.lock) do
        profile = engine.profile
        profile === nothing && return (id=Int32[], dist=Float32[], deleted=Bool[])
        voc = profile.model.voc
        r = resolve_query(engine, query, policy)
        tokens = TokenizedText(_search_tokens(r))
        net, dists = _query_expansion_network(profile, r, policy)
        q = if _is_set_distance(engine.distance)
            bow = bagofwords(voc, tokens)
            net === nothing || expand_query!(bow, voc, net)
            bow
        else
            # Vectorized unnormalized on purpose when there is a network: `expand_query!` adds
            # weight to the vector and normalizes at the end, so normalizing first would scale
            # the typed terms against a norm the expansion then invalidates.
            v = vectorize(profile.model, tokens; normalize=(net === nothing))
            net === nothing ? v : expand_query!(v, voc, net; distances=dists)
        end
        ctx = checkout!(engine.search_ctx_pool)
        try
            res = knnqueue(KnnSorted, max(k, 1))
            search(engine.index, ctx, q, res)
            return _collect_live(engine, res)
        finally
            checkin!(engine.search_ctx_pool, ctx)
        end
    end
end

function _collect_live(engine::AbstractSearchEngine, res)
    ids = view(res.ids, res.sp:res.ep)
    dists = view(res.dists, res.sp:res.ep)
    deleted = [id in engine.deleted_ids for id in ids]
    return (id=ids, dist=dists, deleted=deleted)
end

end # module
