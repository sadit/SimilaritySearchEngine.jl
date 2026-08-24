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
export stored_payload
export calibrate!, current_beamsearch, OptBeamSearch, DEFAULT_MINRECALL_LEVELS
export direct_neighbors, apply_searchgraph_vectors!, build_searchgraph
export invertedfile_objects, build_bm25invertedfile, build_textinvertedfile
export text_profile, text_vocabulary, resolve_query, query_pipeline
export AbstractTextModelSpec, BaseProfile, FitFromCorpus, DefaultProfile
export DEFAULT_PROFILE_NICKNAMES, default_profile_path, train_profile
export is_text_index_type, validate_textmodel

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

The text-model decision a text project is created with, as a value. Three answers to "where
does this project's vocabulary come from", in the order worth preferring them:

1. [`DefaultProfile`](@ref)`(:es)` -- the published profile for a language, refitted to this
   project's own corpus at the first [`index!`](@ref index!(::BM25Engine)) call: weights
   calibrated over millions of paragraphs, plus a stopword set, lemma map and expansion network
   inherited rather than derived. Read its docstring for what the refit narrows and what
   `refit=false` keeps -- the two answer different needs.
2. [`BaseProfile`](@ref)`(profile)` -- a profile the caller already has, used as it is.
3. [`FitFromCorpus`](@ref)`(textconfig)` -- no base profile: fit one here, from this project's
   own staged corpus.

There is no default and no `nothing`, because the third is a trap worth being made to choose.
Fitting from the project's own corpus is delegated to `TextSearch.fit_profile`, which runs the
full pipeline -- LSI, expansion network, lemma clustering -- so it is bounded by a sample cap
rather than by the corpus, and its vocabulary is correspondingly thin. Every term a later batch
introduces that the sample never held is out-of-vocabulary and silently dropped, forever.
Nothing about the project's behaviour afterwards distinguishes "I chose this" from "I never
said", which is exactly why it cannot be arrived at by omission.

**Prefer a profile you did not fit here.** That is the whole shape of this hierarchy: the two
good answers both start from a model fitted over more text than this project has -- which is
where calibrated weights, a stopword set, a lemma map and an expansion network can come from at
all -- and the third exists for the case where no such model is to be had.

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
                  min_ndocs=1, max_documents=1000, stopwords=nothing,
                  encoder=NamedTuple(), expansion=NamedTuple(), lemmas=NamedTuple())

Fit this project's text model from its own staged corpus, under `textconfig` -- the
corpus-independent policy (normalization, tokenization, `language`), with no base profile.
This is the explicit form of "I have no pre-fitted model", and the least good of the three
choices: see [`AbstractTextModelSpec`](@ref) for what it costs and [`DefaultProfile`](@ref)
for what to reach for instead.

The fit is deferred to the first [`index!`](@ref index!(::BM25Engine)) call, and delegated
whole to `TextSearch.fit_profile` -- this package does not fit profiles, it asks the library
to. That means every fit runs the full pipeline: an LSI over the sample, a query-expansion
network from it, and a lemma map clustered from the same embeddings. There is no cheap variant,
and the numbers are not small: measured on 3,000 paragraphs, that pipeline takes 48.8s where a
bare vocabulary-and-weights pass takes 0.2s.

# `max_documents` is why this is affordable at all

The fit reads at most `max_documents` of the staged corpus (1,000 by default), sampled by an
even stride across it rather than taken from the front -- the front is whatever batch arrived
first, one book or one import, and a stride spans everything staged. Without the cap the cost
grows with the corpus: the same pipeline over 300,000 paragraphs took 863s.

The cap is a real trade, not a free win, and it cuts the wrong way from the one thing this path
is worst at. A vocabulary fitted from 1,000 sampled documents is smaller than one fitted from
the whole corpus, so *more* of what gets appended later is out-of-vocabulary and silently
dropped. `max_documents=0` lifts the cap for a caller who would rather pay; the honest fix is
not to fit here at all.

# Options

`min_ndocs`, `stopwords`, `encoder`, `expansion` and `lemmas` are passed through to
`TextSearch.fit_profile`; see it for the full set each accepts.

- `min_ndocs`: drop a token seen in fewer documents than this before the encoder runs. Not
  cosmetic -- the expansion network is an all-pairs kNN over the vocabulary, so pruning cuts
  the most expensive stage quadratically.
- `stopwords`: a document-frequency threshold in `(0, 1]`, or `nothing` (the default) not to
  detect any. Given, it becomes the library's `stopwords=(; doc_freq_threshold=...)`, which
  costs a second pass over the sample because the counts have to be recomputed under the
  pipeline that drops the flagged tokens.
- `encoder`, `expansion`, `lemmas`: the library's own option groups, as `NamedTuple`s --
  `encoder=(; outdim=128)`, `lemmas=(; apply=true)`, and so on.

Two options this spec used to carry are gone, because the library's fit does not express them
and delegation is the policy: the `VectorModel` weighting scheme (it always fits
`IdfWeighting`/`TfWeighting`) and `min_occs` (it prunes by document count only).
"""
struct FitFromCorpus <: AbstractTextModelSpec
    textconfig::TextConfig
    min_ndocs::Int
    max_documents::Int
    stopwords::Union{Nothing,Float64}
    encoder::NamedTuple
    expansion::NamedTuple
    lemmas::NamedTuple

    function FitFromCorpus(textconfig::TextConfig=TextConfig();
                           min_ndocs::Integer=1, max_documents::Integer=1000,
                           stopwords::Union{Nothing,Real}=nothing,
                           encoder::NamedTuple=NamedTuple(),
                           expansion::NamedTuple=NamedTuple(),
                           lemmas::NamedTuple=NamedTuple())
        min_ndocs >= 1 || throw(ArgumentError("min_ndocs must be at least 1; got $min_ndocs"))
        max_documents >= 0 ||
            throw(ArgumentError("max_documents must be non-negative (0 lifts the cap); got $max_documents"))
        stopwords === nothing || 0 < stopwords <= 1 ||
            throw(ArgumentError("stopwords must be a document-frequency threshold in (0, 1], or nothing; got $stopwords"))
        new(textconfig, Int(min_ndocs), Int(max_documents),
            stopwords === nothing ? nothing : Float64(stopwords), encoder, expansion, lemmas)
    end
end

"""
    DefaultProfile(language::Symbol; nickname=..., refit=true, max_documents=1000)

Index against the published profile for `language` (`:en`, `:es`, `:pt`), adapted to this
project's own corpus at the first [`index!`](@ref index!(::BM25Engine)) call.

This is the choice to reach for. `TextSearch.refit_profile` blends the base's token counts with
this project's own, recomputes the weights from the blend, and *inherits* the stopword set,
lemma map and expansion network instead of re-deriving them. No embedding is fitted, which is
what makes a refit cheap next to a fit and is the point of bootstrapping.

# What a refit actually gives you, and what it does not

Measured: the published English paragraph profile holds 335,336 tokens; refitting it against an
800-document sample gives 12,238, of which 2,649 are tokens the sample never contained and the
base kept. Fitting on that sample alone would give 9,589.

So a refit is **not** language-wide vocabulary coverage. It narrows to this corpus's own
vocabulary, widened about a quarter by the base. What it inherits is the part an indexing corpus
cannot produce for itself: idf and BM25 weights calibrated over millions of paragraphs rather
than hundreds, plus the artifacts -- for that English base, 96 stopwords, 16,661 lemmas and
10,655 expansion entries.

`refit=false` is therefore not just "skip a step": it indexes against the base untouched, all
335,336 tokens of it, so a term this project has never seen is still in the vocabulary and still
searchable when a later batch brings it. The weights are Wikipedia's rather than yours. Choose
by which you need -- calibration for this corpus, or coverage beyond it. With `refit=false` this
is exactly `BaseProfile(load_profile(path))` with the path resolved for you.

`max_documents` caps the refit sample as it does on [`FitFromCorpus`](@ref), and here the cap is
cheap: the sample's only job is to say how this corpus differs from the base.

# Where the profile comes from

`~/.textsearch/profiles/<nickname>.zip` (or under `\$TEXTSEARCH_HOME`) -- the library of
installed profiles `textsearch install` maintains, reused rather than reinvented. Nothing is
bundled with this package and nothing is downloaded: the profiles are 70-160 MB each. If the
one you asked for is not installed, [`default_profile_path`](@ref) says so and prints the
command that installs it.

`nickname` defaults to [`DEFAULT_PROFILE_NICKNAMES`](@ref)`[language]` and can be overridden to
point at any installed profile -- a refit of your own, a different Wikipedia snapshot, a
domain-specific base.
"""
struct DefaultProfile <: AbstractTextModelSpec
    language::Symbol
    nickname::String
    refit::Bool
    max_documents::Int

    function DefaultProfile(language::Symbol;
                            nickname::Union{Nothing,AbstractString}=nothing,
                            refit::Bool=true, max_documents::Integer=1000)
        nick = nickname === nothing ? get(DEFAULT_PROFILE_NICKNAMES, language, nothing) : String(nickname)
        nick === nothing && throw(ArgumentError(
            "no default profile is known for language $(repr(language)); known: " *
            join(sort(String.(collect(keys(DEFAULT_PROFILE_NICKNAMES)))), ", ") *
            ". Pass `nickname=` to name an installed profile explicitly."))
        max_documents >= 0 ||
            throw(ArgumentError("max_documents must be non-negative (0 lifts the cap); got $max_documents"))
        new(language, nick, refit, Int(max_documents))
    end
end

"""
    DEFAULT_PROFILE_NICKNAMES

The installed-profile nickname [`DefaultProfile`](@ref) looks for, per language.

Paragraph-level profiles, matching what a project built out of paragraphs indexes: a document
frequency counted over paragraphs separates a real stopword from an artifact, where one counted
over whole articles says almost nothing. English is the `-partial` build, which is what exists.
"""
const DEFAULT_PROFILE_NICKNAMES = Dict{Symbol,String}(
    :en => "wiki20231101-en-paragraphs-partial",
    :es => "wiki20231101-es-paragraphs",
    :pt => "wiki20231101-pt-paragraphs",
)

"""
    textsearch_home() -> String

Where `textsearch install` keeps its profile library: `\$TEXTSEARCH_HOME`, or `~/.textsearch`.

The convention is replicated here rather than called, because it lives in the `textsearch` CLI
app -- which is an application, not a package this one can depend on. One line, and the env var
is the part that matters: a caller who moved their profile library expects both halves to agree
about where it went.
"""
textsearch_home() = get(ENV, "TEXTSEARCH_HOME", joinpath(homedir(), ".textsearch"))

"""
    default_profile_path(spec::DefaultProfile) -> String

The installed profile file `spec` names, under [`textsearch_home`](@ref).

Raises if it is not installed, and the message carries the command that installs it: "no such
file" naming a path under `~/.textsearch` is not actionable to someone who has never run the
CLI, and this is the most likely first thing a caller of [`DefaultProfile`](@ref) hits.
"""
function default_profile_path(spec::DefaultProfile)
    path = joinpath(textsearch_home(), "profiles", spec.nickname * ".zip")
    isfile(path) && return path
    error("""
        the default profile for $(repr(spec.language)) is not installed: no $path
        Install it from a profile zip (they are 70-160 MB, so nothing here downloads one for you):
            textsearch install path/to/$(spec.nickname).zip $(spec.nickname)
        TextSearch ships builds under corpus-profiles/profiles/. To skip the profile library \
        entirely, pass the file directly instead:
            textmodel = BaseProfile(load_profile("path/to/$(spec.nickname).zip"))""")
end

# The two specs that leave a project untrained until its first `index!` call, each carrying what
# that call needs to produce a profile. `BaseProfile` is the third and is trained on arrival.
const DeferredFit = Union{FitFromCorpus, DefaultProfile}

Base.show(io::IO, s::BaseProfile) = print(io, "BaseProfile(", s.profile.model.voc |> vocsize, " tokens)")

function Base.show(io::IO, s::FitFromCorpus)
    print(io, "FitFromCorpus(", s.textconfig.language)
    s.min_ndocs == 1 || print(io, ", min_ndocs=", s.min_ndocs)
    print(io, ", max_documents=", s.max_documents == 0 ? "uncapped" : string(s.max_documents))
    s.stopwords === nothing || print(io, ", stopwords=", s.stopwords)
    isempty(s.encoder) || print(io, ", encoder=", s.encoder)
    isempty(s.expansion) || print(io, ", expansion=", s.expansion)
    isempty(s.lemmas) || print(io, ", lemmas=", s.lemmas)
    print(io, ")")
end

Base.show(io::IO, s::DefaultProfile) = print(io, "DefaultProfile(:", s.language, ", ", s.nickname,
                                             s.refit ? ", refit" : ", as-is", ")")

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
_initial_profile(::DeferredFit) = nothing
_deferred_fit(::BaseProfile) = nothing
_deferred_fit(spec::DeferredFit) = spec

function _require_textmodel(IndexType::Type, textmodel)
    textmodel isa AbstractTextModelSpec && return textmodel
    textmodel === nothing && error("""
        a text project ($(nameof(IndexType))) needs an explicit `textmodel`, because the ways to get a \
        vocabulary are not interchangeable and the cheap one cannot be undone later:
          textmodel=DefaultProfile(:es)  -- recommended. The published profile for a language, refitted \
        to this project's corpus at the first index! call: a vocabulary fitted over a whole Wikipedia \
        edition, with its stopwords, lemmas and expansion network, adapted without fitting an embedding
          textmodel=BaseProfile(load_profile("path/to/profile.zip"))  -- a profile you already have, \
        used as it is
          textmodel=FitFromCorpus(TextConfig())  -- no base profile: fit one here, from at most \
        max_documents of this project's own corpus. Bounded in cost and thin in vocabulary, so every \
        term appended later that the sample never held is out-of-vocabulary and silently dropped""")
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
- `fitspec::Union{Nothing, DeferredFit}`: the recipe for the deferred fit (policy,
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
    fitspec::Union{Nothing, DeferredFit}
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
- `fitspec::Union{Nothing, DeferredFit}`: as on [`BM25Engine`](@ref) -- the recipe for the
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
    fitspec::Union{Nothing, DeferredFit}
    variants::Union{Nothing, Dict{String,Vector{String}}}
    distance::SimilaritySearch.PreMetric
    staged::Vector{String}
    ctx::InvertedFileContext
    search_ctx_pool::ContextPool
    deleted_ids::Set{UInt32}
    lock::ReadWriteLock
end


"""
    direct_neighbors(index::SearchGraph, i::Integer) -> Vector{UInt32}

Object `i`'s direct (not yet reverse-connected) neighbor list, read at exactly the moment
`OBSERVE` reports the range containing `i` -- i.e. before `connect_reverse_links!` has added
anything else into it.

!!! warning "what `index` looks like inside a `CallbackLog` callback, for a `SearchGraph`"
    A `SearchGraph`'s `:add!` event fires *before* `connect_reverse_links!` runs for `sp:ep`
    (`searchgraph/insertions.jl`) -- so `index.adj` for that exact range holds only the
    *direct* links just computed, none of the reverse links other nodes will later add into
    it. This is by design, not a bug to route around: `direct_neighbors` captures exactly
    that direct-links-only slice, and [`build_searchgraph`](@ref) reconnects every reverse
    link *once*, only after every staged vector and graph-indexed object's adjacency entry
    has been replayed -- reconnecting on an already-complete graph isn't safe (it isn't
    idempotent; it would duplicate reverse edges), which is exactly why this format never
    saves reverse links in the first place. `InvertedFile`/`BM25InvertedFile` have no such
    hazard -- their own events fire only after all of a call's mutation is done, so
    [`invertedfile_objects`](@ref) can read `sp:ep`'s objects straight out of `index.db`
    with nothing left pending.
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

"""
    _sample(corpus, max_documents) -> corpus, or an evenly-strided view of it

At most `max_documents` of `corpus`, taken by an even stride rather than from the front.

The front of a staged corpus is whatever batch arrived first -- one book, one import, one day's
ingest -- so a prefix is a biased sample of exactly the axis a text model should not be biased
on. A stride spans everything staged and needs no RNG, so two runs over the same corpus fit the
same model. `max_documents <= 0` means no cap.
"""
function _sample(corpus, max_documents::Integer)
    n = length(corpus)
    (max_documents <= 0 || n <= max_documents) && return corpus
    view(corpus, 1:cld(n, max_documents):n)
end

"""
    TextSearch.fit_profile(spec::FitFromCorpus, corpus; verbose=false) -> TextProfile

Fits a profile from `corpus` as [`FitFromCorpus`](@ref) `spec` asks -- by handing the whole job
to the library.

This method is an adapter and nothing else: it samples `corpus` down to `spec.max_documents`
(see [`_sample`](@ref)) and calls `TextSearch.fit_profile(spec.textconfig, sample; ...)`. This
package does not fit profiles. It used to -- a vocabulary, a weighting scheme and an optional
stopword pass, about forty lines -- and every one of those lines was a second implementation of
something the library does, with the drift that implies. Delegating is the policy, and the
forty lines are gone.

What the caller gets for it is the full pipeline rather than the cheap subset: an LSI over the
sample, a query-expansion network derived from it, and a lemma map clustered from the same
embeddings. That is strictly more than the old body produced and it costs accordingly, which is
what `spec.max_documents` exists to bound and what [`DefaultProfile`](@ref) exists to avoid.

`verbose` is passed through, defaulting to `false` here rather than the library's `true`: this
runs inside an [`index!`](@ref index!(::BM25Engine)) call, which is not a place a caller asked
for a progress report.
"""
function TextSearch.fit_profile(spec::FitFromCorpus, corpus; verbose::Bool=false)
    sample = _sample(corpus, spec.max_documents)
    stopwords = spec.stopwords === nothing ? NamedTuple() : (; doc_freq_threshold=spec.stopwords)
    TextSearch.fit_profile(spec.textconfig, sample;
                           min_ndocs=spec.min_ndocs, stopwords,
                           encoder=spec.encoder, expansion=spec.expansion,
                           lemmas=spec.lemmas, verbose)
end

"""
    train_profile(spec::DeferredFit, corpus; verbose=false) -> TextProfile

The profile a deferred spec produces, given the corpus staged by the time
[`index!`](@ref index!(::BM25Engine)) first runs.

One entry point over the two deferred specs, because `index!` should not care which it is
holding -- it asks for a profile and gets one. What happens underneath is entirely different in
cost and in quality:

- [`FitFromCorpus`](@ref) fits one from the sample, through the library.
- [`DefaultProfile`](@ref) loads the installed base for its language and, unless `refit=false`,
  adapts it with `TextSearch.refit_profile`: the base's counters blended with this corpus's,
  the weights recomputed, the stopword set, lemma map and expansion network inherited. No
  embedding is fitted, which is what makes this the cheap path *and* the one with the better
  vocabulary -- the opposite of the trade `FitFromCorpus` has to make.
"""
train_profile(spec::FitFromCorpus, corpus; verbose::Bool=false) =
    TextSearch.fit_profile(spec, corpus; verbose)

function train_profile(spec::DefaultProfile, corpus; verbose::Bool=false)
    base = TextSearch.load_profile(default_profile_path(spec))
    spec.refit || return base
    TextSearch.refit_profile(base, _sample(corpus, spec.max_documents); verbose)
end

# The variant map cached on a text engine (see `BM25Engine`'s `variants` field): derived once,
# the instant a profile becomes known, never per query. `derive_variants` returns an empty map
# immediately for any policy that already folds both case and diacritics -- the default
# `TextConfig()` among them -- so this is free unless a profile deliberately preserves them.
_derive_variants(profile::Nothing) = nothing
_derive_variants(profile::TextProfile) = derive_variants(profile.model.voc)

# An empty `QueryPipeline` on the index deliberately: policy is a per-query choice, and one
# baked into the index would be all-or-nothing for every search against it. `search_live` builds
# its own through [`query_pipeline`](@ref) and hands the finished query down, which is why the
# index never has to consult this field. The `max(..., 1)` guard covers an empty vocabulary,
# which `InvertedFile` cannot be sized zero for.
_new_textinvertedfile(profile::TextProfile, distance) =
    TextInvertedFile(profile.model, InvertedFile(max(vocsize(profile.model.voc), 1), distance),
                     TextSearch.QueryPipeline())

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

# The two logging slots a context carries, built from the two keywords every
# `create_engine`/`restore_engine` method below accepts. They are separate on purpose:
# `log_io` can never change what gets persisted, and `on_change` can never be lost by
# silencing the console.
#
# `on_change` is an optional `(index, sp, ep) -> nothing` callback -- typically a closure
# over an on-disk store -- installed as a `CallbackLog` *observer*, fired on every `:add!`
# event with exactly the range the library reports, with no batching or throttling.
# `log_io` is an optional extra `IO` (an open file handle, or `stdout`/`stderr`) that gets
# its own `InformativeLog` *reporter* alongside the default one on `stderr`.
function _engine_logging(on_change::Union{Nothing,Function}, log_io::Union{Nothing,IO}=nothing)
    reporters = log_io === nothing ? [InformativeLog()] : [InformativeLog(), InformativeLog(log_io)]
    observers = on_change === nothing ? [] : [CallbackLog(on_change)]
    (; reporters, observers)
end

# `hyperparameters_callback` on a `SearchGraphContext` drives `OptimizeParameters`'
# in-band autotuning of `BeamSearch` toward a `MinRecall` target during index
# construction/growth -- every `SearchGraphEngine` gets one from an explicit `minrecall`
# given at `create_engine` time (default 0.9) rather than running on the library's own
# untuned defaults while it grows; `calibrate!` is a separate, explicit re-optimization
# pass layered on top, not the sole writer of `engine.index.algo[]`.
_searchgraph_context(minrecall::Nothing, logging) = SearchGraphContext(; hyperparameters_callback=nothing, logging...)
_searchgraph_context(minrecall::Real, logging) = SearchGraphContext(; hyperparameters_callback=OptimizeParameters(MinRecall(Float32(minrecall))), logging...)

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
its author thought it was. `on_change::Union{Nothing,Function}`, if given, is installed as a
`SimilaritySearch.CallbackLog` observer: an `(index, sp, ep) -> nothing` callback fired on
every `:add!` event a `push_item!`/`append_items!` call reports -- e.g. to persist the range
`sp:ep` that was just inserted. `log_io::Union{Nothing,IO}`, if given, adds a second
`InformativeLog` reporter writing the same throttled status line to this `IO` -- an open file
handle or `stdout`/`stderr` both work; purely informative, and being a reporter rather than an
observer it changes nothing about what gets persisted.
"""
function create_engine(::Type{SearchGraph}; distance=SimilaritySearch.Dist.SqL2(), minrecall::Union{Nothing,Real}=0.9, textmodel=nothing, on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing)
    _reject_textmodel(SearchGraph, textmodel)
    mr = minrecall === nothing ? nothing : Float32(minrecall)
    return SearchGraphEngine(SearchGraph(distance, VectorDatabase()), _searchgraph_context(mr, _engine_logging(on_change, log_io)), mr, OptBeamSearch(), ContextPool(SearchGraphContext()), Set{UInt32}(), ReadWriteLock())
end
function create_engine(::Type{ExhaustiveSearch}; distance=SimilaritySearch.Dist.SqL2(), minrecall=nothing, textmodel=nothing, on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing)
    _reject_textmodel(ExhaustiveSearch, textmodel)
    GenericEngine{ExhaustiveSearch}(ExhaustiveSearch(distance, VectorDatabase()), GenericContext(; _engine_logging(on_change, log_io)...), ContextPool(GenericContext()), Set{UInt32}(), ReadWriteLock())
end
function create_engine(::Type{ParallelExhaustiveSearch}; distance=SimilaritySearch.Dist.SqL2(), minrecall=nothing, textmodel=nothing, on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing)
    _reject_textmodel(ParallelExhaustiveSearch, textmodel)
    GenericEngine{ParallelExhaustiveSearch}(ParallelExhaustiveSearch(distance, VectorDatabase()), GenericContext(; _engine_logging(on_change, log_io)...), ContextPool(GenericContext()), Set{UInt32}(), ReadWriteLock())
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

`textmodel::`[`AbstractTextModelSpec`](@ref) is **required** -- there is no default. See that
type for the three forms and why the order matters; briefly:

- [`DefaultProfile`](@ref)`(:es)` -- recommended. The published profile for a language, refitted
  to this project's corpus at the first [`index!`](@ref index!(::BM25Engine)) call.
- [`BaseProfile`](@ref)`(profile)` -- a profile in hand, used as it is. The engine is trained
  from this moment: the index is built immediately and nothing is inferred from later data.
- [`FitFromCorpus`](@ref)`(textconfig; ...)` -- no base profile: fit one at the first `index!`
  call from at most `max_documents` of this project's own staged corpus.

Omitting it is an error rather than a default, and that is the point: the third form's
vocabulary is fitted from a capped sample, so every term appended later that the sample never
held is out-of-vocabulary and silently dropped from then on. Nothing about a project's
behaviour later reveals that this choice was made by omission, which is exactly why it cannot
be.
"""
function create_engine(::Type{BM25InvertedFile}; distance=nothing, minrecall=nothing,
                       textmodel::Union{Nothing,AbstractTextModelSpec}=nothing,
                       on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing)
    spec = _require_textmodel(BM25InvertedFile, textmodel)
    profile = _initial_profile(spec)
    index = profile === nothing ? nothing : BM25InvertedFile(profile.model.voc)
    BM25Engine(index, profile, _deferred_fit(spec), _derive_variants(profile), String[],
               InvertedFileContext(; _engine_logging(on_change, log_io)...),
               ContextPool(InvertedFileContext()), Set{UInt32}(), ReadWriteLock())
end

function create_engine(::Type{InvertedFile}; distance=SimilaritySearch.Dist.NormCosine(), minrecall=nothing,
                       textmodel::Union{Nothing,AbstractTextModelSpec}=nothing,
                       on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing)
    spec = _require_textmodel(InvertedFile, textmodel)
    profile = _initial_profile(spec)
    index = profile === nothing ? nothing : _new_textinvertedfile(profile, distance)
    InvertedFileEngine(index, profile, _deferred_fit(spec), _derive_variants(profile), distance, String[],
                       InvertedFileContext(; _engine_logging(on_change, log_io)...),
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
    return SearchGraphEngine(index, _searchgraph_context(state.minrecall, _engine_logging(on_change, log_io)), state.minrecall, state.opt_beamsearch, ContextPool(SearchGraphContext()), state.deleted_ids, ReadWriteLock())
end

function restore_engine(::Type{IndexType}, state; on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing) where {IndexType}
    return GenericEngine{IndexType}(state.index, GenericContext(; _engine_logging(on_change, log_io)...), ContextPool(GenericContext()), state.deleted_ids, ReadWriteLock())
end

function restore_engine(::Type{BM25Engine}, state; on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing)
    profile = state.profile
    index = profile === nothing ? nothing : build_bm25invertedfile(profile.model.voc, state.object_blocks)
    return BM25Engine(index, profile, state.fitspec, _derive_variants(profile), state.staged, InvertedFileContext(; _engine_logging(on_change, log_io)...), ContextPool(InvertedFileContext()), state.deleted_ids, ReadWriteLock())
end

function restore_engine(::Type{InvertedFileEngine}, state; on_change::Union{Nothing,Function}=nothing, log_io::Union{Nothing,IO}=nothing)
    profile = state.profile
    index = profile === nothing ? nothing : build_textinvertedfile(state.distance, profile, state.object_blocks)
    return InvertedFileEngine(index, profile, state.fitspec, _derive_variants(profile), state.distance, state.staged, InvertedFileContext(; _engine_logging(on_change, log_io)...), ContextPool(InvertedFileContext()), state.deleted_ids, ReadWriteLock())
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
            profile = train_profile(engine.fitspec, engine.staged)
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
            profile = train_profile(engine.fitspec, engine.staged)
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

index!(engine::GenericEngine) = engine

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
    stored_payload(engine::AbstractSearchEngine, id::Integer) -> Union{String, Vector{Float32}, Nothing}

The very text or vector indexed under internal `id`, or `nothing` if `id` is out of range.

This is the read-back side of `add_item!`, and it exists because a search result is not much
use without it: a hit carries an id and a score, and everything else a caller knows about the
item comes from its metadata record -- which deliberately does not hold the payload, since the
payload already lives here. Before this existed the only way to a text project's paragraph was
`engine.staged[id]`, reaching past the public surface into a field.

Reads from wherever the engine already keeps the item, with no second copy anywhere: a text
engine's `staged` (every paragraph ever staged, in insertion order, which *is* id order), a
`SearchGraph`'s or a `GenericEngine`'s own `database`. A dense payload is materialized into a
fresh `Vector{Float32}` rather than handed out as a view, because the view aliases the index's
live storage and the caller has no way to know that.
"""
function stored_payload(engine::Union{BM25Engine, InvertedFileEngine}, id::Integer)
    1 <= id <= length(engine.staged) || return nothing
    engine.staged[id]
end

function stored_payload(engine::Union{SearchGraphEngine, GenericEngine}, id::Integer)
    db = database(engine.index)
    1 <= id <= length(db) || return nothing
    convert(Vector{Float32}, db[id])
end

"""
    query_pipeline(engine, policy::QueryPolicy=QueryPolicy()) -> TextSearch.QueryPipeline

How this engine should treat a query, as the one value `TextSearch` takes for it: the caller's
`policy`, this engine's cached variant map, and the profile's expansion network with its
distances.

Built per call rather than once per index, because `policy` is per call: correcting and
expanding are guesses about what a person meant, so the same project has to be able to answer
both ways. The index itself therefore carries an empty pipeline and every
search assembles its own.

The network is included only when the profile *applies* it. A fitted profile carries a network
without applying it -- computing an artifact and deciding to use it are different acts -- and
`QueryPipeline` reads a network it is given as the request to expand with it.
"""
function query_pipeline(engine::Union{BM25Engine, InvertedFileEngine}, policy::QueryPolicy=QueryPolicy())
    profile = engine.profile
    profile === nothing && error("this text engine has not been trained yet -- index! at least one staged item, or create it with a profile, before building a query")
    expansion = (profile.applied.query_expansion && !isempty(profile.query_expansion)) ?
                profile.query_expansion : nothing
    TextSearch.QueryPipeline(;
        policy,
        # `variants` is only read when correction is on; withholding it under `:off` keeps the
        # candidate group empty and makes the resolution a pass-through of what was typed
        variants = policy.correction === :off ? nothing : engine.variants,
        expansion,
        distances = expansion === nothing ? nothing : profile.query_expansion_distances)
end

"""
    resolve_query(engine, text::AbstractString, policy::QueryPolicy=QueryPolicy()) -> TextSearch.ResolvedQuery

Runs `text` through `TextSearch`'s query pipeline under this engine's
[`query_pipeline`](@ref): tokenized by the profile's own `TextConfig` -- the same normalization,
lemma and stopword stages every indexed document went through -- then corrected against the
vocabulary and widened by the expansion network, as `policy` allows.

The result carries both halves a caller needs: `terms`, what to actually search for, and
`resolution`, what correction did to each spelling that was typed. The second is not
bookkeeping: correcting a query is a substitution the person who typed it is owed a report of,
which is what `TextSearch.explain(rq.resolution)` renders and what `QueryPolicy(correction=:off)`
undoes.

This used to be assembled here -- tokenize, `resolve_query_tokens`, rebuild the term list,
restrict the network to `expansion_sources`, weight the neighbours. `TextSearch` published all of
it as `query_tokens`/`querybow`/`queryvector`, routed through both of its inverted files, so this
is now the library's implementation with this engine's profile and cache wired into it. Two
behaviours moved with it and are worth naming, because both were deliberate here and are
deliberate there:

- **Typed terms are deduplicated.** A query of `"casa casa casa"` searches `casa` once. This
  module used to preserve multiplicity out of concern for term frequency; the concern does not
  survive contact with the scorers. `bm25score` reads only which ids the query holds, never their
  counts, and `queryvector` weights the words a person meant once each. A query is a set of
  intents, not a document.
- **Expansion contributions are not deduplicated**, and that asymmetry is the point: a neighbour
  reachable from two query tokens contributes twice, and `queryvector` sums the contributions
  while `querybow` collapses them.

Errors if `engine` has no profile yet -- nothing indexed, so no vocabulary to resolve against.
"""
resolve_query(engine::Union{BM25Engine, InvertedFileEngine}, text::AbstractString, policy::QueryPolicy=QueryPolicy()) =
    TextSearch.query_tokens(engine.profile.model.voc, text, query_pipeline(engine, policy))

# Whether `dist` makes an inverted file score token membership rather than weighted vectors, in
# which case a query is a presence-only bag and not a vector. `TextInvertedFile` branches on
# exactly this in its own `search`; mirrored here because this module builds the query itself,
# per call, to honour a per-call `QueryPolicy` that an index-level pipeline cannot express.
_is_set_distance(dist) = parentmodule(typeof(dist)) === SimilaritySearch.Dist.Sets

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
            snap = copy(ctx.costdists)
            res = knnqueue(KnnSorted, max(k, 1))
            if bs !== nothing
                vstate = SimilaritySearch.getvstate(length(engine.index), ctx)
                search(bs, engine.index, ctx, query, res, engine.index.hints, vstate)
            else
                search(engine.index, ctx, query, res)
            end
            evals = SimilaritySearch.distance_evaluations(ctx, snap)
            return _collect_live(engine, res, evals)
        finally
            checkin!(engine.search_ctx_pool, ctx)
        end
    end
end

function search_live(engine::GenericEngine, query, k::Int; bs_override=nothing, minrecall=nothing, policy=nothing)
    read_lock(engine.lock) do
        ctx = checkout!(engine.search_ctx_pool)
        try
            snap = copy(ctx.costdists)
            res = knnqueue(KnnSorted, max(k, 1))
            search(engine.index, ctx, query, res)
            evals = SimilaritySearch.distance_evaluations(ctx, snap)
            return _collect_live(engine, res, evals)
        finally
            checkin!(engine.search_ctx_pool, ctx)
        end
    end
end

function search_live(engine::BM25Engine, query, k::Int; bs_override=nothing, minrecall=nothing, policy::QueryPolicy=QueryPolicy())
    read_lock(engine.lock) do
        profile = engine.profile
        profile === nothing && return (id=Int32[], dist=Float32[], deleted=Bool[], distance_evaluations=0)
        voc = profile.model.voc
        # presence only, and not a shortcut: BM25 scores from which ids the query holds, never
        # from their counts (see `bm25score`), so a weighted query bag would be carried through
        # the whole search and then ignored
        bow = TextSearch.querybow(voc, resolve_query(engine, query, policy))
        ctx = checkout!(engine.search_ctx_pool)
        try
            snap = copy(ctx.costdists)
            res = knnqueue(KnnSorted, max(k, 1))
            search(engine.index, ctx, bow, res)
            evals = SimilaritySearch.distance_evaluations(ctx, snap)
            return _collect_live(engine, res, evals)
        finally
            checkin!(engine.search_ctx_pool, ctx)
        end
    end
end

function search_live(engine::InvertedFileEngine, query, k::Int; bs_override=nothing, minrecall=nothing, policy::QueryPolicy=QueryPolicy())
    read_lock(engine.lock) do
        profile = engine.profile
        profile === nothing && return (id=Int32[], dist=Float32[], deleted=Bool[], distance_evaluations=0)
        voc = profile.model.voc
        rq = resolve_query(engine, query, policy)
        # The representation decides what to do with the pipeline's weights: a set distance
        # scores membership and ignores them, a vector distance applies them and normalizes.
        q = _is_set_distance(engine.distance) ? TextSearch.querybow(voc, rq) :
                                                TextSearch.queryvector(profile.model, rq)
        ctx = checkout!(engine.search_ctx_pool)
        try
            snap = copy(ctx.costdists)
            res = knnqueue(KnnSorted, max(k, 1))
            search(engine.index, ctx, q, res)
            evals = SimilaritySearch.distance_evaluations(ctx, snap)
            return _collect_live(engine, res, evals)
        finally
            checkin!(engine.search_ctx_pool, ctx)
        end
    end
end

function _collect_live(engine::AbstractSearchEngine, res, evals=0)
    ids = view(res.ids, res.sp:res.ep)
    dists = view(res.dists, res.sp:res.ep)
    deleted = [id in engine.deleted_ids for id in ids]
    return (id=ids, dist=dists, deleted=deleted, distance_evaluations=evals)
end

end # module
