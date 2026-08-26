module IndexEngine

using SimilaritySearch
# Not re-exported at SimilaritySearch's top level (only ExhaustiveSearch is) -- it lives in
# the Exact submodule, brought into SimilaritySearch's own namespace via `using .Exact`.
import SimilaritySearch: ParallelExhaustiveSearch
using TextSearch
using Base.Threads
using SparseArrays: SparseVector
using Dates: Dates

export AbstractSearchEngine, DenseEngine, SparseEngine, FullTextEngine
export GraphBackend, ExactBackend, SparseBackend, TextBackend, backend_tag, payload_kind
export ReadWriteLock, read_lock, write_lock
export ContextPool, checkout!, checkin!
export create_engine, create_sparse_engine, restore_engine, snapshot_state, extra_state_fields, add_item!, index!, search_live, mark_deleted!
export stored_payload
export calibrate!, current_beamsearch, OptBeamSearch, DEFAULT_MINRECALL_LEVELS
export direct_neighbors, apply_searchgraph_vectors!, build_searchgraph
export invertedfile_objects, build_bm25invertedfile, build_textinvertedfile, build_sparseinvertedfile
export text_profile, text_vocabulary, resolve_query
export AbstractTextModelSpec, BaseProfile, FitFromCorpus, DefaultProfile
export DEFAULT_PROFILE_NICKNAMES, default_profile_path, train_profile
export is_text_index_type, validate_textmodel
export BACKENDS, engine_kind, default_backend, validate_backend, default_distance
export BACKENDS, default_backend, validate_backend, default_distance

"""
    AbstractSearchEngine

Common supertype for every concrete engine kind ([`DenseEngine`](@ref),
[`DenseEngine`](@ref), [`SparseEngine`](@ref), [`FullTextEngine`](@ref)). Each kind
carries only the state it actually needs -- e.g. `minrecall` only exists on
`DenseEngine{GraphBackend}`, `profile` only on the text engines -- and the concrete Julia type
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

`template` is never mutated -- it's read-only, existing only to be `deepcopy`'d
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
`deepcopy` of `pool.template`.
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
recall quality (see [`calibrate!`](@ref) and [`DenseEngine`](@ref)).
"""
const OptBeamSearch = Dict{Float32, BeamSearch}

"""
    DEFAULT_MINRECALL_LEVELS

The recall levels [`calibrate!`](@ref) populates a `DenseEngine{GraphBackend}`'s
[`OptBeamSearch`](@ref) with when no explicit `levels` are given.
"""
const DEFAULT_MINRECALL_LEVELS = Float32[0.8, 0.9, 0.95, 0.97]

"""
    AbstractTextModelSpec

The text-model decision a text project is created with, as a value. Three answers to "where
does this project's vocabulary come from", in the order worth preferring them:

1. [`DefaultProfile`](@ref)`(:es)` -- the published profile for a language, refitted to this
   project's own corpus at the first [`index!`](@ref index!(::FullTextEngine)) call: weights
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

The fit is deferred to the first [`index!`](@ref index!(::FullTextEngine)) call, and delegated
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
project's own corpus at the first [`index!`](@ref index!(::FullTextEngine)) call.

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
Official published profiles are available for `:en`, `:es`, `:eu`, `:fr`, `:it`, `:pt`, and `:ru`.
"""
const DEFAULT_PROFILE_NICKNAMES = Dict{Symbol,String}(
    :en => "en",
    :es => "es",
    :eu => "eu",
    :fr => "fr",
    :it => "it",
    :pt => "pt",
    :ru => "ru",
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

Raises if it is not installed, and the message carries the command that installs or downloads it:
"no such file" naming a path under `~/.textsearch` is not actionable to someone who has never run
the CLI, and this is the most likely first thing a caller of [`DefaultProfile`](@ref) hits.
"""
function default_profile_path(spec::DefaultProfile)
    path = joinpath(textsearch_home(), "profiles", spec.nickname * ".zip")
    isfile(path) && return path

    # Check fallback legacy paragraph nicknames if language is in (:en, :es, :pt)
    legacy = if spec.language === :es
        "wiki20231101-es-paragraphs"
    elseif spec.language === :en
        "wiki20231101-en-paragraphs-partial"
    elseif spec.language === :pt
        "wiki20231101-pt-paragraphs"
    else
        nothing
    end
    if legacy !== nothing
        legacy_path = joinpath(textsearch_home(), "profiles", legacy * ".zip")
        isfile(legacy_path) && return legacy_path
    end

    error("""
        the default profile for $(repr(spec.language)) is not installed: no $path
        Install it by calling `download_profile($(repr(spec.nickname)))` or using the textsearch CLI:
            textsearch download $(spec.nickname)
        or from a local profile zip:
            textsearch install path/to/$(spec.nickname).zip $(spec.nickname)
        To skip the profile library entirely, pass the file directly instead:
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

Whether a backend type is one of the text ones -- the ones that take a
[`AbstractTextModelSpec`](@ref) and reject nothing else. `InvertedFile` and
`TextInvertedFile` both name the weighted engine (see [`FullTextEngine`](@ref)).
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
    validate_textmodel(backend::Type, textmodel) -> Union{Nothing, AbstractTextModelSpec}

Checks `textmodel` against a *backend* -- required for a text one, refused for a dense one --
and hands back the spec (or `nothing`). Raises the same errors [`create_engine`](@ref) would.

A backend and not an engine, which is a real limit worth naming: `InvertedFile` is legal under
both `SparseEngine` and `FullTextEngine`, so this cannot tell a sparse project from a text one.
`create_project` therefore checks the sparse case itself, against the engine the caller named,
and only delegates here for a text project.

Split out so a caller can run the check *before* committing to anything: `create_project`
opens the project's RocksDB directory before it ever reaches `create_engine`, so letting the
error surface there would leave a created directory and a held write lock behind on a call
that failed. `create_engine` still checks for itself, for callers that reach it directly.
"""
validate_textmodel(IndexType::Type, textmodel) =
    is_text_index_type(IndexType) ? _require_textmodel(IndexType, textmodel) :
                                    _reject_textmodel(IndexType, textmodel)

"""
    GraphBackend
    ExactBackend{IndexType}
    SparseBackend
    TextBackend

The index a project searches through, with the context it searches with and whatever else that
particular index kind needs to be driven.

Four of them because there are four ways to hold an index, not because there are four kinds of
project: [`DenseEngine`](@ref) is parameterized by which dense backend it has, and
[`FullTextEngine`](@ref) holds a `TextBackend` whose `kind` says which library inverted file is
underneath. The engine is the data it takes; the backend is the index and the distance. Keeping
them apart is what lets a project of sparse vectors exist at all -- before this, an inverted
file and its distance were reachable only through a text engine, and so only by a caller who
had text.

- `GraphBackend` carries `minrecall`/`opt_beamsearch`, and it is the only one that does. A beam
  is a `SearchGraph`'s to tune; putting that state on the engine would give every exact and
  every inverted-file project two fields it can only ever leave empty.
- `ExactBackend{IndexType}` covers `ExhaustiveSearch` and `ParallelExhaustiveSearch`, which
  differ in nothing this module has to know about.
- `SparseBackend` records the `dimension` it was created with, because an `InvertedFile` is a
  fixed array of posting lists: the dimension is structural, not descriptive, and every item
  appended has to agree with it.
- `TextBackend`'s `index` is `nothing` until the engine has a profile to build it from, and its
  `kind` is `BM25InvertedFile` or `TextInvertedFile` -- which, with `distance`, is the whole of
  what used to be the difference between two separate engine types.
"""
mutable struct GraphBackend
    index::SearchGraph
    ctx::SearchGraphContext
    minrecall::Union{Nothing, Float32}
    opt_beamsearch::OptBeamSearch
end

mutable struct ExactBackend{IndexType}
    index::IndexType
    ctx::GenericContext
end

mutable struct SparseBackend
    index::InvertedFile
    ctx::InvertedFileContext
    distance::SimilaritySearch.PreMetric
    dimension::Int
end

mutable struct TextBackend
    index::Union{Nothing, AbstractInvertedFile}
    ctx::InvertedFileContext
    kind::Type
    distance::Union{Nothing, SimilaritySearch.PreMetric}
end

"""
    DenseEngine{B}

A project of dense vectors: it takes `DenseItem`s, and `B` says which
dense index holds them -- [`GraphBackend`](@ref) for an approximate, self-tuning `SearchGraph`,
`ExactBackend` for a brute-force scan.

This replaced a SearchGraphEngine and a GenericEngine{IndexType}, which split dense
projects by *index kind* at the level where a caller chooses. The kind of data is what a caller
knows; which index holds it is a backend decision, and one that can now change without the
project becoming a different type of thing.

Only `GraphBackend` has a `BeamSearch` to calibrate, so [`calibrate!`](@ref),
[`current_beamsearch`](@ref) and `minrecall`-driven search require that backend rather than
being defined-but-inert on the other.

# Fields
- `backend::B`: the index, its context, and the backend's own state.
- `search_ctx_pool::ContextPool`: one private context per concurrent [`search_live`](@ref) call
  (see [`ContextPool`](@ref)). The backend's own `ctx` is reserved for insertion.
- `deleted_ids::Set{UInt32}`: logically deleted document ids -- `UInt32` across every engine
  kind, matching the id type search results come back as.
- `lock::ReadWriteLock`: see its docstring for why a reader/writer lock and not a mutex.
"""
mutable struct DenseEngine{B} <: AbstractSearchEngine
    backend::B
    search_ctx_pool::ContextPool
    deleted_ids::Set{UInt32}
    lock::ReadWriteLock
end

"""
    SparseEngine

A project of sparse vectors: it takes `SparseItem`s and indexes them
in a `SimilaritySearch.InvertedFile` under any distance that reads a sparse vector -- cosine
over the weights, or one of `Dist.Sets.*` over the nonzero positions.

New with this restructuring, and the point of it. An inverted file and its distance used to be
reachable only through a text engine, which meant a caller holding sparse vectors had to
pretend to have text: fit a profile it did not want, stage strings it did not have. This is the
layer that was implied and missing -- vocabulary-free, no profile, no staged text.

Which is also why BM25 is not one of its distances. `BM25InvertedFile` holds a `Vocabulary` and
scores from `getndocs`/`avgdoclen`; BM25 is not a metric over sparse vectors but a scoring rule
over a corpus, so it belongs to [`FullTextEngine`](@ref), which has the corpus statistics.
Sparse in, sparse out, and nothing here knows what a token is.

# Fields
- `backend::SparseBackend`: the inverted file, its context, the distance and the dimension.
- `search_ctx_pool`, `deleted_ids`, `lock`: as on [`DenseEngine`](@ref).
"""
mutable struct SparseEngine <: AbstractSearchEngine
    backend::SparseBackend
    search_ctx_pool::ContextPool
    deleted_ids::Set{UInt32}
    lock::ReadWriteLock
end

"""
    FullTextEngine

A project of text: it takes `TextItem`s, holds the text model that turns
them into something an inverted file can index, and searches through whichever inverted file its
backend names.

This replaced BM25Engine and InvertedFileEngine, which were near-duplicates -- the same
staged text, the same profile, the same deferred fit -- differing in which library index they
built and how a query was encoded against it. Both of those are the backend's business now (see
[`GraphBackend`](@ref) for the family), so the scheme a caller wants (BM25, tf-idf, a set
similarity) is a field rather than a type.

Untrained (`backend.index === nothing`, `profile === nothing`) until either a
[`BaseProfile`](@ref)/[`DefaultProfile`](@ref) supplies a profile at creation or the first
[`index!`](@ref index!(::FullTextEngine)) call produces one. `add_item!`/`append_items!` only
ever *stage* raw text; `index!` is the explicit step that trains, when it has to, and
encodes/indexes the backlog.

# Fields
- `backend::TextBackend`: the inverted file (or `nothing`), its context, which kind it is, and
  the distance for the kinds that take one.
- `profile::Union{Nothing, TextProfile}`: the whole text model -- vocabulary and weights plus the
  corpus-produced artifacts and the lineage. One field, and the `TextConfig` the tokenizer runs
  is derived from it, so an index cannot end up tokenizing documents through a different lemma
  map than the one it saves. `backend.index === nothing` if and only if this is `nothing`.
- `fitspec::Union{Nothing, DeferredFit}`: what the deferred training does, consulted only by the
  `index!` call that has to produce a profile and never again. `nothing` for a project created
  from a [`BaseProfile`](@ref), which has nothing to defer.
- `staged::Vector{String}`: every raw text ever staged, in insertion order -- the text-engine
  counterpart of a dense backend's `.db`. `index!` catches up the range
  `(index === nothing ? 0 : length(index))+1:length(staged)`.
- `search_ctx_pool`, `deleted_ids`, `lock`: as on [`DenseEngine`](@ref).
"""
mutable struct FullTextEngine <: AbstractSearchEngine
    backend::TextBackend
    profile::Union{Nothing, TextProfile}
    fitspec::Union{Nothing, DeferredFit}
    staged::Vector{String}
    search_ctx_pool::ContextPool
    deleted_ids::Set{UInt32}
    lock::ReadWriteLock
end

"""
    payload_kind(engine::AbstractSearchEngine) -> Symbol

What this project indexes: `:dense`, `:sparse` or `:text`.

Replaces an is_text_index predicate that a dozen guards used as a stand-in for "not dense".
With two engine kinds that reading happened to hold; with three it is simply false -- a
`SparseEngine` is neither text nor dense -- so a guard meaning "this needs raw vectors" now says
`payload_kind(engine) === :dense` rather than `!is_text_index(engine)`.

It also names the item type a project accepts: `:dense` takes `DenseItem`s, `:sparse` takes
`SparseItem`s, `:text` takes `TextItem`s.
"""
payload_kind(::DenseEngine) = :dense
payload_kind(::SparseEngine) = :sparse
payload_kind(::FullTextEngine) = :text


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
index!(::DenseEngine{GraphBackend}))'s stage-then-index split: `append_items!`/`add_item!` only
ever stage into `.db`, never graph-connect by themselves anymore).

`graph_len` (persisted separately, see `Persistence`'s `:graph_len` engine field) is the
count that actually matters for the *graph* structure: it can be `<= length(index.db)` if
the process closed (or crashed) after staging some vectors but before an explicit
[`index!`](@ref index!(::DenseEngine{GraphBackend})) call caught them up. Only object ids `1:graph_len`
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
posting lists *before* `LOG` fires at all (see `SimilaritySearch.CallbackLog`'s docstring), so this
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
  `_materialize` dispatches on the *target index type*, not just the view.
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
has not been trained yet -- see [`FullTextEngine`](@ref)). `text_vocabulary` is the shortcut for
the vocabulary inside it, which is what `bagofwords`/`token2id`/[`resolve_query`](@ref) all
work against.
"""
text_profile(engine::FullTextEngine) = engine.profile
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
runs inside an [`index!`](@ref index!(::FullTextEngine)) call, which is not a place a caller asked
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
[`index!`](@ref index!(::FullTextEngine)) first runs.

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

"""
    _text_index(profile::TextProfile, kind::Type, distance) -> AbstractInvertedFile

The library index a trained text engine searches through, built from `profile` itself.

Built from the profile rather than from `profile.model`/`.voc` because the profile-taking
constructors assemble the index's `QueryPipeline` too -- deriving the variant map once and
attaching the profile's expansion network if the profile applies it. That is state this module
used to keep a second copy of, in a `variants` field it derived and cached itself. One map, in
the index, is the whole reason that field is gone.

`distance` is ignored for BM25, which scores through its own `bm25score` and has no metric to
choose (see [`default_distance`](@ref)).
""" 
function _text_index(profile::TextProfile, kind::Type, distance)
    kind === BM25InvertedFile && return BM25InvertedFile(profile)
    TextInvertedFile(profile; dist=distance)
end

"""
    build_bm25invertedfile(profile::TextProfile, object_blocks) -> BM25InvertedFile

Rebuilds a `BM25InvertedFile` against a trained `voc` by replaying every saved raw object
(as produced incrementally via [`invertedfile_objects`](@ref) and read back via
`Persistence.load_object_blocks`) through the library's own `push_item!`, in order -- a
full rebuild-by-reinsertion, not an incremental deserialize (see
`Persistence.InvertedFileObjectStore`'s docstring for why, and its documented scaling
limits).
"""
function build_bm25invertedfile(profile::TextProfile, object_blocks)
    index = BM25InvertedFile(profile)
    ctx = InvertedFileContext()
    for block in object_blocks, obj in block
        push_item!(index, ctx, obj)
    end
    return index
end

"""
    build_sparseinvertedfile(distance, dimension, object_blocks) -> InvertedFile

Rebuilds a [`SparseEngine`](@ref)'s inverted file by replaying every saved sparse vector through
the library's own `push_item!`, in order -- see [`build_bm25invertedfile`](@ref) for the same
rebuild-by-reinsertion approach and its scaling caveat.

`dimension` comes from the saved state rather than from the vectors: an empty project has none
to read it off, and one whose blocks happen to hold no nonzero in the last position would
otherwise come back a different shape than it was created with.
"""
function build_sparseinvertedfile(distance, dimension::Integer, object_blocks)
    index = InvertedFile(Int(dimension), distance)
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
    index = TextInvertedFile(profile; dist=distance)
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
function _engine_logging(on_change::Union{Nothing,Function}, log_io::Union{Nothing,IO})
    reporters = log_io === nothing ? [InformativeLog()] : [InformativeLog(), InformativeLog(log_io)]
    observers = on_change === nothing ? [] : [CallbackLog(on_change)]
    (; reporters, observers)
end

# `hyperparameters_callback` on a `SearchGraphContext` drives `OptimizeParameters`'
# in-band autotuning of `BeamSearch` toward a `MinRecall` target during index
# construction/growth -- every `DenseEngine{GraphBackend}` gets one from an explicit `minrecall`
# given at `create_engine` time (default 0.9) rather than running on the library's own
# untuned defaults while it grows; `calibrate!` is a separate, explicit re-optimization
# pass layered on top, not the sole writer of `engine.backend.index.algo[]`.
_searchgraph_context(minrecall::Nothing, logging) = SearchGraphContext(; hyperparameters_callback=nothing, logging...)
_searchgraph_context(minrecall::Real, logging) = SearchGraphContext(; hyperparameters_callback=OptimizeParameters(MinRecall(Float32(minrecall))), logging...)

"""
    BACKENDS

Which backends each engine kind accepts, and in which order -- the first is that engine's
default.

A table rather than a set of methods because it is read three ways: to default a backend, to
check one, and to say in an error message what the alternatives were. Three methods would drift
from one another; one table cannot.

`InvertedFile` appears under both `SparseEngine` and `FullTextEngine`, and that is not a
mistake: the same library index serves a project of sparse vectors and a project of text whose
vectors come from a profile. Which of the two you get is `engine`'s to say -- the entire reason
`engine` and `backend` are separate arguments.
"""
const BACKENDS = Dict{Type,Vector{Type}}(
    DenseEngine    => Type[SearchGraph, ExhaustiveSearch, ParallelExhaustiveSearch],
    SparseEngine   => Type[InvertedFile],
    FullTextEngine => Type[BM25InvertedFile, TextInvertedFile, InvertedFile],
)

"""
    engine_kind(engine::Type) -> Symbol

The [`payload_kind`](@ref) an engine *type* stands for, before any instance exists.

`payload_kind` answers it for a live engine, which is what every guard wants; this answers it
for the type a caller named, which is what `create_project` needs before it has built anything.
Two functions rather than one because at that point there is no instance to ask.
"""
function engine_kind(engine::Type)
    engine === DenseEngine && return :dense
    engine === SparseEngine && return :sparse
    engine === FullTextEngine && return :text
    error("unknown engine $(nameof(engine)); expected DenseEngine, SparseEngine or FullTextEngine")
end

"""
    default_backend(engine::Type) -> Type
    validate_backend(engine::Type, backend::Type) -> Type

The backend an engine kind gets when the caller does not name one, and the check that a named
one belongs to it.

`SearchGraph` for a dense project (approximate and self-tuning, the useful default at any size),
`InvertedFile` for a sparse one (its only backend), `BM25InvertedFile` for text -- what
"full-text search" means before anybody asks for something else.

`validate_backend` earns its place because the pairing is checkable and the failure is otherwise
obscure: `engine=DenseEngine, backend=BM25InvertedFile` would reach `create_engine` and die
there on a `MethodError` about keyword arguments, naming nothing a caller could act on.
"""
function default_backend(engine::Type)
    haskey(BACKENDS, engine) ||
        error("unknown engine $(nameof(engine)); expected DenseEngine, SparseEngine or FullTextEngine")
    first(BACKENDS[engine])
end

function validate_backend(engine::Type, backend::Type)
    haskey(BACKENDS, engine) ||
        error("unknown engine $(nameof(engine)); expected DenseEngine, SparseEngine or FullTextEngine")
    backend in BACKENDS[engine] && return backend
    error("$(nameof(backend)) is not a backend for $(nameof(engine)); it takes " *
          join(("$(nameof(b))" for b in BACKENDS[engine]), ", ", " or "))
end

"""
    default_distance(backend::Type) -> PreMetric

The distance a backend is built with when the caller does not name one.

It lives here, as a function of the index kind, because the alternative is what this package
had: a default on `create_engine` *and* a `distance=nothing` sentinel on `create_project`,
with the caller branching on which of the two to let win. Two defaults for one decision, and a
`?:` at the call site to arbitrate them. Now `create_project` resolves the sentinel through
this function once and always passes a real distance down, which is also why nothing below the
public surface carries a default any more (see DEVELOPMENT_STRATEGY.md).
"""
default_distance(::Type{SearchGraph}) = SimilaritySearch.Dist.SqL2()
default_distance(::Type{ExhaustiveSearch}) = SimilaritySearch.Dist.SqL2()
default_distance(::Type{ParallelExhaustiveSearch}) = SimilaritySearch.Dist.SqL2()
default_distance(::Type{InvertedFile}) = SimilaritySearch.Dist.NormCosine()
default_distance(::Type{TextInvertedFile}) = SimilaritySearch.Dist.NormCosine()
# BM25 scores through its own `bm25score`; there is no distance to choose, and `nothing` is the
# honest answer rather than a placeholder metric nothing consults.
default_distance(::Type{BM25InvertedFile}) = nothing

"""
    create_engine(::Type{SearchGraph}; distance, minrecall, textmodel, on_change, log_io) -> DenseEngine{GraphBackend}
    create_engine(::Type{ExhaustiveSearch}; distance, minrecall, textmodel, on_change, log_io) -> DenseEngine{ExactBackend{ExhaustiveSearch}}
    create_engine(::Type{ParallelExhaustiveSearch}; distance, minrecall, textmodel, on_change, log_io) -> DenseEngine{ExactBackend{ParallelExhaustiveSearch}}

Every keyword is required, and none of them has a default here -- see
[`default_distance`](@ref) and DEVELOPMENT_STRATEGY.md. `create_project` is the public
boundary that chooses; this function is told.

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
function create_engine(::Type{SearchGraph}; distance, minrecall::Union{Nothing,Real}, textmodel, on_change::Union{Nothing,Function}, log_io::Union{Nothing,IO})
    _reject_textmodel(SearchGraph, textmodel)
    mr = minrecall === nothing ? nothing : Float32(minrecall)
    backend = GraphBackend(SearchGraph(distance, VectorDatabase()),
                           _searchgraph_context(mr, _engine_logging(on_change, log_io)),
                           mr, OptBeamSearch())
    DenseEngine(backend, ContextPool(SearchGraphContext()), Set{UInt32}(), ReadWriteLock())
end

# `minrecall` is accepted and ignored on the exact backends: they have no beam to tune, and
# taking the same keyword set for every index kind is what lets `create_project` pass one bundle
# down without knowing which backend it is talking to.
function _create_exact_engine(IndexType::Type, distance, textmodel,
                              on_change::Union{Nothing,Function}, log_io::Union{Nothing,IO})
    _reject_textmodel(IndexType, textmodel)
    backend = ExactBackend(IndexType(distance, VectorDatabase()),
                           GenericContext(; _engine_logging(on_change, log_io)...))
    DenseEngine(backend, ContextPool(GenericContext()), Set{UInt32}(), ReadWriteLock())
end

create_engine(::Type{ExhaustiveSearch}; distance, minrecall, textmodel, on_change::Union{Nothing,Function}, log_io::Union{Nothing,IO}) =
    _create_exact_engine(ExhaustiveSearch, distance, textmodel, on_change, log_io)
create_engine(::Type{ParallelExhaustiveSearch}; distance, minrecall, textmodel, on_change::Union{Nothing,Function}, log_io::Union{Nothing,IO}) =
    _create_exact_engine(ParallelExhaustiveSearch, distance, textmodel, on_change, log_io)

"""
    create_engine(::Type{BM25InvertedFile}; textmodel, distance, minrecall, on_change, log_io) -> FullTextEngine
    create_engine(::Type{InvertedFile}; textmodel, distance, minrecall, on_change, log_io) -> FullTextEngine
    create_engine(::Type{TextInvertedFile}; ...) -> FullTextEngine

Creates a new, empty text search engine of the given index type. `minrecall` is accepted and
ignored on both, and `distance` is accepted and ignored on `BM25InvertedFile` (BM25 always
scores via its own `bm25score`), so a caller can pass the same keyword set uniformly
regardless of index type. `on_change`/`log_io` are as in the dense `create_engine` methods
above. `TextInvertedFile` and `InvertedFile` select the same engine and are interchangeable
here; `TextInvertedFile` is the name of what actually gets built (see
[`FullTextEngine`](@ref)).

`textmodel::`[`AbstractTextModelSpec`](@ref) is **required** -- there is no default. See that
type for the three forms and why the order matters; briefly:

- [`DefaultProfile`](@ref)`(:es)` -- recommended. The published profile for a language, refitted
  to this project's corpus at the first [`index!`](@ref index!(::FullTextEngine)) call.
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
# One body for all three selectors, because which inverted file to build stopped being a
# difference between engine types when it became a field on the backend. `InvertedFile` and
# `TextInvertedFile` name the same backend; the former is kept because it is what a caller
# reaching for "a weighted inverted file" writes.
function _create_text_engine(selector::Type, distance, textmodel,
                             on_change::Union{Nothing,Function}, log_io::Union{Nothing,IO})
    spec = _require_textmodel(selector, textmodel)
    kind = selector === BM25InvertedFile ? BM25InvertedFile : TextInvertedFile
    dist = kind === BM25InvertedFile ? nothing : distance
    profile = _initial_profile(spec)
    index = profile === nothing ? nothing : _text_index(profile, kind, dist)
    backend = TextBackend(index, InvertedFileContext(; _engine_logging(on_change, log_io)...),
                          kind, dist)
    FullTextEngine(backend, profile, _deferred_fit(spec), String[],
                   ContextPool(InvertedFileContext()), Set{UInt32}(), ReadWriteLock())
end

create_engine(::Type{BM25InvertedFile}; distance, minrecall, textmodel::Union{Nothing,AbstractTextModelSpec}, on_change::Union{Nothing,Function}, log_io::Union{Nothing,IO}) =
    _create_text_engine(BM25InvertedFile, distance, textmodel, on_change, log_io)
create_engine(::Type{InvertedFile}; distance, minrecall, textmodel::Union{Nothing,AbstractTextModelSpec}, on_change::Union{Nothing,Function}, log_io::Union{Nothing,IO}) =
    _create_text_engine(InvertedFile, distance, textmodel, on_change, log_io)
create_engine(::Type{TextInvertedFile}; distance, minrecall, textmodel::Union{Nothing,AbstractTextModelSpec}, on_change::Union{Nothing,Function}, log_io::Union{Nothing,IO}) =
    _create_text_engine(TextInvertedFile, distance, textmodel, on_change, log_io)

"""
    create_engine(::Type{InvertedFile}; distance, dimension, on_change, log_io) -> SparseEngine

Creates a project of sparse vectors -- a `SimilaritySearch.InvertedFile` under `distance`, with
no vocabulary, no profile and no staged text. See [`SparseEngine`](@ref).

`dimension` is required and structural: an inverted file is a fixed array of posting lists, so
it has to be sized before the first item, and every `SparseItem`
appended must agree with it. It is not inferred from the first item on purpose -- a project
whose dimension depends on which item happened to arrive first is a project whose second batch
can fail for reasons the caller never stated.

This shares an index type with the text engine's weighted backend, which is why it is reached
through `create_sparse_engine` rather than by dispatching on `InvertedFile`: the same library
index serves two different kinds of project, and the kind is the caller's to name.
"""
function create_sparse_engine(; distance, dimension::Integer,
                              on_change::Union{Nothing,Function}, log_io::Union{Nothing,IO})
    dimension >= 1 || throw(ArgumentError("dimension must be at least 1; got $dimension"))
    ctx = InvertedFileContext(; _engine_logging(on_change, log_io)...)
    backend = SparseBackend(InvertedFile(Int(dimension), distance), ctx, distance, Int(dimension))
    SparseEngine(backend, ContextPool(InvertedFileContext()), Set{UInt32}(), ReadWriteLock())
end

"""
    backend_tag(backend) -> Symbol

Which backend a saved project had, as a symbol.

A symbol and not a type, and that is the whole reason `kind` is one too. The previous shape
persisted Julia types -- BM25Engine, ExhaustiveSearch -- into the project's own store, which
tied the on-disk format to the names in this file: this restructuring renamed every one of those
types, and any project written under the old names became unreadable by construction. Symbols
cut that tie. A rename is now a rename, and `restore_engine` maps a symbol it does not
recognise to an error naming the ones it does, rather than to a `MethodError` about a type the
reader has never heard of.
"""
backend_tag(::GraphBackend) = :graph
backend_tag(b::ExactBackend) = b.index isa ParallelExhaustiveSearch ? :parallel_exhaustive : :exhaustive
backend_tag(::SparseBackend) = :inverted_file
backend_tag(b::TextBackend) = b.kind === BM25InvertedFile ? :bm25 : :text_inverted_file

"""
    snapshot_state(engine::AbstractSearchEngine) -> NamedTuple

The full set of `engine`'s state that must survive a save (see `Persistence.EngineStore`)
-- `ctx`/`lock` are transient and rebuilt fresh by [`restore_engine`](@ref). The returned
`kind` field says what the project holds (`:dense`, `:sparse` or `:text`) and `backend` which
index held it (see [`backend_tag`](@ref)) -- two symbols rather than one Julia type, which is
what makes a saved project survive a rename of the types in this file. Together they are the
discriminator [`restore_engine`](@ref) dispatches on, via `Val{kind}`. Used for a one-shot
full save (e.g. right after `create_engine`, when there's nothing indexed yet); routine
mutations instead persist just the one or two fields they touched (see
[`extra_state_fields`](@ref) for the field lists a caller needs to read back *before* an
engine instance exists to call this on).

Only `DenseEngine{<:ExactBackend}` carries a plain `:index` -- for a brute-force index the whole
thing *is* a single value, so saving it is saving the field. No other backend has such a field to
save at all, and each carries instead what its own `build_*` function needs to replay its saved
insertion blocks:

- a graph-backed dense project: `distance`, plus `minrecall`/`opt_beamsearch` (see
  `SimilaritySearch.CallbackLog`/[`direct_neighbors`](@ref)/[`build_searchgraph`](@ref));
- a sparse project: `distance` and `dimension` -- structural, since an `InvertedFile` is a fixed
  array of posting lists and cannot be rebuilt without its size;
- a text project: `profile`, `fitspec` and `distance` (see
  [`invertedfile_objects`](@ref)/[`build_bm25invertedfile`](@ref)/[`build_textinvertedfile`](@ref)).

A text project saves one `profile` where two engines used to save a `voc`/`model` pair: a
`TextSearch.TextProfile` holds both, along with the artifacts and lineage neither of them
carried, so there is one value to write and no way for the two halves to be saved out of
step with each other.

No backend includes `staged`/`.db`'s raw items either, for the same reason: those
live in their own dedicated, incrementally-persisted store (a `SearchGraph`'s
`MMapMatrixDatabase` file, a text engine's `Persistence.StagedTextStore`), not a plain
`EngineStore` field -- rewriting the whole (potentially large, ever-growing) staged
sequence into `EngineStore` on every mutation is exactly the "single ever-growing blob"
this design avoids. A caller restoring any of them assembles `staged`/`vector_blocks`
into the `state` NamedTuple by hand from that dedicated store (see [`restore_engine`](@ref)'s
docstring) rather than getting it from `snapshot_state`.
"""
snapshot_state(engine::DenseEngine{GraphBackend}) =
    (kind=:dense, backend=:graph, distance=engine.backend.index.dist,
     minrecall=engine.backend.minrecall, opt_beamsearch=engine.backend.opt_beamsearch,
     deleted_ids=engine.deleted_ids)
snapshot_state(engine::DenseEngine{<:ExactBackend}) =
    (kind=:dense, backend=backend_tag(engine.backend), index=engine.backend.index,
     deleted_ids=engine.deleted_ids)
snapshot_state(engine::SparseEngine) =
    (kind=:sparse, backend=:inverted_file, distance=engine.backend.distance,
     dimension=engine.backend.dimension, deleted_ids=engine.deleted_ids)
snapshot_state(engine::FullTextEngine) =
    (kind=:text, backend=backend_tag(engine.backend), profile=engine.profile,
     fitspec=engine.fitspec, distance=engine.backend.distance, deleted_ids=engine.deleted_ids)

"""
    extra_state_fields(kind::Type) -> Tuple{Vararg{Symbol}}

The field names [`snapshot_state`](@ref) includes for a saved project *beyond* `:kind`,
`:backend`, `:deleted_ids` and `:index` -- the static list a loader needs before an engine
instance exists to call `snapshot_state` on.

Only the exact dense backends are covered, and they need nothing: their whole index is a single
`:index` field. Every other backend needs its own dedicated path instead -- a graph's
distance/minrecall/beam plus its replayed insertion blocks, a sparse project's
distance/dimension, a text project's profile and objects -- because none of them has a plain
`:index` field to read back at all (see `snapshot_state`'s docstring).
"""
extra_state_fields(::Symbol) = ()

"""
    restore_engine(state; on_change, log_io) -> AbstractSearchEngine

Rebuilds a concrete engine from `state` (as produced by [`snapshot_state`](@ref), or
assembled field-by-field via [`extra_state_fields`](@ref) from an `EngineStore`),
dispatching on `state.kind` to reconstruct exactly the engine type that was saved --
including, for a `DenseEngine{GraphBackend}`, the `minrecall` target it was growing toward, so a
reopened project keeps autotuning toward the same target instead of reverting to some
other default. `on_change`/`log_io` are as in [`create_engine`](@ref).

Only an exact dense project's `state` carries a plain `index`; every other kind carries what its
own `build_*` function needs instead:
- `DenseEngine{GraphBackend}`: `distance`, `vector_blocks` (`Persistence.load_dense_vector_blocks(dense_vectors_store)`),
  `load_neighbors` (typically `i -> Persistence.load_neighbors(adjacency_store, i)`),
  `graph_len` (`Persistence.load_field(store, :graph_len, 0)` -- may be `< length` of the
  restored `.db` if some staged vectors were never caught up by an explicit
  [`index!`](@ref index!(::DenseEngine{GraphBackend})) call before the project last closed) -- see
  [`build_searchgraph`](@ref).
- `FullTextEngine`: `profile`, `fitspec`, `object_blocks`
  (`Persistence.load_object_blocks(obj_store)`, or `nothing`/empty if `profile === nothing`,
  i.e. never trained), `staged` (every raw text ever staged, flattened from
  `Persistence.load_staged_text_blocks`, which can be longer than `object_blocks`'s total
  count if a backlog was still pending an [`index!`](@ref index!(::FullTextEngine)) call when the
  project last closed) -- see [`build_bm25invertedfile`](@ref).
  A `TextInvertedFile`-backed project carries the same fields and is rebuilt by
  [`build_textinvertedfile`](@ref) instead -- `state.backend` is what says which.
- `SparseEngine`: `distance`, `dimension`, and `object_blocks` -- the same encoded posting-list
  blocks a text project replays, minus the profile there is no vocabulary for.
"""
restore_engine(state; on_change::Union{Nothing,Function}, log_io::Union{Nothing,IO}) = restore_engine(Val(state.kind), state; on_change, log_io)

# Dispatch is on `Val(state.kind)` because `kind` is a symbol now rather than a Julia type (see
# [`backend_tag`](@ref)): the same dispatch, without the on-disk format naming the types in this
# file. An unrecognised kind lands on the fallback below, with a message naming the ones that
# exist -- which is what a project written by another version deserves instead of a `MethodError`
# about `Val{:whatever}`.
function restore_engine(::Val{:dense}, state; on_change::Union{Nothing,Function}, log_io::Union{Nothing,IO})
    if state.backend === :graph
        index = build_searchgraph(state.distance, state.vector_blocks, state.load_neighbors, state.graph_len)
        backend = GraphBackend(index, _searchgraph_context(state.minrecall, _engine_logging(on_change, log_io)),
                               state.minrecall, state.opt_beamsearch)
        return DenseEngine(backend, ContextPool(SearchGraphContext()), state.deleted_ids, ReadWriteLock())
    end
    state.backend in (:exhaustive, :parallel_exhaustive) ||
        error("unknown dense backend $(repr(state.backend)); expected :graph, :exhaustive or :parallel_exhaustive")
    backend = ExactBackend(state.index, GenericContext(; _engine_logging(on_change, log_io)...))
    DenseEngine(backend, ContextPool(GenericContext()), state.deleted_ids, ReadWriteLock())
end

function restore_engine(::Val{:sparse}, state; on_change::Union{Nothing,Function}, log_io::Union{Nothing,IO})
    ctx = InvertedFileContext(; _engine_logging(on_change, log_io)...)
    index = build_sparseinvertedfile(state.distance, state.dimension, state.object_blocks)
    backend = SparseBackend(index, ctx, state.distance, state.dimension)
    SparseEngine(backend, ContextPool(InvertedFileContext()), state.deleted_ids, ReadWriteLock())
end

function restore_engine(::Val{:text}, state; on_change::Union{Nothing,Function}, log_io::Union{Nothing,IO})
    profile = state.profile
    kind = state.backend === :bm25 ? BM25InvertedFile :
           state.backend === :text_inverted_file ? TextInvertedFile :
           error("unknown text backend $(repr(state.backend)); expected :bm25 or :text_inverted_file")
    index = profile === nothing ? nothing :
            kind === BM25InvertedFile ? build_bm25invertedfile(profile, state.object_blocks) :
                                        build_textinvertedfile(state.distance, profile, state.object_blocks)
    backend = TextBackend(index, InvertedFileContext(; _engine_logging(on_change, log_io)...),
                          kind, state.distance)
    FullTextEngine(backend, profile, state.fitspec, state.staged,
                   ContextPool(InvertedFileContext()), state.deleted_ids, ReadWriteLock())
end

restore_engine(::Val{K}, state; on_change, log_io) where {K} =
    error("unknown project kind $(repr(K)); this version restores :dense, :sparse and :text. A " *
          "project written before the engine kinds were restructured records a Julia type there " *
          "instead of a symbol, and cannot be read by this version.")



"""
    index!(engine::FullTextEngine)
    index!(engine::FullTextEngine)

Catches up encoding/indexing over whatever's been staged into `engine.staged` since the
last call (or since creation) -- the exact same contract as
[`index!`](@ref index!(::DenseEngine{GraphBackend})): idempotent, safe to call any number of
times, only ever processes the backlog (`(engine.backend.index === nothing ? 0 :
length(engine.backend.index))+1:length(engine.staged)`), and a no-op when nothing new is staged.

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

`add_item!`/`append_items!` on a `FullTextEngine` only ever stage raw
text into `engine.staged` -- exactly like `DenseEngine{GraphBackend}`, they do *not* make new
items searchable by themselves; [`search_live`](@ref) only ever sees items this has
processed. `DenseEngine{<:ExactBackend}` (`ExhaustiveSearch`/`ParallelExhaustiveSearch`) is the one
engine kind with no such split at all -- it always evaluates directly against `db`, so it
has no `index!` method of its own.

Errors if `engine.staged` is completely empty (nothing has ever been staged) -- mirrors
`index!(engine::DenseEngine{GraphBackend})`'s empty-`.db` error.
"""
function index!(engine::FullTextEngine)
    write_lock(engine.lock) do
        n = length(engine.staged)
        n == 0 && error("this text project has nothing staged yet -- add_item!/append_items! at least one item before calling index!")
        already = engine.backend.index === nothing ? 0 : length(engine.backend.index)
        if engine.profile === nothing
            profile = train_profile(engine.fitspec, engine.staged)
            engine.profile = profile
            engine.backend.index = _text_index(profile, engine.backend.kind, engine.backend.distance)
        end
        for i in already+1:n
            # The staged text goes in as text, whichever backend this is. Both take it and both
            # encode it their own way -- `BM25InvertedFile` into the bag its scorer reads,
            # `TextInvertedFile` into a weighted vector through the profile's model -- which is
            # why one loop serves both and why neither encoding is spelled out here. This module
            # used to do the encoding, in two methods that differed in nothing else.
            push_item!(engine.backend.index, engine.backend.ctx, engine.staged[i])
        end
    end
    return engine
end

# (the weighted text path merged into `index!(::FullTextEngine)` above)

"""
    current_beamsearch(engine::AbstractSearchEngine) -> Union{Nothing, BeamSearch}

The `BeamSearch` configuration currently installed on a `DenseEngine{GraphBackend}`'s index
(`nothing` for anything else — exact indices have no beam to configure, text indices
aren't dense at all). This is `SimilaritySearch.jl`'s own `BeamSearch()` default until
`calibrate!` has been called at least once (or until the growth autotuner has had a chance
to run), since construction no longer installs a calibrated `BeamSearch` synchronously.
"""
current_beamsearch(engine::DenseEngine{GraphBackend}) = read_lock(() -> engine.backend.index.algo[], engine.lock)
current_beamsearch(::AbstractSearchEngine) = nothing

"""
    calibrate!(engine::DenseEngine{GraphBackend}; levels=DEFAULT_MINRECALL_LEVELS, numqueries=64, ksearch=10, queries=nothing) -> OptBeamSearch

Runs `SimilaritySearch.jl`'s real `optimize_index!` hyperparameter sweep (PLAN.md §5.6 —
a `SearchModels`-driven stochastic search over `BeamSearchSpace`, not a hand-rolled one)
against `engine.backend.index` once per recall level in `levels`, storing each resulting
`BeamSearch` into `engine.backend.opt_beamsearch[level]` (and returning that table) so
[`search_live`](@ref) can later serve a per-request `minrecall` from calibrated
hyperparameters instead of a single one-size-fits-all default. Only defined for a
`DenseEngine{GraphBackend}`; errors for any other engine kind, which has no `BeamSearch` to
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
function calibrate!(engine::DenseEngine{GraphBackend}; levels=DEFAULT_MINRECALL_LEVELS, numqueries::Int=64, ksearch::Int=10, queries=nothing)
    Q = queries === nothing ? nothing : VectorDatabase([convert(Vector{Float32}, q) for q in queries])
    target_levels = levels isa Real ? Float32[levels] : Float32.(levels)

    write_lock(engine.lock) do
        for r in target_levels
            optimize_index!(engine.backend.index, engine.backend.ctx, MinRecall(r); queries=Q, numqueries, ksearch)
            engine.backend.opt_beamsearch[r] = engine.backend.index.algo[]
        end
    end

    return engine.backend.opt_beamsearch
end
calibrate!(::AbstractSearchEngine; kwargs...) = error("calibrate! only applies to a dense project on a SearchGraph backend -- nothing else has a BeamSearch to tune")

"""
    _nearest_beamsearch(engine::DenseEngine{GraphBackend}, minrecall::Real) -> BeamSearch

The `BeamSearch` calibrated for `minrecall` (see [`calibrate!`](@ref)), auto-calibrating
[`DEFAULT_MINRECALL_LEVELS`](@ref) first if `engine.backend.opt_beamsearch` is still empty. An
exact match is used if present; otherwise the closest calibrated level *at or below*
`minrecall` is used, falling back to the lowest calibrated level available if `minrecall`
sits below all of them.
"""
function _nearest_beamsearch(engine::DenseEngine{GraphBackend}, minrecall::Real)
    isempty(engine.backend.opt_beamsearch) && calibrate!(engine)
    target = Float32(minrecall)
    haskey(engine.backend.opt_beamsearch, target) && return engine.backend.opt_beamsearch[target]
    levels = keys(engine.backend.opt_beamsearch)
    below = Iterators.filter(<=(target), levels)
    key = isempty(below) ? minimum(levels) : maximum(below)
    return engine.backend.opt_beamsearch[key]
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
# index!(::DenseEngine{GraphBackend})) -- so a caller controls when the expensive step runs (a
# batch job, a cron, before the next round of searches) instead of paying it per item.
# `ExhaustiveSearch`/`ParallelExhaustiveSearch` (exact search, no graph to build) have no
# such split to make -- `push_item!` already *is* the whole insertion, cheap either way.
insert_dense!(index::SearchGraph, ctx, item) = push_item!(index.db, item)
insert_dense!(index::Union{ExhaustiveSearch,ParallelExhaustiveSearch}, ctx, item) = push_item!(index, ctx, item)
insert_dense!(index, ctx, item) = push_item!(index, ctx, item)

"""
    index!(engine::DenseEngine{GraphBackend})

Catches up the graph structure over whatever's been staged into `engine.backend.index.db` since
the last call (or since creation) -- extends `SimilaritySearch.index!` with the same
meaning the text engines' [`index!`](@ref index!(::FullTextEngine)) methods
give it: "do the expensive part explicitly, now." Unlike those, this is *idempotent* and
can be called any number of times: `SimilaritySearch.index!(idx, ctx)` itself already
starts from `length(idx) + 1` (the current graph-indexed count) up through
`length(database(idx))` (everything staged so far, see `insert_dense!`), so
calling this repeatedly as more vectors accumulate only ever processes the backlog, never
redoing already-indexed work, and calling it with nothing new staged is a cheap no-op.

`add_item!`/`append_items!` on a `DenseEngine{GraphBackend}` only ever stage raw vectors into
`engine.backend.index.db` -- they do *not* make new items searchable by themselves anymore (unlike
`DenseEngine{<:ExactBackend}`/`FullTextEngine`, which still index synchronously on every
`add_item!`). Call this explicitly -- interactively, on a schedule, whatever fits the
caller -- to actually build graph connections for the backlog; [`search_live`](@ref) only
ever sees items this has processed. This is also what unlocks batch-shaped insertion:
staging is cheap and safe to do many times in a row without paying graph-construction cost
until the caller actually wants it.

Errors if `engine.backend.index.db` is completely empty (nothing has ever been staged) -- mirrors
`SimilaritySearch.index!`'s own `@assert n > 0`.
"""
function index!(engine::DenseEngine{GraphBackend})
    write_lock(engine.lock) do
        n = length(database(engine.backend.index))
        n == 0 && error("DenseEngine{GraphBackend} has nothing staged yet -- add_item!/append_items! at least one vector before calling index!")
        SimilaritySearch.index!(engine.backend.index, engine.backend.ctx)
    end
    return engine
end

index!(engine::DenseEngine{<:ExactBackend}) = engine
index!(engine::SparseEngine) = engine

"""
    add_item!(engine::AbstractSearchEngine, item)

Adds a single item to the index. Thread-safe wrapper.
For a `DenseEngine{GraphBackend}`, this only *stages* `item` (see `insert_dense!`) -- it
does not become searchable until an explicit [`index!`](@ref index!(::DenseEngine{GraphBackend}))
call. `FullTextEngine` have the exact same split: `item` is raw text,
staged into `engine.staged` -- always allowed, whether or not the engine has been trained
yet -- and does not get encoded/indexed until an explicit
[`index!`](@ref index!(::FullTextEngine)) call. `DenseEngine{<:ExactBackend}`
(`ExhaustiveSearch`/`ParallelExhaustiveSearch`) is the one engine kind that still indexes
synchronously, with no staging step at all.
"""
function add_item!(engine::DenseEngine, item)
    write_lock(engine.lock) do
        insert_dense!(engine.backend.index, engine.backend.ctx, item)
    end
end

# An inverted file indexes on the push, so a sparse project has no staging split to speak of --
# like the exact dense backends and unlike the text engines, whose split exists because text
# cannot be encoded before a vocabulary exists. `index!` below is correspondingly a no-op.
function add_item!(engine::SparseEngine, item)
    write_lock(engine.lock) do
        push_item!(engine.backend.index, engine.backend.ctx, item)
    end
end

function add_item!(engine::FullTextEngine, item)
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
`SearchGraph`'s or a `DenseEngine{<:ExactBackend}`'s own `database`. A dense payload is materialized into a
fresh `Vector{Float32}` rather than handed out as a view, because the view aliases the index's
live storage and the caller has no way to know that.
"""
function stored_payload(engine::FullTextEngine, id::Integer)
    1 <= id <= length(engine.staged) || return nothing
    engine.staged[id]
end

function stored_payload(engine::DenseEngine, id::Integer)
    db = database(engine.backend.index)
    1 <= id <= length(db) || return nothing
    convert(Vector{Float32}, db[id])
end

# Materialized into a real `SparseVector` rather than handed out as the view `database` returns:
# that view aliases the inverted file's own packed storage, and a caller has no way to know it.
function stored_payload(engine::SparseEngine, id::Integer)
    db = database(engine.backend.index)
    1 <= id <= length(db) || return nothing
    v = db[id]
    SparseVector{Float32,Int32}(engine.backend.dimension,
                                convert(Vector{Int32}, collect(v.nzind)),
                                convert(Vector{Float32}, collect(v.nzval)))
end

"""
    resolve_query(engine, text::AbstractString, policy::QueryPolicy) -> TextSearch.ResolvedQuery

Runs `text` through the library's query pipeline -- the one living on this engine's index --
under `policy`, and hands back both what to search for and what correction did to each spelling
that was typed.

Exists for [`ftexplain`](@ref SimilaritySearchEngine.ftexplain), and for nothing else: [`search_live`](@ref) does not call it,
because `TextSearch`'s own `search` takes the same `policy` keyword and resolves the query
itself. Correcting a query is a substitution the person who typed it is owed a report of, which
is what `TextSearch.explain(rq.resolution)` renders and what `QueryPolicy(correction=:off)`
undoes.

This module used to assemble the pipeline here: tokenize, resolve, rebuild the term list,
restrict the network to `expansion_sources`, weight the neighbours -- and cache its own variant
map to afford it. All of that is `query_tokens` now, and the map lives once, inside the index,
where the profile-taking constructor put it.

Errors if `engine` has no profile yet -- nothing indexed, so no vocabulary to resolve against.
"""
function resolve_query(engine::FullTextEngine, text::AbstractString, policy::QueryPolicy)
    engine.profile === nothing && error("this text engine has not been trained yet -- index! at least one staged item, or create it with a profile, before resolving a query")
    voc = engine.profile.model.voc
    TextSearch.query_tokens(voc, text, engine.backend.index.query; policy)
end

"""
    search_live(engine::AbstractSearchEngine, query, k::Int; bs_override, minrecall, policy) -> (id=..., dist=..., deleted=...)

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
  instead of `engine.backend.index.algo[]`'s calibrated default. Only meaningful for a
  `DenseEngine{GraphBackend}` — accepted and ignored on every other engine kind, which has no
  `BeamSearch` to override in the first place. Takes priority over `minrecall` below.
- `minrecall::Union{Nothing, Real}`: search at (approximately) this target recall instead
  of `engine.backend.index.algo[]`'s current default, by looking up the matching calibrated
  `BeamSearch` in `engine.backend.opt_beamsearch` (see [`calibrate!`](@ref)/[`_nearest_beamsearch`](@ref)
  — auto-calibrating [`DEFAULT_MINRECALL_LEVELS`](@ref) first if that table is still empty).
  Only meaningful for a `DenseEngine{GraphBackend}`; accepted and ignored on every other engine
  kind, which has no calibrated `BeamSearch` table to consult.
- `policy::QueryPolicy`: how to treat a *text* query -- whether to correct its spelling
  against the vocabulary and whether to widen it with the profile's expansion network (see
  [`resolve_query`](@ref)). Only meaningful for `FullTextEngine`; accepted
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
    `DenseEngine{GraphBackend}`'s own `minrecall`-driven auto-`calibrate!` below has to be resolved
    *before* the read lock is acquired, not nested inside it.

    That guarantees the *index data* (`index.adj`/`index.db`) tolerates concurrent reads,
    but not by itself enough: `engine.backend.ctx` is a *mutable, reused* scratch context (beam
    buffers, visited-vertices state, per-batch cost counters, ...), and two concurrent
    searches sharing that same object corrupt each other's scratch space -- confirmed
    directly, a segfault inside `SearchGraph`'s own `beamsearch_inner_beam` from two
    threads racing on shared visited-vertices state via a shared `ctx`. So every
    `search_live` method borrows a private context from [`ContextPool`](@ref) via
    [`checkout!`](@ref)/[`checkin!`](@ref) instead of reusing `engine.backend.ctx` (which stays
    reserved for insertion, itself already exclusive via `write_lock`, so sharing it
    *there* is fine) or allocating a brand-new context on every single call.
"""
function search_live(engine::DenseEngine{GraphBackend}, query, k::Int; bs_override, minrecall, policy)
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
                vstate = SimilaritySearch.getvstate(length(engine.backend.index), ctx)
                search(bs, engine.backend.index, ctx, query, res, engine.backend.index.hints, vstate)
            else
                search(engine.backend.index, ctx, query, res)
            end
            evals = SimilaritySearch.distance_evaluations(ctx, snap)
            return _collect_live(engine, res, evals)
        finally
            checkin!(engine.search_ctx_pool, ctx)
        end
    end
end

function search_live(engine::Union{DenseEngine{<:ExactBackend}, SparseEngine}, query, k::Int; bs_override, minrecall, policy)
    read_lock(engine.lock) do
        ctx = checkout!(engine.search_ctx_pool)
        try
            snap = copy(ctx.costdists)
            res = knnqueue(KnnSorted, max(k, 1))
            search(engine.backend.index, ctx, query, res)
            evals = SimilaritySearch.distance_evaluations(ctx, snap)
            return _collect_live(engine, res, evals)
        finally
            checkin!(engine.search_ctx_pool, ctx)
        end
    end
end

function search_live(engine::FullTextEngine, query, k::Int; bs_override, minrecall, policy::QueryPolicy)
    read_lock(engine.lock) do
        profile = engine.profile
        profile === nothing && return (id=Int32[], dist=Float32[], deleted=Bool[], distance_evaluations=0)
        ctx = checkout!(engine.search_ctx_pool)
        try
            snap = copy(ctx.costdists)
            res = knnqueue(KnnSorted, max(k, 1))
            # The library resolves, corrects, expands and encodes. `policy` travels with the
            # call while the variant map and the expansion network stay on the index, derived
            # once by the profile-taking constructor. This module used to do all four steps
            # itself, and keep a second variant map, only to be able to vary the policy.
            search(engine.backend.index, ctx, query, res; policy)
            evals = SimilaritySearch.distance_evaluations(ctx, snap)
            return _collect_live(engine, res, evals)
        finally
            checkin!(engine.search_ctx_pool, ctx)
        end
    end
end

# (the weighted text path merged into `search_live(::FullTextEngine)` above)

function _collect_live(engine::AbstractSearchEngine, res, evals)
    ids = view(res.ids, res.sp:res.ep)
    dists = view(res.dists, res.sp:res.ep)
    deleted = [id in engine.deleted_ids for id in ids]
    return (id=ids, dist=dists, deleted=deleted, distance_evaluations=evals)
end

end # module
