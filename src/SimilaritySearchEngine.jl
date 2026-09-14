module SimilaritySearchEngine

using Avro
using JSON3
using SimilaritySearch
using TextSearch
using RocksDB
# For the sparse half of the public surface: a sparse project's items and its queries are
# `SparseVector{Float32,Int32}`, so the names have to be in scope where that surface is defined.
using SparseArrays: SparseArrays, SparseVector, AbstractSparseVector, sparsevec

# Extracted verbatim from SimilaritySearchServer.jl (PLAN.md §8.5, chunk 22) -- these four
# modules were already the real shared engine both `similarity-search` (CLI) and
# `similarity-search-serve` (HTTP) call into; they never depended on anything else in that
# package (Jobs/Cursors/Tokens/Telemetry/Executors/Server are HTTP/CLI-specific glue that
# stayed behind). Schema before Project (Project needs it); IndexEngine/Persistence are
# independent of both and of each other.
# Errors first: every submodule below raises them, and a consumer maps them (see
# DEVELOPMENT_STRATEGY.md, "Errors: typed in the engine, mapped at each boundary").
include("errors.jl")
using .Errors
include("schema.jl")
include("project.jl")
include("index_engine.jl")
include("persistence.jl")

using .Schema
using .Persistence
# open_project/close_project deliberately excluded: embedded.jl below defines its own
# EmbeddedEngine-returning open_project (identical (String, String) positional signature to
# Project.open_project's 2-arg form) and a distinctly-named close_project!. Bringing
# Project's raw open_project bare into this same scope would make embedded.jl's unqualified
# `function open_project(...)` silently extend/overwrite Project.open_project itself --
# corrupting the one every other qualified call site (`Project.open_project(...)` in
# cli_handlers.jl/server.jl) actually depends on, since Julia methods are owned by the
# generic function they extend, not by the module doing the extending. embedded.jl
# references `Project.open_project`/`Project.close_project` qualified instead.
using .Project: ProjectManager, put_metadata!, get_metadata, get_meta, get_raw_meta, find_by_doc_id, find_all_by_doc_id, generate_id
using .IndexEngine

# The friendly, no-HTTP-required embedded API (PLAN.md §8.5) built on top of the four
# modules above -- a script does `using SimilaritySearchEngine` and calls these directly,
# with no Job/token/HTTP surface at all (the trust boundary here is OS file permissions on
# workdir, same as the CLI).
include("embedded.jl")

export EmbeddedEngine, create_project, open_project, close_project!, compact_project!,
       append_items!, index!,
       search, searchbatch, ftsearch, ftexplain, delete_item!, fetch_items, exists, calibrate!,
       allknn, fft, dnet, neardup, closestpairs, bichromatic_kclosestpairs,
       get_metadata, get_meta, get_raw_meta

# The typed data contract: what goes in, what comes back. This package does not take a
# JSON-shaped dictionary as an item and pick it apart, and does not answer with one either --
# see `Schema.AbstractItem`. `get_raw_meta` above is the single exception, reserved for an HTTP
# layer forwarding stored metadata bytes it never inspects.
export AbstractItem, DenseItem, SparseItem, TextItem, MetadataRecord, StoredItem, payload
export SearchResult, SearchStats, ExistsResult, KnnRow, FFTResult, CenterSelectionResult, NearDupResult

# Re-exported from TextSearch.jl, not defined here: they are the vocabulary a caller needs to
# say anything to a *text* project -- `textmodel=FitFromCorpus(TextConfig(language=:es))`,
# `textmodel=BaseProfile(load_profile("wiki20231101-es.zip"))`,
# `ftsearch(h, q; policy=QueryPolicy(correction=:off))`. Re-exporting them keeps a script that
# only ever does `using SimilaritySearchEngine` from needing a second `using` line to reach the
# arguments this package's own API asks for.
export TextConfig, TextProfile, QueryPolicy, load_profile, save_profile,
       download_profile, list_remote_profiles

# Defined in the IndexEngine submodule and re-exported here (embedded.jl adds the
# `EmbeddedEngine` method to `text_profile` by extension, so it is one generic function, not a
# package-level copy of a submodule one). `fit_profile` is TextSearch's, which IndexEngine adds
# a `FitFromCorpus` method to -- also one function, for the same reason.
export text_profile, fit_profile
export meta_schema, declare_meta_schema!, MetaSchema, MetaField

# The text-model decision a text project is created with: `create_project` refuses to guess,
# so these two are part of the minimum vocabulary for creating one at all.
export AbstractTextModelSpec, BaseProfile, DefaultProfile, FitFromCorpus
export DEFAULT_PROFILE_NICKNAMES, default_profile_path, train_profile

# What kind of project to create, and which index to hold it in. `create_project` takes both as
# types, so both have to be nameable by a caller who wrote one `using` line: `engine=` says what
# the project indexes (dense vectors, sparse vectors, text) and `backend=` which index does it.
# `BACKENDS` is the table of legal pairings, and it is exported because it is also the answer to
# "what else could I have passed"; `default_backend` says what an omitted `backend=` becomes.
export DenseEngine, SparseEngine, FullTextEngine
export BACKENDS, default_backend

# The backend types themselves, re-exported from SimilaritySearch.jl and TextSearch.jl for the
# same reason `TextConfig` is: they are values this package's own API asks for, and a caller who
# wrote `using SimilaritySearchEngine` should be able to write `backend=BM25InvertedFile`
# without first having to know which of the two libraries that name comes from.
export SearchGraph, ExhaustiveSearch, ParallelExhaustiveSearch, InvertedFile
export BM25InvertedFile, TextInvertedFile

# For asking an *open* project what it holds -- which a Julia caller needs, because the item
# type it must build follows from it (`:dense` takes DenseItem, `:sparse` SparseItem, `:text`
# TextItem) and `open_project` is free to be handed a directory whose kind it did not choose.
export payload_kind

# The error contract, re-exported so a caller writing `catch e; e isa NotFound` does not have
# to reach into a submodule for the name.
export EngineError, InvalidRequest, NotFound, ConflictingState, StorageFailure
export PayloadMismatch, WrongDimension, UnknownBackend, InvalidOption, UnsupportedOperation
export ProfileNotInstalled, PendingBacklog, NothingStaged, NoIndex, NotTrained, EmptyProject
export CorruptedStorage

# ---------------------------------------------------------------------------------------
# Precompilation workload.
#
# Without this, the first `create_project`/`append_items!`/`index!`/`search` cycle in a fresh
# process spent ~11.7s compiling (measured 2026-09-10; `using` itself was 1.3s of that
# 13s total). That cost is paid by every script and every CLI invocation, and none of it is
# work: the methods are the same ones the second call reuses for free.
#
# All three payload kinds, deliberately. They share almost no specialization below
# `create_project` -- a dense project compiles `SearchGraph` insertion and `MMapMatrixDatabase`
# flushing, a text one the whole TextSearch profile fit and BM25 query pipeline, a sparse one
# the inverted file over caller-encoded vectors -- so covering one leaves the others paying
# nearly the full bill on first use. The corpora are tiny (a few dozen items): what is being
# compiled is the code path, and the path does not get wider with more data.
#
# Everything runs against a temporary directory that is removed afterwards, and the whole
# block is wrapped so that a failure here degrades to "slower first call", never to a package
# that fails to precompile.
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    dim = 8
    vectors = [Float32[sin(i * j) for j in 1:dim] for i in 1:24]
    texts = [isodd(i) ? "the cat sat on the mat number $i" : "a dog barked at the moon $i"
             for i in 1:24]
    @compile_workload begin
        try
            workdir = mktempdir()
            # The engine's default reporter narrates every insertion, and a package that
            # narrates its own precompilation into the user's terminal is just noise. Both
            # streams: the progress lines go to stderr, the rest to stdout.
            try
                redirect_stdout(devnull) do
                redirect_stderr(devnull) do
                h = create_project(workdir, "pc_dense"; engine=DenseEngine, backend=SearchGraph)
                append_items!(h, [DenseItem(v; doc_id="d$i") for (i, v) in enumerate(vectors)])
                index!(h)
                search(h, vectors[1], 3)
                search(h, vectors[1], 3; filter=(record, meta) -> true)
                search(h, vectors[1:2], 3)
                searchbatch(h, vectors[1:2], 3)
                fetch_items(h, ["d1", 2])
                exists(h, ["d1"])
                delete_item!(h, 3)
                close_project!(h)
                close_project!(open_project(workdir, "pc_dense"))

                hs = create_project(workdir, "pc_sparse"; engine=SparseEngine, dimension=dim)
                append_items!(hs, [SparseItem(sparsevec(Int32[1, 3], Float32[1.0, 0.5], dim); doc_id="s$i")
                                   for i in 1:8])
                search(hs, sparsevec(Int32[1, 3], Float32[1.0, 0.5], dim), 3)
                close_project!(hs)

                ht = create_project(workdir, "pc_text"; engine=FullTextEngine,
                                    backend=BM25InvertedFile, textmodel=FitFromCorpus())
                append_items!(ht, [TextItem(t; doc_id="t$i") for (i, t) in enumerate(texts)])
                index!(ht)
                ftsearch(ht, "cat mat", 3)
                close_project!(ht)
            end
            end
            finally
                rm(workdir; force=true, recursive=true)
            end
        catch
            # A workload that cannot run is a missed optimization, not a broken package.
        end
    end
end

end # module
