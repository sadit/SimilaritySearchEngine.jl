module SimilaritySearchEngine

using Avro
using JSON3
using SimilaritySearch
using TextSearch
using RocksDB
# For the sparse half of the public surface: a sparse project's items and its queries are
# `SparseVector{Float32,Int32}`, so the names have to be in scope where that surface is defined.
using SparseArrays: SparseVector, AbstractSparseVector, sparsevec

# Extracted verbatim from SimilaritySearchServer.jl (PLAN.md §8.5, chunk 22) -- these four
# modules were already the real shared engine both `similarity-search` (CLI) and
# `similarity-search-serve` (HTTP) call into; they never depended on anything else in that
# package (Jobs/Cursors/Tokens/Telemetry/Executors/Server are HTTP/CLI-specific glue that
# stayed behind). Schema before Project (Project needs it); IndexEngine/Persistence are
# independent of both and of each other.
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
using .Project: ProjectManager, put_metadata!, get_metadata, get_meta, get_raw_meta, find_by_doc_id, generate_id
using .IndexEngine

# The friendly, no-HTTP-required embedded API (PLAN.md §8.5) built on top of the four
# modules above -- a script does `using SimilaritySearchEngine` and calls these directly,
# with no Job/token/HTTP surface at all (the trust boundary here is OS file permissions on
# workdir, same as the CLI).
include("embedded.jl")

export EmbeddedEngine, create_project, open_project, close_project!, append_items!, index!,
       search, ftsearch, ftexplain, delete_item!, fetch_items, exists, calibrate!, allknn,
       fft, dnet, neardup, closestpairs, bichromatic_kclosestpairs,
       get_metadata, get_meta, get_raw_meta

# The typed data contract: what goes in, what comes back. This package does not take a
# JSON-shaped dictionary as an item and pick it apart, and does not answer with one either --
# see `Schema.AbstractItem`. `get_raw_meta` above is the single exception, reserved for an HTTP
# layer forwarding stored metadata bytes it never inspects.
export AbstractItem, DenseItem, SparseItem, TextItem, MetadataRecord, StoredItem, payload
export SearchResult, ExistsResult, KnnRow, FFTResult, CenterSelectionResult, NearDupResult

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

end # module
