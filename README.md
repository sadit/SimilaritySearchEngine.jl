# SimilaritySearchEngine.jl

`SimilaritySearchEngine.jl` is an embedded, no-HTTP-required Julia API for running
similarity search *projects*: named, persistent collections of dense vectors or text
documents backed by [SimilaritySearch.jl](https://github.com/sadit/SimilaritySearch.jl),
[TextSearch.jl](https://github.com/sadit/TextSearch.jl), and
[RocksDB.jl](https://github.com/sadit/RocksDB.jl). A script does `using
SimilaritySearchEngine` and calls its functions directly — no running server, no CLI
subprocess, no job queue. The trust boundary is just OS file permissions on a working
directory, the same as any other embedded database.

## Key features

- **Dense vector search** over a `SearchGraph` (an approximate, self-tuning graph index),
  or an exact `ExhaustiveSearch`/`ParallelExhaustiveSearch` — insertion (`append_items!`)
  and graph-building (`index!`) are separate, explicit steps for `SearchGraph`, so a
  caller controls when the expensive graph-connection cost is paid.
- **Full-text search** over a `BM25InvertedFile` or a TF-IDF-weighted `TextInvertedFile`,
  sharing the exact same `append_items!`(stage)/`index!`(build)/`ftsearch` surface as the
  dense case — the first `index!` call on a text project also fits its `TextProfile`
  from whatever's been staged so far, so there's no separate training step to call first.
  `ExhaustiveSearch`/`ParallelExhaustiveSearch` are the one exception, with no staging
  split at all: items are searchable the instant `append_items!` returns.
- **A text model you state, never one you get by default**: creating a text project requires a
  `textmodel`. Prefer `DefaultProfile(:es)` — the published profile for a language, refitted to
  your corpus at the first `index!` call, so your weights are calibrated over millions of
  paragraphs and you inherit a stopword set, a lemma map and a query-expansion network instead
  of deriving them (`refit=false` keeps the base's full language-wide vocabulary instead of
  narrowing to yours; the docstring has the measured trade). `BaseProfile(p)`
  uses a profile you already have. `FitFromCorpus(TextConfig(language=:es))` is the explicit
  "I have no profile" escape: it delegates to `TextSearch.fit_profile`, reads at most
  `max_documents` of your corpus to bound the cost, and gives you a correspondingly thin
  vocabulary — every term appended later that the sample never held is dropped, silently and
  forever. Omitting `textmodel` is an error because nothing about the project afterwards
  reveals that the choice was made by omission. Queries then
  travel with a `QueryPolicy`: whether to correct their spelling against the vocabulary
  (`ftexplain` reports what was corrected, `QueryPolicy(correction=:off)` undoes it) and
  whether to widen them with the expansion network.
- **Typed in, typed out**: items are `DenseItem`/`TextItem` values carrying their own
  `doc_id`, `keywords`, `refs` and `meta`, and results are `SearchResult`/`StoredItem`/
  `ExistsResult` — the library never takes a JSON-shaped dictionary and picks it apart by key
  name, and never answers with one. `meta` stays a free-form `Dict{String,Any}` because it is
  free-form by definition, and `get_raw_meta` is the single raw path, reserved for an HTTP
  layer forwarding stored bytes it never inspects.
- **Soft deletes**: `delete_item!` marks a document as logically deleted without touching
  the underlying index; search reports a `deleted` marker per candidate instead of
  silently hiding or backfilling it.
- **Recall calibration**: `calibrate!` runs `SimilaritySearch.jl`'s real hyperparameter
  search so later queries can ask for `minrecall=0.95` (or any other target) and get back
  a `BeamSearch` tuned for it.
- **Whole-dataset operations** — `allknn`, `fft` (farthest-first diverse sampling),
  `closestpairs` (near-duplicate detection), and `bichromatic_kclosestpairs` (closest
  pairs across two collections) — all run synchronously, in-process.
- **Incremental, per-field persistence**: every mutation writes only the storage it
  actually touched — dense vectors go straight into a memory-mapped file, a soft delete
  never rewrites the index, and routine inserts never rewrite the whole engine state.

## Quick start

```julia
using Pkg
Pkg.develop(path="path/to/SimilaritySearchEngine.jl")

using SimilaritySearchEngine
using SimilaritySearch: SearchGraph

workdir = mktempdir()
h = create_project(workdir, "demo"; index_type=SearchGraph, minrecall=0.9)

# typed items in: this library never takes a JSON-shaped dictionary and picks it apart
append_items!(h, [DenseItem(rand(Float32, 32); doc_id="item-$i") for i in 1:1000])
index!(h)  # build graph connections for everything staged so far

results = search(h, rand(Float32, 32), 5)   # Vector{SearchResult}
for r in results
    println(r.doc_id, " => distance ", r.distance)
end

close_project!(h)
```

See the [tutorial](manual/tutorials/getting-started.qmd) for a full walkthrough — dense
search with LSI-derived embeddings, BM25/TF-IDF text search, soft deletes, filtering,
calibration, closing/reopening, and the whole-dataset operations — using a real novel's
worth of paragraphs as a running example.

## Installation

Not registered yet — develop it from a local checkout (or a git URL) alongside its own
dependencies:

```julia
using Pkg
Pkg.develop(path="path/to/SimilaritySearchEngine.jl")
```

## Documentation

Published at **<https://sadit.github.io/SimilaritySearchEngine.jl/>** — the
[API reference](https://sadit.github.io/SimilaritySearchEngine.jl/dev/) and the
[manual](https://sadit.github.io/SimilaritySearchEngine.jl/manual/) (architecture and
tutorials). `./publish-docs.sh` rebuilds and republishes both; it is a script rather than a
CI workflow because this package builds only against unreleased local checkouts of
SimilaritySearch and TextSearch, which no runner can resolve.

- **[Architecture](manual/architecture.qmd)** — the module map, the engine type
  hierarchy, the staging-vs-indexing split, the concurrency model, and the persistence
  layout, each with a diagram. Render the site with `quarto render manual` / `quarto
  preview manual` (requires [Quarto](https://quarto.org/)).
- **[Tutorial](manual/tutorials/getting-started.qmd)** — the hands-on walkthrough linked
  above.
- **API reference** — generated from the source docstrings with
  [Documenter.jl](https://documenter.juliadocs.org/), in `docs/`: `julia --project=docs
  docs/make.jl` builds it to `docs/build/`.

## Where this fits

This package covers project lifecycle (`create_project`/`open_project`/`close_project!`),
core CRUD, calibration, and the whole-dataset operations above — every function here
mirrors an existing code path in the sibling `SimilaritySearchServer` package (same
on-disk layout, same insertion/search/persistence semantics), so a project touched
through this embedded API stays fully interoperable with `SimilaritySearchServer`'s CLI
and HTTP server. Development happens here first, at the engine level, and is ported to
`SimilaritySearchServer` afterwards (see `DEVELOPMENT_STRATEGY.md`).
