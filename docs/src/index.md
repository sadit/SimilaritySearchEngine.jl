```@meta
CurrentModule = SimilaritySearchEngine
```

# SimilaritySearchEngine

`SimilaritySearchEngine.jl` is an embedded, no-HTTP-required Julia API for running
similarity search *projects*: named, persistent collections of dense vectors or text
documents backed by [SimilaritySearch.jl](https://github.com/sadit/SimilaritySearch.jl),
[TextSearch.jl](https://github.com/sadit/TextSearch.jl), and
[RocksDB.jl](https://github.com/sadit/RocksDB.jl). A script does `using
SimilaritySearchEngine` and calls its functions directly — no running server, no CLI
subprocess, no job queue.

It covers project lifecycle (`create_project`/`open_project`/`close_project!`), core CRUD
(`append_items!`/`search`/`ftsearch`/`delete_item!`/`fetch_items`/`exists`), recall
calibration (`calibrate!`), and one representative heavy operation (`allknn`) — every
function here mirrors an existing code path in the sibling `SimilaritySearchServer`
package, so a project touched through this embedded API stays fully interoperable with
its CLI and HTTP server.

# Installing

Not registered yet — develop it directly from a checkout:

```julia
using Pkg
Pkg.develop(path="path/to/SimilaritySearchEngine.jl")
```

# Where to read what

This site is the **reference**: the [API](api.md) page, generated from the source
docstrings, is every exported function and type with the reasoning behind it.

Everything narrative lives in the **manual**, a separate Quarto site published alongside
this one:

- **[Manual home](https://sadit.github.io/SimilaritySearchEngine.jl/manual/)** — what the package gives you, in one page.
- **[Architecture](https://sadit.github.io/SimilaritySearchEngine.jl/manual/architecture.html)** — the module map, the engine type
  hierarchy, the staging-vs-indexing split, the concurrency model and the on-disk layout,
  each with a diagram.
- **[Getting started](https://sadit.github.io/SimilaritySearchEngine.jl/manual/tutorials/getting-started.html)** — the smaller tour,
  on paragraphs from one novel: dense vector search with LSI-derived embeddings, full-text
  search with BM25 and TF-IDF, soft deletes, filtering, recall calibration, close/reopen,
  and the whole-dataset operations.
- **[Paragraph search over books and Wikipedia](https://sadit.github.io/SimilaritySearchEngine.jl/manual/tutorials/paragraph-search.html)**
  — the larger tour, on four real corpora in three languages (100 Project Gutenberg books
  each in English, Spanish and Portuguese, plus 1000 Wikipedia articles, 827,638 paragraphs
  in all): fitting and loading text profiles, correcting a query against the vocabulary,
  expanding it through a profile's network, and what building the whole thing cost. Its
  numbers and search results are computed when the page is rendered, not typed in.

The manual's sources are the `.qmd` files under `manual/`; `quarto preview manual` serves
them locally.
