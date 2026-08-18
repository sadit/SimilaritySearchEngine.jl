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

# Getting started

See the [API](api.md) page for the full function/type reference, generated directly from
the source docstrings.

For a narrative introduction — the architecture (with diagrams) and a hands-on tutorial —
see this package's Quarto site in `manual/` (`quarto render manual` / `quarto preview
manual`).
