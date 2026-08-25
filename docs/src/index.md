```@meta
CurrentModule = SimilaritySearchEngine
```

# SimilaritySearchEngine.jl

`SimilaritySearchEngine.jl` is an embedded, high-performance similarity search library for Julia. It provides persistent, transactional management for collections of dense vectors, sparse representations, and full-text documents. The storage layer is backed by [SimilaritySearch.jl](https://github.com/sadit/SimilaritySearch.jl), [TextSearch.jl](https://github.com/sadit/TextSearch.jl), and [RocksDB.jl](https://github.com/sadit/RocksDB.jl).

The library executes directly within the host Julia process without requiring external HTTP services, background daemons, or separate IPC brokers.

## Core Capabilities

- **Project Lifecycle Management**: Structured management of project storage via `create_project`, `open_project`, and `close_project!`.
- **Decoupled Architecture**: Clear distinction between the stored payload type (`DenseEngine`, `SparseEngine`, `FullTextEngine`) and the underlying indexing backend (`SearchGraph`, `ExhaustiveSearch`, `InvertedFile`, `BM25InvertedFile`).
- **Dense Vector Search**: Approximate nearest neighbor search via `SearchGraph` (supporting automatic hyperparameter recall tuning) or exact linear scanning via `ExhaustiveSearch` and `ParallelExhaustiveSearch`.
- **Sparse Vector Search**: Native inverted index retrieval (`InvertedFile`) over user-provided sparse representations, evaluated under cosine similarity or set distance metrics (`Dist.Sets.*`).
- **Full-Text Retrieval**: Integrated text preprocessing, BM25 scoring, and TF-IDF inverted files configured via linguistic profiles (`TextProfile`).
- **Lifecycle Decoupling (Staging vs. Indexing)**: Incremental staging of documents via `append_items!` with deferred, batch-oriented index construction via `index!`.
- **Logical Deletions**: Non-destructive soft deletes (`delete_item!`) that preserve index integrity while marking candidates in query responses.
- **Global Metric Operations**: In-process execution of all-pairs nearest neighbors (`allknn`), diverse sampling (`fft`), closest pair discovery (`closestpairs`), and cross-dataset closest pairs (`bichromatic_kclosestpairs`).
- **Granular Column-Family Persistence**: Fine-grained persistence where mutations update only the affected RocksDB column families or memory-mapped files.

---

## Installation

Install `SimilaritySearchEngine.jl` in development mode directly from its repository:

```julia
using Pkg
Pkg.develop(path="path/to/SimilaritySearchEngine.jl")
```

---

## Documentation Structure

- **[API Reference](api.md)**: Generated documentation for all exported types, interfaces, and function signatures.
- **[Manual Home](https://sadit.github.io/SimilaritySearchEngine.jl/manual/)**: Architectural overview and feature summary.
- **[Architecture](https://sadit.github.io/SimilaritySearchEngine.jl/manual/architecture.html)**: Comprehensive specification of the submodules, type hierarchies, concurrency locks, and on-disk storage layout.
- **[Getting Started Tutorial](https://sadit.github.io/SimilaritySearchEngine.jl/manual/tutorials/getting-started.html)**: End-to-end tutorial covering dense search, sparse retrieval, BM25 text search, filtering, calibration, and whole-dataset operations.
- **[Multilingual Paragraph Search](https://sadit.github.io/SimilaritySearchEngine.jl/manual/tutorials/paragraph-search.html)**: Large-scale case study evaluating full-text retrieval across multilingual Project Gutenberg corpora and Wikipedia.
