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
- **Persisted Text Index**: Text projects on either backend persist the index itself -- posting lists read lazily from storage behind a least-frequently-used cache, document vectors resident -- so reopening a project assembles it instead of recomputing it (1.4 s against 20.4 s over 265,000 paragraphs).
- **Lifecycle Decoupling (Staging vs. Indexing)**: Incremental staging of documents via `append_items!` with deferred, batch-oriented index construction via `index!`.
- **Logical Deletions**: Non-destructive soft deletes (`delete_item!`) that preserve index integrity while marking candidates in query responses.
- **Batch Query Execution**: `searchbatch(handle, queries, k)` resolves an entire set of queries in one parallel pass and returns the raw `(ids, dists)` matrices; `search(handle, queries, k)` returns the same hits hydrated with their `doc_id` and soft-delete state.
- **Global Metric Operations**: In-process execution of all-pairs nearest neighbors (`allknn`), diverse sampling (`fft`), closest pair discovery (`closestpairs`), and cross-dataset closest pairs (`bichromatic_kclosestpairs`).
- **Query Cost Accounting**: `search(handle, query, k; stats=SearchStats())` reports the distance computations performed by that one call -- the only vantage point from which they can be attributed, since a query runs on a context borrowed from the engine's pool and returned when it ends.
- **Indexed External Identifiers**: `doc_id` resolution is backed by its own column family, and a `doc_id` shared by several items resolves to all of them.
- **Storage Compaction**: `compact_project!`, and an automatic compaction when a writing session closes, keep reopen latency flat after bulk ingestion.
- **Typed Errors**: `EngineError` and its four categories (`InvalidRequest`, `NotFound`, `ConflictingState`, `StorageFailure`) describe *why* a call failed, in types rather than prose.
- **Granular Column-Family Persistence**: Fine-grained persistence where mutations update only the affected RocksDB column families or memory-mapped files.

---

## Installation

The package is not in the General registry yet, so install it from its repository. Its whole
dependency chain — `SimilaritySearch`, `TextSearch`, `RocksDB` and the `RocksDB_jll` binaries —
is registered, so this resolves and precompiles without local checkouts or a build step:

```julia
using Pkg
Pkg.add(url="https://github.com/sadit/SimilaritySearchEngine.jl")
```

To develop the package itself, clone the repository and develop the clone. No `Manifest.toml` is
committed, so `Pkg.instantiate()` resolves the base libraries from the registry; use
`Pkg.develop(path=...)` on each of them to work against your own checkouts instead.

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
