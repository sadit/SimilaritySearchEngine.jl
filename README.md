# SimilaritySearchEngine.jl

[![CI](https://github.com/sadit/SimilaritySearchEngine.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/sadit/SimilaritySearchEngine.jl/actions/workflows/ci.yml)
[![Manual](https://img.shields.io/badge/docs-manual-blue.svg)](https://sadit.github.io/SimilaritySearchEngine.jl/manual/)
[![API Reference](https://img.shields.io/badge/docs-reference-blue.svg)](https://sadit.github.io/SimilaritySearchEngine.jl/dev/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`SimilaritySearchEngine.jl` is an embedded similarity search engine for Julia. It provides transactional, persistent storage and nearest-neighbor search for collections of dense vectors, sparse representations, and full-text documents. It integrates [SimilaritySearch.jl](https://github.com/sadit/SimilaritySearch.jl), [TextSearch.jl](https://github.com/sadit/TextSearch.jl), and [RocksDB.jl](https://github.com/sadit/RocksDB.jl) into a unified, in-process engine.

---

## Architectural Highlights

- **Decoupled Engine and Backend Models**: 
  - **Payload Engines**: Specify the payload data domain (`DenseEngine`, `SparseEngine`, `FullTextEngine`).
  - **Index Backends**: Specify the underlying indexing algorithm (`SearchGraph`, `ExhaustiveSearch`, `ParallelExhaustiveSearch`, `InvertedFile`, `BM25InvertedFile`).
  - Strong typing with dedicated item containers: `DenseItem`, `SparseItem`, and `TextItem`.
- **Dense Vector Search**: Approximate nearest-neighbor search via `SearchGraph` with dynamic beam-search tuning, or exact brute-force search via `ExhaustiveSearch`.
- **Sparse Vector Search**: Direct indexing of user-provided sparse representations with fixed dimensionality via `InvertedFile`, evaluated under cosine distance or discrete set metrics (`Dist.Sets.*`).
- **Full-Text Retrieval**: Lexical search with BM25 (`BM25InvertedFile`) or cosine-weighted TF-IDF (`TextInvertedFile`), configured through linguistic profiles (`TextProfile`).
- **Persisted Text Index**: A text project -- BM25 or weighted TF-IDF -- stores its inverted index itself, posting lists and per-document vectors in dedicated column families, instead of storing the indexed objects and recomputing the index on every open. Reopening 265,000 Project Gutenberg paragraphs costs 1.4 s against 20.4 s, and indexing them 25.7 s against 147.3 s. Posting lists are read from storage on demand behind a least-frequently-used cache; document vectors stay resident, because scoring reads a vector per candidate.
- **Decoupled Staging and Indexing**: Items are appended and made durable immediately via `append_items!`, while computationally demanding index construction (graph building or full-text vocabulary encoding) is deferred to explicit `index!` calls.
- **Linguistic Profiles & Query Policies**: Explicit text modeling via `DefaultProfile`, `BaseProfile`, or `FitFromCorpus`, combined with query-time orthographic correction and semantic expansion via `QueryPolicy`.
- **Logical Deletions**: Non-destructive soft deletes via `delete_item!`, reporting candidate deletion states without requiring costly index rebuilds.
- **Target Recall Calibration**: Optimization of graph traversal hyperparameters via `calibrate!` to guarantee minimum recall constraints (`minrecall`).
- **Batch Query Execution**: `searchbatch` answers many queries in one parallel pass -- 39,000 queries/s against 3,600/s query-by-query, measured over 50,000 dense vectors on 8 threads -- returning raw `(ids, dists)` matrices, while `search(handle, queries, k)` returns the same results hydrated.
- **Whole-Dataset Metric Operations**: In-process execution of all-pairs nearest neighbors (`allknn`), diverse center sampling (`fft`), closest pair discovery (`closestpairs`), and cross-dataset closest pairs (`bichromatic_kclosestpairs`).
- **Indexed External Identifiers**: `doc_id` lookups resolve through a dedicated column family (0.12 ms against 59 ms for a scan, at 50,000 items) and are not required to be unique -- `fetch_items` returns every item carrying the requested identifier.
- **Storage Compaction**: `close_project!` compacts the project after a session that wrote, which is what keeps the next open fast (0.28 s against 2.97 s); `compact_project!` runs it mid-session.
- **Granular Storage Persistence**: High-throughput persistence utilizing a hybrid layout of RocksDB column families and dedicated memory-mapped vector files (`dense_vectors.mmapdb`).

---

## Installation

The package is not in the General registry yet, so install it from its repository. Every
dependency it needs — including `RocksDB.jl` and its `RocksDB_jll` binaries — *is* registered, so
this resolves and precompiles with no local checkouts and no build step:

```julia
using Pkg
Pkg.add(url="https://github.com/sadit/SimilaritySearchEngine.jl")
```

To work on the package itself, clone it and `Pkg.develop` the clone. Note that no `Manifest.toml`
is committed: `Pkg.instantiate()` resolves the base libraries from the registry. Point them at
your own checkouts with `Pkg.develop(path=...)` when you need to.

```julia
using Pkg
Pkg.develop(path="path/to/SimilaritySearchEngine.jl")
```

---

## Quickstart

```julia
using SimilaritySearchEngine

# 1. Initialize a working directory and create a dense project
workdir = mktempdir()
h = create_project(workdir, "demo_dense"; engine=DenseEngine, backend=SearchGraph, minrecall=0.9)

# 2. Stage dense vector items
batch = [DenseItem(rand(Float32, 32); doc_id="doc-$i") for i in 1:1000]
append_items!(h, batch)

# 3. Construct the search graph over the staged backlog
index!(h)

# 4. Execute a 5-nearest-neighbor query
results = search(h, rand(Float32, 32), 5)
for r in results
    println("Document ID: ", r.doc_id, " | Distance: ", r.distance)
end

# 5. Close project handles
close_project!(h)
```

---

## Documentation

- **[Manual](https://sadit.github.io/SimilaritySearchEngine.jl/manual/)**:
  - [Architecture Guide](manual/architecture.qmd): Detailed specifications of engine submodules, concurrency controls, and the on-disk storage layout.
  - [Getting Started Tutorial](manual/tutorials/getting-started.qmd): Step-by-step walkthrough covering dense embeddings, sparse indexing, BM25 text retrieval, filtering, calibration, and whole-dataset operations.
  - [Multilingual Case Study](manual/tutorials/paragraph-search.qmd): Real-world evaluation over Project Gutenberg and Wikipedia paragraph corpora.
- **[API Reference](https://sadit.github.io/SimilaritySearchEngine.jl/dev/)**: Complete reference documentation generated with Documenter.jl.

---

## License

MIT. See [LICENSE](LICENSE).

---

## Ecosystem Integration

`SimilaritySearchEngine.jl` provides the embedded engine core. Higher-level services, including HTTP REST endpoints, asynchronous job queues, and command-line interfaces, are implemented in the sibling package `SimilaritySearchServer.jl`, which consumes this embedded API directly.
