# SimilaritySearchEngine.jl

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
- **Decoupled Staging and Indexing**: Items are appended and made durable immediately via `append_items!`, while computationally demanding index construction (graph building or full-text vocabulary encoding) is deferred to explicit `index!` calls.
- **Linguistic Profiles & Query Policies**: Explicit text modeling via `DefaultProfile`, `BaseProfile`, or `FitFromCorpus`, combined with query-time orthographic correction and semantic expansion via `QueryPolicy`.
- **Logical Deletions**: Non-destructive soft deletes via `delete_item!`, reporting candidate deletion states without requiring costly index rebuilds.
- **Target Recall Calibration**: Optimization of graph traversal hyperparameters via `calibrate!` to guarantee minimum recall constraints (`minrecall`).
- **Whole-Dataset Metric Operations**: In-process execution of all-pairs nearest neighbors (`allknn`), diverse center sampling (`fft`), closest pair discovery (`closestpairs`), and cross-dataset closest pairs (`bichromatic_kclosestpairs`).
- **Granular Storage Persistence**: High-throughput persistence utilizing a hybrid layout of RocksDB column families and dedicated memory-mapped vector files (`dense_vectors.mmapdb`).

---

## Quickstart

```julia
using Pkg
Pkg.develop(path="path/to/SimilaritySearchEngine.jl")

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

## Ecosystem Integration

`SimilaritySearchEngine.jl` provides the embedded engine core. Higher-level services, including HTTP REST endpoints, asynchronous job queues, and command-line interfaces, are implemented in the sibling package `SimilaritySearchServer.jl`, which consumes this embedded API directly.
