# SimilaritySearchEngine.jl

[![CI](https://github.com/sadit/SimilaritySearchEngine.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/sadit/SimilaritySearchEngine.jl/actions/workflows/ci.yml)
[![Manual](https://img.shields.io/badge/docs-manual-blue.svg)](https://sadit.github.io/SimilaritySearchEngine.jl/)
[![API Reference](https://img.shields.io/badge/docs-reference-blue.svg)](https://sadit.github.io/SimilaritySearchEngine.jl/api/engine.html)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`SimilaritySearchEngine.jl` is an embedded similarity search engine for Julia. It provides transactional, persistent storage and nearest-neighbor search for collections of dense vectors, sparse representations, and full-text documents. It integrates [SimilaritySearch.jl](https://github.com/sadit/SimilaritySearch.jl), [TextSearch.jl](https://github.com/sadit/TextSearch.jl), and [RocksDB.jl](https://github.com/sadit/RocksDB.jl) into a unified, in-process engine.

This repository holds **three packages**:

| | what it is | where |
|---|---|---|
| `SimilaritySearchEngine` | the embedded engine: a library, no HTTP, no CLI, no daemon | repository root |
| `SimilaritySearchServer` | REST API, CLIs, job queue, and three installable apps over that engine | [`server/`](server/) |
| `SimilaritySearchClient` | a Julia client of the REST API, with the engine's function names; depends on HTTP.jl and JSON.jl only | [`server/julia/`](server/julia/) |

Two more clients of the REST API are in `server/`: one for Python, in [`server/python/`](server/python/), and one for Node.js, in TypeScript, in [`server/node/`](server/node/).

They are developed together and tested in one CI run, with separate dependency sets: loading the server's tree costs 1.32 s and 28 transitive packages, which a caller embedding the engine never pays. The section "Development Strategy" of [`PLAN.md`](PLAN.md) says why the packages are separate, how they are kept in step, and where the errors are mapped. `PLAN.md` also describes the state of the packages and the planned work, and [`CHANGELOG.md`](CHANGELOG.md) records their history.

---

## Architectural Highlights

- **Project Specifications**: A project is described by one value, built from smaller ones (issue #2):
  - What it holds: `DenseVectors(distance; index, storage)`, `SparseVectors(dimension, distance)`, `TextDocuments(model; index)`.
  - The index: `Graph(; minrecall)` (a `SearchGraph`) or `Exhaustive(; parallel)` for dense vectors, `BM25()` or `VectorSpace(; distance)` for text.
  - The storage of dense vectors: `Float32Vectors()`, `Quantized(SQ(bits; range); rotation, keep_vectors)`, or `Sketched(hyperplanes; bits, rerank, keep_vectors)`.
  - Each type holds only its own options, so a combination that does not exist cannot be written. `project_spec(h)` returns the spec of an open project.
  - Strong typing with dedicated item containers: `DenseItem`, `SparseItem`, and `TextItem`.
- **Dense Vector Search**: Approximate nearest-neighbor search via `SearchGraph` with dynamic beam-search tuning, or exact brute-force search via `ExhaustiveSearch`.
- **Sparse Vector Search**: Direct indexing of user-provided sparse representations with fixed dimensionality via `InvertedFile`, evaluated under cosine distance or discrete set metrics (`Dist.Sets.*`).
- **Full-Text Retrieval**: Lexical search with BM25 (`BM25InvertedFile`) or cosine-weighted TF-IDF (`TextInvertedFile`), configured through linguistic profiles (`TextProfile`).
- **Persisted Text Index**: A text project -- BM25 or weighted TF-IDF -- stores its inverted index itself, posting lists and per-document vectors in dedicated column families, instead of storing the indexed objects and recomputing the index on every open. Reopening 265,000 Project Gutenberg paragraphs costs 1.4 s against 20.4 s, and indexing them 25.7 s against 147.3 s. Posting lists are read from storage on demand behind a least-frequently-used cache; document vectors stay resident, because scoring reads a vector per candidate.
- **Decoupled Staging and Indexing**: Items are appended and made durable immediately via `append_items!`, while computationally demanding index construction (graph building or full-text vocabulary encoding) is deferred to explicit `index!` calls.
- **Linguistic Profiles & Query Policies**: Explicit text modeling via `DefaultProfile`, `BaseProfile`, or `FitFromCorpus`, combined with query-time orthographic correction and semantic expansion via `QueryPolicy`.
- **Logical Deletions**: Non-destructive soft deletes via `delete_item!`, reporting candidate deletion states without requiring costly index rebuilds.
- **Target Recall Calibration**: Optimization of graph traversal hyperparameters via `calibrate!` for target recall values (`minrecall`), with the reached recall measured on held-out queries and reported with a bootstrap interval.
- **Query Cost Accounting**: `search(h, q, k; stats=SearchStats())` reports the distance computations that one call performed -- 922 at `minrecall=0.8` against 1,623 at `minrecall=0.95`, over 20,000 random 32-dimensional vectors (means over 100 queries and three builds, measured on 2026-11-07 with SimilaritySearch 1.6.11). The count comes from inside the search because a query runs on a context borrowed from the engine's pool: counters read from outside afterwards belong to whatever has searched since.
- **Batch Query Execution**: `searchbatch` answers many queries in one parallel pass -- 39,000 queries/s against 3,600/s query-by-query, measured over 50,000 dense vectors on 8 threads -- returning raw `(ids, dists)` matrices, while `search(handle, queries, k)` returns the same results hydrated.
- **Whole-Dataset Metric Operations**: In-process execution of all-pairs nearest neighbors (`allknn`), diverse center sampling (`fft`), closest pair discovery (`closestpairs`), and cross-dataset closest pairs (`bichromatic_kclosestpairs`).
- **Indexed External Identifiers**: `doc_id` lookups resolve through a dedicated column family (0.12 ms against 59 ms for a scan, at 50,000 items) and are not required to be unique -- `fetch_items` returns every item carrying the requested identifier.
- **Storage Compaction**: `close_project!` compacts the project after a session that wrote, which is what keeps the next open fast (0.28 s against 2.97 s); `compact_project!` runs it mid-session.
- **Granular Storage Persistence**: High-throughput persistence utilizing a hybrid layout of RocksDB column families and dedicated memory-mapped vector files (`dense_vectors.mmapdb`).
- **Followers**: A follower organizes the items of another project, its leader, and does not copy them (issue #5). A partition follower restricts a search to the items of one key. A keyword follower indexes the keywords of each item in an inverted file. A follower is stored in its own RocksDB database, and its policy says when it processes the new items of its leader.
- **Events of a Project**: `index!` and `append_items!` report the items that became searchable as one `ItemsIndexed(sp, ep)` event per call, to the observers that `observe!` added. The followers are observers of their leader. An error of an observer is written to the log, and the call of the project returns.
- **Typed Errors**: Every failure the engine raises on purpose is an `EngineError` in one of four categories -- `InvalidRequest`, `NotFound`, `ConflictingState`, `StorageFailure` -- with a concrete type underneath (`PayloadMismatch`, `PendingBacklog`, `WrongDimension`, ...). A caller acts on the category instead of parsing a message; the server maps categories to HTTP status codes and the CLIs to exit codes.

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
h = create_project(workdir, "demo_dense", DenseVectors(Dist.SqL2(); index=Graph(minrecall=0.9)))

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

## Incompatible Changes in 0.2.0

Version 0.2.0 (2026-11-26) is the first change of the version number. The package was 0.1.0
from its first commit, on 2026-08-18, and its API changed under that number. This list covers
the changes since 2026-09-12, the date of the license and of the first compat bounds. Each item
says what to change in a program written for an earlier commit. "Commits by Date" in
[`CHANGELOG.md`](CHANGELOG.md) lists every commit. The server changed to 0.2.0 on the same date,
and [its list](#incompatible-changes-in-the-server-020) is in the section about the server.

**Errors (2026-09-12).** The engine raises an `EngineError`. Its category is `InvalidRequest`,
`NotFound`, `ConflictingState` or `StorageFailure`, and each error has its own type under the
category, such as `PendingBacklog`. Before, the engine raised `ErrorException` with a message.
Catch a type, not a message. Since 2026-11-24, a query or an item of a dense `Float32` project
with another length raises `WrongDimension`. Before, such a query raised the `DimensionMismatch`
of Julia, and an exact project kept such an item in its metadata but not in its vector file.

**Calibration (2026-10-13).** `calibrate!` returns one `CalibrationLevel` per level, with the
recall that the level reaches. Before, it returned the table of `BeamSearch` configurations.

**Project specs (2026-11-21, issue #2).** `create_project` takes a `ProjectSpec` as its third
argument, in place of keywords:

| Before | Now |
|---|---|
| `engine=DenseEngine, backend=SearchGraph, distance=d, minrecall=r` | `DenseVectors(d; index=Graph(minrecall=r))` |
| `backend=ExhaustiveSearch` | `DenseVectors(d; index=Exhaustive())` |
| `backend=ParallelExhaustiveSearch` | `DenseVectors(d; index=Exhaustive(parallel=true))` |
| `quantizer=...` (2026-11-20 only) | `DenseVectors(d; storage=Quantized(SQ(bits)))` |
| `engine=SparseEngine, dimension=n, distance=d` | `SparseVectors(n, d)` |
| `engine=FullTextEngine, backend=BM25InvertedFile, textmodel=m` | `TextDocuments(m; index=BM25())` |
| `engine=FullTextEngine, backend=TextInvertedFile, textmodel=m` | `TextDocuments(m; index=VectorSpace())` |
| `partition_by=f, partition_graph_min=n` | `partitions=PartitionBy(f; graph_min=n)` |
| `maxbatches`, `postings_cache_max`, `postings_cache_base`, `edit_correction` | `runtime=Runtime(...)`, with the same names |

`meta_schema` and `schema_version` are unchanged. The same commit made these changes too:

- `open_project` takes `runtime=Runtime(...)` in place of `maxbatches`, `postings_cache_max`,
  `postings_cache_base` and `edit_correction`.
- `declare_partition!(h, field)` is `declare_partition!(h, PartitionBy(field))`.
- These names are no longer exported: `DenseEngine`, `SparseEngine`, `FullTextEngine`,
  `SearchGraph`, `ExhaustiveSearch`, `ParallelExhaustiveSearch`, `InvertedFile`,
  `BM25InvertedFile` and `TextInvertedFile`. They still exist, for example as
  `SimilaritySearchEngine.SearchGraph`. `BACKENDS`, `default_backend` and `ScalarQuantizer`
  were removed.
- A project saved before this change has no saved spec. `open_project` builds the spec from the
  fields that the project saved. A quantized project created on 2026-11-20 must be created
  again.

**Exact projects (2026-11-22, issue #3).** An exact project of `Float32` vectors keeps its
vectors in its vector file. The first writable open with 0.2.0 moves them to that file and
removes the copy that earlier versions read. A read-only open does not change the project.

**Metadata (2026-11-24).** The `meta` that a filter function receives, and the value of
`get_meta(...; lazy=true)`, are a `JSON.Object{String,Any}` of JSON.jl v1. Before, they were a
`JSON3.Object`, whose keys are `Symbol`s. `meta["year"]`, `meta[:year]` and `meta.year` work as
before. Code that iterates the keys, or compares them with `Symbol`s, has to compare them with
`String`s.

---

## Documentation

Both packages of this repository are documented on
[one site](https://sadit.github.io/SimilaritySearchEngine.jl/): one manual with one navigation
and one search index, and one generated reference. GitHub Pages serves a single site per
repository, and the engine and the service layer are two halves of one thing anyway.

- **[Manual](https://sadit.github.io/SimilaritySearchEngine.jl/)** — prose, for both packages:
  - [Architecture Guide](manual/architecture.qmd): Detailed specifications of engine submodules, concurrency controls, and the on-disk storage layout.
  - [Getting Started Tutorial](manual/tutorials/getting-started.qmd): Step-by-step walkthrough covering dense embeddings, sparse indexing, BM25 text retrieval, filtering, calibration, followers, observers of a project, and whole-dataset operations.
  - [Multilingual Case Study](manual/tutorials/paragraph-search.qmd): Real-world evaluation over Project Gutenberg and Wikipedia paragraph corpora.
  - [Server Manual](manual/server/manual.qmd) and [Server Tutorial](manual/server/tutorial.qmd): the REST API, the job queue, the two command lines, and the error mapping.
  - [Julia Client](manual/server/julia-client.qmd): the engine's functions over the REST API of a remote server.
  - [Node.js Client](manual/server/node-client.qmd): one method per route of the REST API, in TypeScript, for Node.js 22 or later.
- **API Reference**, generated with Documenter.jl from the docstrings of both packages:
  [Engine API](https://sadit.github.io/SimilaritySearchEngine.jl/api/engine.html),
  [Server API](https://sadit.github.io/SimilaritySearchEngine.jl/api/server.html) and
  [Julia client API](https://sadit.github.io/SimilaritySearchEngine.jl/api/client.html).

`publish-docs.sh` builds the reference and publishes it together with the pre-rendered manual.

---

## Testing

Two levels, in both packages: light on every change, full before a release.

```julia
using Pkg
Pkg.test()                          # light   -- engine 1,119 assertions, 7m12s
Pkg.test(test_args=["full"])        # full    -- engine 1,166 assertions, 8m26s
Pkg.test(test_args=["text"])        # only the files named or tagged "text"
```

The times are those of the outer testset, measured on 2026-11-23 with 8 threads (`julia -t 8`). The four suites ran at the same time on one machine of 60 cores.

A run that receives names or tags includes only the test files that match one of them, with their heavy testsets. Each suite lists its files, with their names and tags, in the table `TEST_FILES` of its `runtests.jl`. The server's files are `api/01` to `api/21`, `cli/01` to `cli/08`, `e2e`, `boundary`, `dispatcher` and `config`. The engine's files include `dense`, `text`, `persistence`, `quantized`, `sketches`, `followers` and `events`. `SSE_TEST_SELECT="api/13 jobs"` gives the same selection from the environment. A name that matches no file is an error. `test/selection.jl` implements the selection for both suites.

The Node.js client has its own tests, which need no server: `npm install` and `npm test` in `server/node/`, with Node.js 22 or later. Its tests against a server are `api/21` of the server's full level, which uses the `node` of `SSE_NODE` or of `PATH`.

The split follows the clock, not importance. The engine's full level adds the whole-dataset algorithms, the concurrency stress test, the text testsets that fit a linguistic profile from a corpus, and the resident memory of a quantized project -- everything they cover is covered in the light run too, at a size that runs in about a second each. The server's light level (365 assertions, 37 s, measured on 2026-11-25) covers the boundary with the engine -- wire names to engine types, engine errors to status and exit codes, wire items to engine items -- without starting a server or spawning a subprocess; its full level (1,451 assertions, 9 min, measured on 2026-11-25) adds the HTTP, job, CLI and Julia client end-to-end suites. `SSE_TEST_LEVEL=full` selects the full level from CI.

A skipped testset is named at the end of a light run: a light run that looks identical to a full one is how a suite quietly stops testing something.

---

## License

MIT. See [LICENSE](LICENSE).

---

## The server subpackage

[`server/`](server/) holds `SimilaritySearchServer`: a REST API (Oxygen), two CLIs, an asynchronous job queue with a filesystem spool, tokens and telemetry, plus three installable apps (`similarity-search`, `similarity-search-admin`, `similarity-search-serve`). It consumes this engine's public API and nothing below it.

```julia
using Pkg
Pkg.develop(path=".")            # the engine from this repository
Pkg.activate("server")
Pkg.develop(path=".")            # ... and the server against it
```

Julia 1.12 or later is required by both, and by the server outright: its `[apps]` entries need `Pkg.Apps`, which does not exist before 1.12.

Its documentation is not separate either: the [server manual](https://sadit.github.io/SimilaritySearchEngine.jl/server/manual.html), the [server tutorial](https://sadit.github.io/SimilaritySearchEngine.jl/server/tutorial.html) and the [Server API](https://sadit.github.io/SimilaritySearchEngine.jl/api/server.html) are sections of the same site as the engine's.

### Incompatible Changes in the Server 0.2.0

The server changed from 0.1.0 to 0.2.0 on 2026-11-26, with the engine. It requires the engine
0.2. This list covers the changes since 2026-09-12, the date the server came into this
repository. Each item says what to change in a client, a script or a working directory made for
an earlier commit.

**Routes (2026-09-13).** An operation on one dataset has its route under that dataset.
`/api/v1/simsearch/{id}/X` is now `/api/v1/datasets/{id}/X`, for `append`, `search`,
`ftsearch`, `fetch`, `exists`, `delete` and `calibrate`. `POST /api/v1/simsearch/hybrid_search`
is now `POST /api/v1/search/hybrid`. The old routes have no aliases.

**Identifiers of an item (2026-09-14).** Every response names an item with `doc_id` and `_id`.
`doc_id` is the identifier that the caller gave, and it can be `null`. `_id` is the integer that
the engine assigned. Before, a search result used `id` for the first and `doc_id` for the
second. The response of `fetch` no longer has `id`. `POST /api/v1/datasets/{id}/delete` takes
`_id`. Before, it took that integer under the name `doc_id`.

**Operation log (2026-09-14).** An entry of `GET /api/v1/datasets/{id}/log` has `user` and
`token_fingerprint` in place of `token`. `user` is the name on the token, when the token is
valid. `token_fingerprint` is the first 16 hexadecimal digits of the SHA-256 of the token.
Before, the entry kept the token itself. The server does not rewrite the entries written before
2026-09-14.

**Join groups (2026-11-10).** The server has no join groups. `POST /api/v1/datasets` answers 400
to a request with `fields`, `join_group` or `holds_metadata`, and the error message names what
replaces them. The routes `POST /api/v1/search/group` and `GET /api/v1/datasets/{id}/join_group`
were removed. Until 2026-09-13, the first one was `POST /api/v1/simsearch/ftsearch`. Each text
field is now a dataset of its own. To search several fields with one text, send one `ftsearch`
request to each dataset, as `ftsearch_fields` of the Python client does. A descriptor written
before 2026-11-10 still loads, and the server ignores the three fields.

**Text queries (2026-11-10).** `ftsearch` answers 400 to a `text` that is not a string. The
command `searchbatch` exits with a status other than 0 after an engine error. Before, it exited
with 0.

**Size of a response (2026-11-23).** The server has no result cursors. The route
`/api/v1/cursors/{cursor_id}` was removed, and a request with `page_size` answers 400
`InvalidOption`. Each dataset has a `max_k`, and a request with a larger `k` answers 400
`InvalidOption`. The limit applies to `search`, `ftsearch`, the hybrid search and the searches
of a follower. A dataset created before 2026-11-23 has the default `max_k`, 100.
`POST /api/v1/admin/datasets/{id}/reload` with the body `{"max_k": n}` changes it. The Python
client no longer has `page_size`, `poll_cursor` and `iter_cursor`. The Julia client no longer
has `search_pages`.

**Dump bundles (2026-11-24).** `dump` writes the metadata of the items to `dataset.jsonl`, one
JSON object per item, in place of `dataset.avro`. A program that reads `dataset.avro` from a
bundle has to read `dataset.jsonl`. `load` accepts a bundle with `dataset.avro`. It ignores that
file with a warning, and it restores the metadata from `project/`.

**Datasets of a working directory (2026-11-22).** `serve` opens each dataset with write access.
So the first start of 0.2.0 converts the exact projects of the working directory, as the item
"Exact projects" of [the list of the engine](#incompatible-changes-in-020) says. The conversion
removes the copy of the vectors that a server of an earlier commit reads.
