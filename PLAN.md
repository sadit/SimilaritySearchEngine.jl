# Plan

This file describes the three packages of this repository as they are on 2026-11-24, the
decisions that hold, and the work that is planned. It does not record how the packages reached
this state. [`CHANGELOG.md`](CHANGELOG.md) records that: the commits by date, every design with
its phases and measurements, and the earlier versions of this file. The
[manual](https://sadit.github.io/SimilaritySearchEngine.jl/) describes the behavior in detail.
This file links to it and does not repeat it.

## Development Strategy

### One Repository, Three Packages

| package | directory | what it is |
|---|---|---|
| `SimilaritySearchEngine` | the root | the embedded engine: a library, with no HTTP, no command line and no daemon |
| `SimilaritySearchServer` | `server/` | the REST API, the three command lines, the jobs, the tokens and the telemetry |
| `SimilaritySearchClient` | `server/julia/` | the Julia client of the REST API |

The packages have separate `Project.toml` files and separate dependencies. On 2026-09-12,
loading the dependencies of the server took 1.32 s and 28 transitive packages. On the same
date, `using SimilaritySearchEngine` took 1.45 s and a first search took 2.3 s. A program that
embeds the engine does not load the dependencies of the server.

The packages are in one repository because they change together. Until 2026-09-12 the engine
and the server were in two repositories. The server was then three breaking changes of the
engine behind, for 20 days, and no test detected it. One CI run now tests the three packages.

The rules that follow from this:

- A change to an exported name, a keyword or an error type of the engine is finished when the
  server compiles and the suites of both packages pass, in the same commit.
- The server is developed against the engine of this repository:
  `julia --project=server -e 'using Pkg; Pkg.develop(path=".")'`.
- Each package is released on its own. The server declares a `[compat]` range on the engine,
  as it does for any other dependency.
- The client depends on HTTP.jl and JSON3 only. It never depends on the engine or on the
  server. It declares its own copy of the error types of the engine, and the light suite of
  the server checks that the two lists are equal.
- No `Manifest.toml` is committed. The three packages require Julia 1.12 or later, because the
  `[apps]` entries of the server need `Pkg.Apps`.

### Errors

The engine raises typed errors: `EngineError` and its subtypes, in `src/errors.jl`. Each type
belongs to one of four categories: `InvalidRequest`, `NotFound`, `ConflictingState` and
`StorageFailure`. The engine does not know HTTP status codes or exit codes. Each consumer maps
the categories to its own codes:

| category | HTTP status | exit code |
|---|---|---|
| `InvalidRequest` | 400 | 2 |
| `NotFound` | 404 | 4 |
| `ConflictingState` | 409 | 5 |
| `StorageFailure` | 500 | 70 |

A Julia program catches the type that it needs. No code classifies an error by the text of
its message: the category is the contract, and the message is for people. A new error type of
the engine needs a type with the same name and the same category in the Julia client
(`server/julia/src/errors.jl`).

### The Users of the Engine

The engine is written for a Julia program that has it as a dependency and nothing else. The
server is one consumer of the engine, and it is not the only one. The tutorial "Getting
Started" and the paragraph-search case study are written for a program that uses the engine
directly.

- The exports of `SimilaritySearchEngine` are a public API. Programs outside this repository
  call them.
- A breaking change of an export has a cost for those programs. While the version is 0.x, a
  breaking change is allowed. It goes with a version bump and a note in the README.
- The submodules `IndexEngine`, `Persistence`, `Project` and `Schema` are not the public API.
  The server calls some of their functions. A function that a user of the package needs is
  exported from `SimilaritySearchEngine`. A function that only the server needs is not
  exported.

### Defaults

Defaults are on the public surface of the engine, its exports, and nowhere below it. A
function below that surface takes every argument explicitly, positional and keyword.
`test/policy.jl` checks this rule. It parses `src/` with the parser of Julia, and it fails with
the file and the name of each function below the surface that has a default.

A default on an internal function gives a second answer to the same question, in a second
place. Before 2026-08-25, `create_project` took `distance=nothing` and `create_engine` took
`distance=SqL2()`, and a conditional expression decided which of the two applied. In the same
way, `fit_profile` of the engine set `verbose` to `false` by default, and the method of
TextSearch sets it to `true`.

Since issue #2 (2026-11-21), the defaults of a project are in the constructors of its spec.
`DenseVectors()` has the distance `SqL2()`, the index `Graph(minrecall=0.9)` and the storage
`Float32Vectors()`. Every function below `create_project` receives a complete spec.

There are two exceptions:

- A qualified definition, such as `Base.show` or `TextSearch.fit_profile`, follows the
  signature of the function that it extends.
- A public constructor of a public type is on the public surface, for example
  `TextItem(text; doc_id=nothing)`.

Not checked yet: the rule by the direction of the call. A function whose callers are all in
this repository should have no default, also when it is in a submodule that the server calls.
`test/policy.jl` checks whether a function is exported, not who calls it. A check by caller
needs a decision for each function.

### Documentation

One site documents the three packages. The prose is in `manual/`, as Quarto pages. The
reference is generated by `docs/make.jl` from the docstrings of the packages. `AGENTS.md` has
the rules for writing the prose and for rendering the pages. A change that a user can see is
finished when the page that describes it is updated.

### How This File Is Kept

This file describes one date, and it is cleaned from time to time so that it stays short. A
cleanup does these steps:

1. It moves to `CHANGELOG.md` the text that is no longer a plan: a design that was built, an
   item that was finished, a decision that was replaced. The text moves without changes.
2. It adds the commits since the previous cleanup to "Commits by Date" in `CHANGELOG.md`.
3. It rewrites "State on 2026-11-24" and "Planned Work" for the date of the cleanup.
4. For each design that moved, it replaces each reference `PLAN.md §5.N` in the code, the
   tests and the manual with `CHANGELOG.md §5.N`.

Each design has a number of the form §5.N, and it keeps that number when it moves to
`CHANGELOG.md`. The numbers start with 5 because the first designs were subsections of section
5 of the first specification. A new design takes the next number. The last number used is
§5.16, so the next design is §5.17. The code cites a design by its number: `PLAN.md §5.17`
while it is in this file, and `CHANGELOG.md §5.17` after the cleanup that moves it.

A design is written in this file before it is implemented. The user decides the open questions
of a design before its phases start.

Cleanups: 2026-11-24, the first one.

## State on 2026-11-24

### Engine

A **project** is a directory with a RocksDB database and memory-mapped files for dense
vectors. `create_project(workdir, id, spec)` creates one. The spec says what the project holds:

- `DenseVectors(distance; index, storage)`. The index is `Graph(; minrecall)`, a
  `SearchGraph`, or `Exhaustive(; parallel)`. The storage is `Float32Vectors()`,
  `Quantized(SQ(bits); rotation, keep_vectors)` or
  `Sketched(hyperplanes; bits, rerank, keep_vectors)`.
- `SparseVectors(dimension, distance)`, with an `InvertedFile`.
- `TextDocuments(model; index)`, with the index `BM25()` or `VectorSpace(; distance)` and a
  text profile.

An item has an `_id`, which the engine assigns. A client never sets an `_id`. The caller gives
the payload and, optionally, `doc_id`, `keywords`, `refs` and `meta`:

- `doc_id` has an index in a column family, and it does not have to be unique.
- `keywords` are metadata. The engine stores them and does not index them. A keyword follower
  indexes them.
- `meta` is a JSON object. A `MetaSchema` declares the type of some of its fields. A declared
  field is checked when an item is appended, and a filter compares it in its type. No field of
  `meta` is indexed, except a partition field.

`append_items!` stores items and makes them durable. `index!` adds the stored items to the
index. An exact dense project and a sparse project index each item when it is appended.
`delete_item!` marks an item as deleted, and a search skips it. `close_project!` compacts the
database after a session that wrote.

The searches are `search`, `searchbatch` and `ftsearch`, with `filter`, `candidates` and
`stats`. `calibrate!` tunes the beam search of a graph for a target recall. `allknn`, `fft`,
`dnet`, `neardup`, `closestpairs` and `bichromatic_kclosestpairs` work over a whole project.

A **partition** restricts a search to the items that have one value of a key. A partition is
either inside a project (`PartitionBy`, `declare_partition!`), or a partition follower.
`migrate_partitions!` moves the first form to the second. A dense partition with at least
`graph_min` members has its own graph. A smaller dense partition is searched with an exact
scan. `DEFAULT_PARTITION_GRAPH_MIN` is 4096.

A **follower** is a dataset that organizes the items of another dataset, its **leader**. It
reads the items of the leader and does not copy them. It accepts no direct writes, and it is
stored in its own RocksDB database, under `<leader>.followers/`.

- The kinds are `PartitionFollower(source; disjoint, graph_min)` and
  `KeywordFollower(source; distance)`. A plugin can add kinds.
- The key sources are `Keywords()`, `Refs()`, `DocId()` and `MetaValue(name)`. A plugin can add
  key sources.
- A policy says when a follower processes the new items of its leader: `OnIndex()`, the
  default, `Periodic(; seconds, items)` or `Manual()`. `OnWrite()` is another name of
  `OnIndex()`.
- `posting_list(h, key; limit)` reads a list of a keyword follower, a text project or a sparse
  project.

**Events.** `append_items!` and `index!` report the items that became searchable as one event,
`ItemsIndexed(sp, ep)`, per call. The observers added with `observe!` receive it. The followers
are observers of their leader. An error of an observer is written to the log, and the call of
the leader returns.

Where to read more: `manual/architecture.qmd` and the tutorial "Getting Started"
(`manual/tutorials/getting-started.qmd`).

### Server

`similarity-search-server serve` answers the REST API and runs the jobs.

- **Datasets.** A dataset is one project of the engine, with one index, and a `descriptor.json`
  that holds `index_kind`, `distance`, `created_at`, `edit_correction` and `max_k`. A dense
  dataset takes a `storage` object, and any dataset takes a `meta_schema`.
- **Routes.** 45 routes are under `/api/v1`, plus `/healthz`, `/readyz` and `/metrics`. They
  cover the datasets, the items, the searches (`search`, `ftsearch`, and
  `/api/v1/search/hybrid` with reciprocal rank fusion), the followers, the jobs, the tokens and
  the plugins.
- **Size of a response.** A search answers its `k` results in one response. `k` is limited by
  `max_k` of the dataset, 100 by default. `max_k` is set when the dataset is created, and the
  reload route changes it. The server has no cursors. The listings use `offset` and `limit`.
- **Compression.** A response of at least 1,024 bytes is compressed with gzip when the request
  accepts gzip (`[server] gzip` and `gzip_min_bytes`). The result of a job is sent in parts.
- **Jobs.** The kinds are `allknn`, `fft`, `neardup`, `hsp`, `dump` and `load`. A job is a
  subprocess that runs a command of `similarity-search`, and its state is a file in a spool
  directory. `[resources] batch_threads_pct` (20 by default) sets how many jobs run at once
  and how many threads each one has. `max_concurrent_queries` limits the searches that run at
  once.
- **Tokens.** `[auth] enabled` is false by default. A permission is `operation:dataset`, with
  the operation `read`, `write` or `admin`. `similarity-search add-token` creates the first
  token.
- **Telemetry.** Each dataset has an operation log. A record has the user, the fingerprint of
  the token, the elapsed time and the distance computations. `/metrics` reports counters in the
  format of Prometheus.
- **Followers.** `serve` opens the followers of each dataset. A task processes the periodic
  followers every `[followers] check_seconds` seconds, 10 by default.
- **Plugins.** `[plugins]` of `config.toml` loads Julia modules that add kinds of follower and
  key sources.
- **Bundles.** `dump` writes a bundle with `project/`, `followers/`, `dataset.jsonl` and a
  manifest. `dataset.jsonl` has the metadata of each item, one JSON object per line. `load`
  restores a bundle under a new id.

Where to read more: `manual/server/manual.qmd` and the server tutorial
(`manual/server/tutorial.qmd`).

### Command Lines

| command line | installed app | what it does |
|---|---|---|
| `similarity-search` | `similarity-search` | commands that run in the process of the caller, over a working directory |
| `similarity-search-ctl` | `similarity-search-admin` | administration of a running server, over HTTP |
| `similarity-search-server` | `similarity-search-serve` | `serve` |

- `similarity-search` has `build`, `searchbatch`, `allknn`, `fft`, `neardup`, `hsp`,
  `closestpair`, `describe`, `rebuild`, `dump`, `load`, `add-token`, `postings`,
  `migrate-partitions` and the commands of followers. A command that writes a dataset exits
  with code 5 when another process, such as a server, holds the dataset.
- `similarity-search-ctl` has `list`, `stats`, `log`, `jobs` (`queue`, `result`, `block`,
  `resume`, `kill`, `gc`), the commands of tokens, `dump`, `load`, `plugins`,
  `migrate-partitions` and the commands of followers.
- Each of the three has `interactive`. It asks for each argument with the menus of
  `REPL.TerminalMenus` and runs the same command as the arguments would. It refuses to start
  without a terminal.

### Clients

- **Julia.** `SimilaritySearchClient`, in `server/julia/`. Its functions have the names and
  the result types of the engine: `create_project`, `append_items!`, `search`, `ftsearch`,
  `hybrid_search`, the followers, the jobs and `plugins`. It has no administration functions.
  Its page is `manual/server/julia-client.qmd`.
- **Python.** `server/python/similarity_search/`, with `client`, `async_client`,
  `jobs.run_job` and `fields.ftsearch_fields`. `stress_client.py` sends requests to a server.
  The repository has no tests of the Python client. Its changes are checked by hand against a
  server: the last check was on 2026-11-23, with httpx 0.28.1.
- **Node.js.** There is no client.

### Tests

| suite | assertions | time | measured |
|---|---:|---:|---|
| engine, light | 1,119 | 7 min 12 s | 2026-11-23, before phase 9c |
| engine, light | 1,130 | | 2026-11-23, after phase 9c |
| engine, full | 1,166 | 8 min 26 s | 2026-11-23, before phase 9c |
| server, light | 321 | 29 s | 2026-11-23 |
| server, full | 1,385 | 38 min 11 s | 2026-11-23 |
| Julia client, without a server | 63 | | 2026-11-23 |

The times were measured with 8 threads. `Pkg.test(test_args=["jobs", "api/17"])` runs only the
files with those names or tags (`TEST_FILES` in each `runtests.jl`). The full suite of the
server runs before a release, and before a commit that changes the job model, the dispatcher or
the shared test server.

### Documentation

- `manual/architecture.qmd` and the tutorials of `manual/tutorials/` describe the engine.
- `manual/server/manual.qmd`, `manual/server/tutorial.qmd` and
  `manual/server/julia-client.qmd` describe the server and the clients.
- `docs/make.jl` generates the reference of the three packages.
- The rendered manual is committed under `manual/docs/`. `publish-docs.sh` publishes it with the
  reference.

## Planned Work

No phase is queued on 2026-11-24. The last design, §5.16, was built on 2026-11-23. The user
sets the next queue.

### Known Work, Not Scheduled

1. **Tokens stored as a hash.** The token database stores each token as a RocksDB key, and
   `GET /api/v1/admin/tokens` returns the tokens in full. With a hash, a token is shown once,
   when it is created, and it is revoked by its fingerprint. The operation log stores a
   fingerprint since 2026-09-14 (`CHANGELOG.md` §0, item 8).
2. **The default `graph_min` of a dense partition.** `DEFAULT_PARTITION_GRAPH_MIN` was not
   measured on real embeddings. This was phase 3 of §5.10.
3. **A second evaluation for `Quantized`.** `Sketched` scores `rerank * k` candidates again
   with the kept vectors. `Quantized` has no second evaluation (issue #1).
4. **The whole spec over HTTP.** The server takes `index_type`, `distance`, `dimension` and a
   nested `storage` object. It does not accept the nested form `{"dense": {...}}` of issue #2.
5. **Skipping an item that a follower cannot process.** A follower stops at such an item
   (§5.12, "Decided after phase 5", decision 3). Skipping it changes no `_id`.
6. **Delaying the fitting of an encoder in the server.** The first `append` of a quantized or
   sketched dataset fits the encoder on the vectors of that request (§5.14). The manual says to
   send enough vectors in the first request.
7. **A binary format of results** (§5.16, decision 3). It was postponed on 2026-11-23.
8. **More kinds of event** (§5.15). `ItemsStaged` and `ItemsDeleted` are added when a consumer
   needs them. Other consumers: counters for the telemetry, a notification to another system,
   and observers added by a plugin.
9. **The check of defaults by caller** ("Defaults" above).
10. **A Node.js client** (§8.4). It was never started.

### Possible Later Work

The designs recorded these as "not in this version". None of them is planned.

- Partitions of sparse projects; more than one partition field; the intersection of two
  partitions (§5.9, §5.10).
- Loading the graph of a partition at its first query (§5.10).
- A follower of a follower; a follower in another working directory or on another server; the
  propagation of deletes to a follower; a log of changes (§5.11, §5.12).
- A job over a follower, such as `allknn` over one partition (§5.12).
- Processing an `OnIndex` follower outside the request of `append`, if a measurement shows that
  the follower takes most of the time of `append` (§5.12).
- A route of queries for a plugin kind; loading a plugin while `serve` runs; plugins in a job
  (§5.13).
- A route that runs several searches in one request, if a measurement shows that the round
  trips matter next to the time of the searches (§0, item 11).
- `SlurmExecutor`: jobs that run on SLURM with the same commands (§5.5).

## Open Questions

1. **When are `PartitionBy` and `declare_partition!` removed?** The partition follower
   replaces them, and `migrate_partitions!` moves a project to it (issue #5).
2. **Does a request collector add anything over the limit of concurrent searches?** §2 of the
   first specification proposed a channel of pending requests and a pool of worker tasks. The
   server has a semaphore. The metrics `simsearch_queries_waiting` and
   `simsearch_query_wait_seconds_total` answer the question (§0, item 9).
3. **Is an index of `refs` worth its cost?** The engine writes `idx_refs`, and the benefit was
   not measured (issue #5). A keyword follower with the key source `Refs()` is the other way to
   index them.
4. **What happens with `[search] long_query_policy`?** The key is in `config.toml`, and no code
   reads it. §3 of the first specification proposed to answer `429` to a search whose cost
   exceeds a threshold, with a pointer to a job. The options are to build it or to remove the
   key.

The section numbers in this file without a file name (§0, §2, §5.10, ...) are sections of
`CHANGELOG.md`.
