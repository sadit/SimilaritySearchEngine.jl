# Technical Manual: SimilaritySearchServer

`SimilaritySearchServer` exposes [`SimilaritySearchEngine`](../../README.md) — the embedded
engine in the root of this repository — through a REST API, two CLIs, and an asynchronous job
queue. It holds no index logic of its own: **everything about storing, indexing and searching
lives in the engine**, and this package is the layer that turns HTTP requests and command
lines into engine calls, and engine results and failures into responses and exit codes.

That boundary is the thing to keep in mind while reading the rest: when something about
projects, persistence, profiles or search behaviour is unclear, the answer is in the engine's
[architecture guide](https://sadit.github.io/SimilaritySearchEngine.jl/manual/architecture.html),
not here.

## 1. What this package contains

| file | what it does |
| :--- | :--- |
| `config.jl`, `cli.jl`, `ctl.jl` | TOML configuration, and the two `ArgParse` command lines (`similarity-search`, `similarity-search-admin`) |
| `cli_handlers.jl`, `ctl_handlers.jl` | one function per subcommand: `build`, `search`, `searchbatch`, `allknn`, `fft`, `neardup`, `hsp`, `closestpair`, `describe`, `rebuild`, `dump`, `load` |
| `server.jl` | the HTTP API (Oxygen), one `EmbeddedEngine` per dataset, and the request/response mapping |
| `jobs.jl`, `executors.jl` | the job spool: an atomic directory tree (`queued/`, `running/`, `completed/`, `failed/`) of `.job.toml` files, dispatched as independent Julia subprocesses |
| `tokens.jl`, `telemetry.jl`, `cursors.jl` | access tokens, the `op_log` operation history, and server-side result cursors for pagination |
| `interactive.jl` | guided invocation through `REPL.TerminalMenus` |
| `apps/` | the three installable app entry points |

Schema, persistence, the index engine and project management are **not** here: they are
`SimilaritySearchEngine.Schema`, `.Persistence`, `.IndexEngine` and `.Project`.

## 2. How a dataset is held

One `SimilaritySearchEngine.EmbeddedEngine` per dataset id, in `AppState.handles`. That single
handle carries both halves of a project (its RocksDB metadata store and its live search
engine), which is what keeps them consistent across a restart: `reload_datasets!` reopens
every dataset under `<workdir>/datasets/` with `open_project`, and a reopened dataset comes
back with its index, profile, staged backlog and tombstones intact.

There is no snapshot file. Earlier versions of this package wrote a JLD2 snapshot per dataset
and reloaded it on demand, because the engine did not persist its own index; it does now, and
a dataset is simply the directory it lives in.

**Heavy jobs run as subprocesses and open their project read-only**, since the serving process
may hold it open and RocksDB's write lock is exclusive per process.

## 3. Kinds of project

The API's `index_kind` names a pair — what a project holds, and what indexes it:

| `index_kind` | engine | backend | notes |
| :--- | :--- | :--- | :--- |
| `searchgraph` | `DenseEngine` | `SearchGraph` | approximate, self-tuning; the default |
| `exhaustive_search` | `DenseEngine` | `ExhaustiveSearch` | exact brute force |
| `parallel_exhaustive_search` | `DenseEngine` | `ParallelExhaustiveSearch` | exact, data-parallel |
| `bm25_invfile` | `FullTextEngine` | `BM25InvertedFile` | BM25 scoring |
| `invfile` | `FullTextEngine` | `TextInvertedFile` | TF-IDF cosine |
| `sparse_invfile` | `SparseEngine` | `InvertedFile` | caller-encoded sparse vectors; needs `dimension` |

Text projects are created with `FitFromCorpus`: the vocabulary is fitted from whatever the
dataset itself has been given by the first `index!`. Shipping a pre-trained profile to the
server is not exposed over HTTP yet.

## 4. REST API (base endpoints)

### Control
- `GET /readyz` — health.
- `GET /metrics` — Prometheus-style gauges, including per-dataset document and tombstone counts.

### Datasets
- `POST /api/v1/datasets` — creates one. Accepts `id`, `index_kind` (see above), `distance`, `dimension`, and the join-group fields `join_group`, `holds_metadata`, `key`.
- `DELETE /api/v1/datasets/{id}` — destroys it.
- `POST /api/v1/datasets/{id}/unload` / `reload` — closes or reopens a dataset without restarting the server (what the CLI's `rebuild` needs, since RocksDB's write lock is exclusive).

### Search and indexing
- `POST /api/v1/simsearch/{id}/append` — appends items. Dense items carry `vector`, text items `text`, sparse items `indices`+`values`; `doc_id`, `keywords`, `refs` and any other field become the item's metadata.
- `POST /api/v1/simsearch/{id}/search` — dense/sparse search, with optional `filter`, `page_size` and `beamsearch_overrides`.
- `POST /api/v1/simsearch/{id}/ftsearch` — text search.
- `POST /api/v1/simsearch/{id}/delete` — soft delete.
- `POST /api/v1/simsearch/{id}/calibrate` — recall calibration for a graph-backed project.
- `POST /api/v1/simsearch/hybrid` and `/ftsearch_group` — reciprocal-rank fusion across two datasets, and text fan-out across a join group.

### Telemetry
- `GET /api/v1/datasets/{id}/log?offset=&limit=` — the dataset's `op_log`: one record per `search`/`ftsearch`/`append`/`delete`, with elapsed time and `distance_evaluations`.

The evaluation count for a search comes from the engine (`SimilaritySearchEngine.SearchStats`,
passed into `search`/`ftsearch` by `_run_search`), not from reading a context before and after
the call: a search runs on a context borrowed from the engine's pool, so counters read from out
here belong to whatever else has searched in the meantime. `append` does read its context that
way, correctly — insertion holds the engine's own context under an exclusive write lock.

### Asynchronous jobs
- `POST /api/v1/jobs/{kind}` — submits (`allknn`, `fft`, `neardup`, `hsp`, `searchbatch`, `closestpair`, `build`, `dump`, `load`). Returns `202` with a `job_id`.
- `GET /api/v1/jobs/{job_id}` and `/result` — state and output.

## 5. How failures are reported

The engine raises typed errors, and this package maps them **by category**, never by message:

| engine category | HTTP | CLI exit code |
| :--- | ---: | ---: |
| `NotFound` | 404 | 4 |
| `ConflictingState` (staged backlog, untrained profile, empty project) | 409 | 5 |
| `InvalidRequest` (wrong payload, wrong dimension, bad option) | 400 | 2 |
| `StorageFailure` | 500 | 70 |

An error response carries `error` (the message, for a human) and `kind` (the concrete engine
type, for a client that wants to branch). A CLI's own refusals — no such dataset directory,
unreadable input file — keep exit code `1`, decided before the engine is ever called.

## 6. Configuration

`config.toml` (or `.similarity-search-server.toml`):

```toml
[paths]
workdir = "./workdir"

[server]
host = "127.0.0.1"
port = 8080

[resources]
query_threads_pct = 80
batch_threads_pct = 20
```

## 7. Tests

Two levels, selected with `test_args=["full"]` or `SSE_TEST_LEVEL=full`:

- **light** (50 assertions, ~5 s): the boundary with the engine — wire names to engine types,
  engine errors to status and exit codes, wire items to engine items — with no server started
  and no subprocess spawned. This is what catches an engine-side change on the day it lands.
- **full** (585 assertions, ~19 min): the API, job and CLI end-to-end suites, which start real
  servers and poll real subprocesses. It includes a stretch of some ten minutes with no output
  at all, near the end, where the CLI suite waits for job expiry — the run is not stuck.

```bash
julia --project=server -e 'using Pkg; Pkg.develop(path="."); Pkg.test()'
julia --project=server -e 'using Pkg; Pkg.test(test_args=["full"])'
```
