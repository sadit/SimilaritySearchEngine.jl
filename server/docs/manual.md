# Technical Manual: SimilaritySearchServer

`SimilaritySearchServer` is a high-performance asynchronous server designed to expose the functionality of `SimilaritySearch.jl` and `TextSearch.jl` through a RESTful API and a Command Line Interface (CLI).

## 1. System Architecture

The server is structured into 6 main layers:

1. **Core Infrastructure and CLI (`config.jl`, `cli.jl`)**: Manages the loading of the global TOML configuration and parses commands (e.g., `serve`, `build`, `searchbatch`).
2. **Model and Persistence (`schema.jl`, `dataset.jl`, `persistence.jl`)**: 
   - Uses `RocksDB` for disk storage, logically dividing data via *Column Families*.
   - Supports a dynamic `MetaSchema` per dataset, storing explicit fields and extra metadata in JSON format.
3. **Telemetry and Access Control (`telemetry.jl`, `tokens.jl`)**:
   - `op_log`: A central table in RocksDB to keep an operation history with UTC timestamps.
   - `TokenManager`: Access control list (ACL) based on tokens.
4. **Index Engine (`index_engine.jl`)**:
   - Wraps the underlying indices (`SearchGraph`, `BM25InvertedFile`, etc.).
   - Introduces support for **Soft Deletes**, filtering logically deleted results (via an in-memory `Set`) without impacting search performance.
5. **Job Spooling and Executors (`jobs.jl`, `executors.jl`)**:
   - Manages heavy batch jobs without blocking the server.
   - Uses an atomic directory tree (`queued/`, `running/`, `completed/`, etc.) with `.job.toml` files, launching independent child processes (`LocalCLIExecutor`).
6. **HTTP API (`server.jl`)**: Asynchronous server managed via `HTTP.jl`.

## 2. REST API (Base Endpoints)

The server natively exposes the following HTTP API:

### Control
- `GET /readyz`: Returns the health status of the server (HTTP 200 OK).

### Datasets
- `POST /api/v1/datasets`: Initializes a new dataset. Returns its UUID (64-bit hexadecimal).
- `DELETE /api/v1/datasets/{id}`: Destroys the specified dataset.

### Search and Indices
- `POST /api/v1/simsearch/{id}/append`: Appends an item or document to an existing index.
- `POST /api/v1/simsearch/{id}/search`: Performs a similarity query over an index.
- `POST /api/v1/simsearch/{id}/delete`: Logically deletes a document from the index (Soft Delete).

### Asynchronous Jobs (Job Spooling)
- `POST /api/v1/jobs/{kind}`: Submits a new job (e.g., `searchbatch`, `build`). Returns `202 Accepted` with the `job_id`.
- `GET /api/v1/jobs/{job_id}`: Queries the state (`queued`, `running`, `completed`, `failed`) of a job in the Spooling system.

## 3. Configuration

The base configuration file (`config.toml` or `.similarity-search-server.toml`) supports options such as:

```toml
workdir = "/path/to/data"

[server]
host = "127.0.0.1"
port = 8080

[threading]
query_pct = 0.8
```
