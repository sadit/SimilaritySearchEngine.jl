```@meta
CurrentModule = SimilaritySearchServer
DocTestSetup = quote
    using SimilaritySearchServer
end
```

# SimilaritySearchServer API

The service layer: the HTTP API, the job spool, the two command lines, and the supporting
machinery for access tokens, telemetry and result cursors. It is a separate package, in
`server/` of this same repository, and it reaches the engine only through the public API
documented under [Engine API](engine.md) -- the same surface any other caller has.

Prose first, if you are arriving here cold: the [server manual](https://sadit.github.io/SimilaritySearchEngine.jl/server/manual.html)
explains how a dataset is held and how failures are reported, and the
[server tutorial](https://sadit.github.io/SimilaritySearchEngine.jl/server/tutorial.html)
walks a corpus from `create` to a query over HTTP.

## Package-level API

Configuration, the `similarity-search` and `similarity-search-ctl` command lines, and the
handlers each subcommand dispatches to.

```@autodocs
Modules = [SimilaritySearchServer]
Order   = [:type, :function, :constant, :macro]
```

## `Server`

The HTTP layer: `AppState` (one `EmbeddedEngine` per dataset), the endpoint handlers, and the
mapping from the engine's typed errors to status codes.

```@autodocs
Modules = [SimilaritySearchServer.Server]
Order   = [:type, :function, :constant]
```

## `Jobs` and `Executors`

The job spool -- an atomic directory tree of `.job.toml` files -- and the executor that
dispatches each job as an independent Julia subprocess.

```@autodocs
Modules = [SimilaritySearchServer.Jobs, SimilaritySearchServer.Executors]
Order   = [:type, :function, :constant]
```

## `Tokens`, `Telemetry` and `Cursors`

Access tokens, the `op_log` operation history (whose `distance_evaluations` figure comes from
the engine's [`SearchStats`](engine.md#SimilaritySearchEngine.SearchStats)), and server-side
cursors for paging a materialized result set.

```@autodocs
Modules = [SimilaritySearchServer.Tokens, SimilaritySearchServer.Telemetry, SimilaritySearchServer.Cursors]
Order   = [:type, :function, :constant]
```
