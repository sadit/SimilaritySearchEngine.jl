# AGENTS.md

Guidance for coding agents working in this repository.

## What is here

Two packages, one repository:

- **`SimilaritySearchEngine`** (repository root) -- an embedded similarity search engine:
  transactional, persistent storage and nearest-neighbour search over dense vectors, sparse
  representations and full-text documents. No HTTP, no CLI, no daemon. `src/embedded.jl` is
  its public surface; `IndexEngine`, `Persistence`, `Project` and `Schema` are the submodules
  underneath.
- **`SimilaritySearchServer`** (`server/`) -- everything that exposes the engine to something
  other than a Julia process: the REST API (`server/src/server.jl`, Oxygen), the CLIs
  (`server/src/cli_handlers.jl`, `server/src/ctl_handlers.jl`, ArgParse), the job model
  (`server/src/jobs.jl`), tokens, cursors, telemetry, and the three installable apps in
  `server/src/apps/`.

`server/PLAN.md` is the system specification the server implements (REST routes, job model,
pagination, tokens). `DEVELOPMENT_STRATEGY.md` at the root says why the split is where it is,
how the two stay in sync, and where the error boundary lives -- read it before moving code
across that line.

## Working in both at once

The two are developed together. A change to an exported engine name, a keyword or an error
type is not finished until the server compiles and both suites pass in the same commit.

```julia
# engine
julia --project=. -t auto -e 'using Pkg; Pkg.test()'

# server (the engine comes from this repository, not the registry)
julia --project=server -e 'using Pkg; Pkg.develop(path=".")'
julia --project=server -t auto -e 'using Pkg; Pkg.test()'
```

Julia 1.10 or later; 1.12 is what development happens on, and CI runs both ends.

## Things worth knowing before changing something

- **Errors are typed in the engine and mapped at each boundary** (`src/errors.jl`). The engine
  raises `EngineError` subtypes and knows nothing about HTTP status codes or exit codes; the
  server and the CLIs each map categories to their own vocabulary. Do not classify by message
  text, and do not teach the engine about its consumers.
- **Defaults live on the public surface and nowhere below it**, enforced by `test/policy.jl`.
- **No `Manifest.toml` is committed**, in either package: both resolve from the registry, and
  the base libraries are developed by path only on a developer's machine.
- **Integer storage keys are big-endian**, always (`Schema.be_key`), because RocksDB compares
  keys bytewise.
