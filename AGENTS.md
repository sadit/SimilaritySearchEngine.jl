# AGENTS.md

Guidance for coding agents working in this repository.

## What is here

Three packages, one repository:

- **`SimilaritySearchEngine`** (repository root) -- an embedded similarity search engine:
  transactional, persistent storage and nearest-neighbour search over dense vectors, sparse
  representations and full-text documents. No HTTP, no CLI, no daemon. `src/embedded.jl` is
  its public surface; `IndexEngine`, `Persistence`, `Project` and `Schema` are the submodules
  underneath.
- **`SimilaritySearchServer`** (`server/`) -- everything that exposes the engine to something
  other than a Julia process: the REST API (`server/src/server.jl`, Oxygen), the CLIs
  (`server/src/cli_handlers.jl`, `server/src/ctl_handlers.jl`, ArgParse), the job model
  (`server/src/jobs.jl`), tokens, telemetry, and the three installable apps in
  `server/src/apps/`.
- **`SimilaritySearchClient`** (`server/julia/`) -- the Julia client of the HTTP API. Its
  functions have the names and the result types of the engine's functions, and it depends on
  HTTP.jl and JSON.jl only: it must never depend on the engine or the server. It declares its
  own copy of the engine's error types, and the server's light suite checks that the two lists
  are equal. Its tests against a running server are in the server's full suite.

`PLAN.md` describes the three packages on the date of its last cleanup, the decisions that
hold, and the planned work. Its first section, "Development Strategy", says why the packages are
separate, how they stay in sync, and where the errors are mapped. Read it before you move code
from one package to another. `CHANGELOG.md` keeps the history: the commits by date, and every
design with its phases and measurements. The code cites a design as `CHANGELOG.md §5.N`, or as
`PLAN.md §5.N` while the design is planned. "How This File Is Kept" in `PLAN.md` says how a
cleanup moves text from `PLAN.md` to `CHANGELOG.md`.

## Working in both at once

The two are developed together. A change to an exported engine name, a keyword or an error
type is not finished until the server compiles and both suites pass in the same commit.

```julia
# engine
julia --project=. -t auto -e 'using Pkg; Pkg.test()'

# server (the engine comes from this repository, not the registry)
julia --project=server -e 'using Pkg; Pkg.develop(path=".")'
julia --project=server -t auto -e 'using Pkg; Pkg.test()'

# Julia client (no server needed; the tests against a server are in the server's full suite)
julia --project=server/julia -e 'using Pkg; Pkg.test()'

# Node.js client, Node.js 22 or later (no server needed; api/21 of the server's full suite
# runs its tests against a server)
cd server/node && npm install && npm test
```

While working, run only the test files a change touches: `Pkg.test(test_args=["jobs", "api/17"])`
selects files by name or tag (`TEST_FILES` in each `runtests.jl`, `test/selection.jl`). The
server's full suite took 13 minutes on 2026-11-24. Run it with `test_args=["full"]` before a release
and before a commit that changes the job model, the dispatcher or the shared test server.

A new engine error type is not finished until the Julia client has a type of the same name, in
the same category (`server/julia/src/errors.jl`), and the Node.js client has a class of the same
name, in the same category and in `ERROR_KINDS` (`server/node/src/errors.ts`). The light suite of
the server checks both.

Julia 1.12 or later, in every package: it is what CI verifies, and the server's `[apps]`
entries need `Pkg.Apps`, which does not exist before it.

## How the documentation is written

The manual, the tutorials and the docstrings are read by people who do not read English
natively. Write for them:

- **No analogies, no metaphors, no figurative verbs.** State the technical fact. Not "the seam
  between the two packages" but "the public API of the engine"; not "what a recall target buys
  is paid for in distance computations" but "a higher recall target requires more distance
  computations". A figure of speech is one more step for a reader who is already translating,
  and it adds nothing to the plain statement.
- **One fact per sentence.** Keep sentences under about 35 words, and prefer subject-verb-object
  order. Split a sentence rather than joining its parts with a dash, a semicolon or a
  parenthetical aside; the aside forces the reader to hold two ideas at once.
- **No superlatives and no evaluative adjectives.** Not "high-quality stopword lists",
  "comprehensive analyses", "first-class engine", "the entire point of calibration", "not small
  at scale". If something is better, say by how much, with the measurement.
- **Ordinary vocabulary.** Avoid leverage, encapsulate, agnostic, orchestrate, hydrate, splice,
  amortize, provenance, and similar terms when a common word exists. Technical terms of the
  domain (posting list, beam search, column family, write lock) are not in this category: use
  them, and define each one the first time it appears.
- **Section titles name a topic**, not a claim: "Measuring the Cost of a Query", not "Cost
  Accounting Across the Pool Boundary".
- **Numbers, not adjectives.** A performance statement carries the measurement and the date it
  was taken, as the existing pages do.

This applies to prose. Code examples, tables, endpoint lists and measurements are unchanged by
it.

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
- **Documentation is one site for both packages.** Prose lives in `manual/` as Quarto pages --
  the engine's at the top level, the server's under `manual/server/` -- and the reference is
  generated by `docs/make.jl` from the docstrings of both packages (`docs/src/engine.md`,
  `docs/src/server.md`). `publish-docs.sh` mounts the manual at `/` and the reference at
  `/api/` of the one Pages site this repository gets. The rendered manual is committed under
  `manual/docs/`, and it is re-rendered **one page at a time** (`quarto render manual/x.qmd`),
  never as a project. A user-visible change to either package is not finished until the page
  that describes it is updated.
