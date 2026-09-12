# Development strategy

## One repository, two packages (as of 2026-09-12)

This repository holds two packages, and the split is the point:

```
SimilaritySearchEngine.jl/     the engine -- an embedded library, no HTTP, no CLI
└── server/                    SimilaritySearchServer -- REST API, CLIs, jobs, the three apps
```

They are **kept in sync on features and separate on failure handling**. One pull request
changes both when an engine change reaches the surface the server exposes; one CI run tests
both, so an engine rename breaks the server's build in the commit that caused it rather than
weeks later.

**Why they are not one package.** Loading the server's dependencies costs 1.32s and 28
transitive packages (Oxygen, HTTP, JLD2, ArgParse and their trees, measured 2026-09-12).
The engine's whole point is to be embeddable -- `using SimilaritySearchEngine` takes 1.45s
and a first search 2.3s, both of which a merged package would roughly double for a caller
who never serves a request. So: separate `Project.toml`s, separate dependency sets,
separate registration, one repository.

**Why they are not two repositories,** which is what they were until now: the server sat
three engine-breaking changes behind for twenty days, and nothing noticed. There is no
automation that makes two repositories notice each other; a single suite does it for free.

**How to apply.** Develop against `server/`'s environment with the engine developed by path
(`julia --project=server -e 'using Pkg; Pkg.develop(path=".")'`). A change that alters an
exported engine name, a keyword, or an error is not done until the server compiles and its
suite passes in the same commit. Release-wise the two are independent: the server declares a
`[compat]` range on the engine like any other dependency, and the registry records it as a
subdirectory package.

## Errors: typed in the engine, mapped at each boundary (as of 2026-09-12)

The engine raises typed exceptions (`EngineError` and its subtypes, `src/errors.jl`). It
does not know what an HTTP status code is, and it never will. Each consumer maps categories
to its own vocabulary:

- The HTTP server maps them to status codes -- not found to 404, a conflicting state to 409,
  an invalid request to 400, anything else to 500.
- The CLIs map them to exit codes, so a script can branch on the reason without parsing
  output.
- A caller using the engine as a library catches whichever type it cares about.

**Why typed rather than `error("...")`:** classifying by message text is what a consumer is
forced into otherwise, and it breaks silently the first time somebody improves the wording.
The category is the contract; the message is for humans.

## Audience: the server is a consumer, not the audience (as of 2026-08-25, still true in one repo)

`SimilaritySearchEngine.jl` **also expects people who reach for it as a Julia package** — a
script or an application that does `using SimilaritySearchEngine`, indexes its own data and
searches it, with no HTTP server and no CLI anywhere in the picture. That is what
`src/embedded.jl` is for, and the paragraph-search tutorial is written for exactly that reader.

Living in the same repository as the server does not make the server the audience. The two
travel together; the engine is still written for the reader who has it as a dependency and
nothing else.

**Why it changes decisions, not just framing:**

- **The public surface is published, not a seam.** What `SimilaritySearchEngine` exports is an
  API somebody outside this repository reads, calls and depends on. It has to be ergonomic on
  its own terms — which is also why that is exactly where defaults belong (see the policy
  below): a caller choosing `create_project(workdir, name)` should not have to name everything
  the internals name.
- **A breaking change costs more than a port.** Renaming an exported type or reshaping a
  keyword used to mean "update the server's call sites". It now also means somebody else's
  script stops working. Breaks are still on the table at `0.x` — the engine-kind redesign is
  one — but they are a cost to weigh rather than a free move, and they belong in a version
  bump and the README rather than in a quiet commit.
- **The submodules are not the API.** `IndexEngine`, `Persistence`, `Project` and `Schema` are
  reachable from outside, and the server does call into them, but the package's exports are
  what an outside user is invited to use. Anything a package user genuinely needs should be
  exported from `SimilaritySearchEngine` rather than reached for as `IndexEngine.something`.

**How to apply:** when a change touches an exported name or the shape of a public call, ask
what it does to a reader who has this package as a dependency and nothing else — not only what
it does to the server. When something is only for the server, keep it out of the package's
exports. And when a public function grows an argument, give it a default: the surface is where
defaults are supposed to live.

## Policy: no defaults below the public surface (as of 2026-08-25)

**Defaults live on the package's public surface — what `SimilaritySearchEngine` exports — and
nowhere below it.** Every function under that line takes every argument explicitly, positional
and keyword alike.

**Why:** a default is an answer to a question. Put one on an internal function and the same
question now has two answers in two places, and nothing tells you they disagree. This package
had exactly that: `create_project` took `distance=nothing` and `create_engine` took
`distance=SqL2()`, so the caller could not pass its own sentinel down without overriding the
other default, and the code arbitrated with a `?:` whose only purpose was to decide whose
default won:

```julia
engine = distance === nothing ?
    IndexEngine.create_engine(index_type; minrecall, textmodel, on_change) :
    IndexEngine.create_engine(index_type; distance, minrecall, textmodel, on_change)
```

The same shape appeared with `verbose`: this package's `fit_profile` adapter defaulted it to
`false` while the library's own method defaults it to `true`, so which one you got depended on
which layer you called. Neither was a bug yet. Both were a bug waiting for a reader.

**How to apply:** when an internal function needs a value it does not have, the answer is to
name it at the boundary and pass it down — not to guess. Where a default genuinely belongs to a
*kind* rather than to a call, give it a name of its own: `IndexEngine.default_distance(index_type)`
is a function, resolved once by `create_project`, instead of a default repeated per method.

Two exemptions, both narrow:

- **Qualified definitions** (`Base.show`, `TextSearch.fit_profile`) implement somebody else's
  contract, and the shape of that signature — defaults included — belongs to whoever owns it.
- **Public constructors of public types** are on the surface by definition (`TextItem(text;
  doc_id=nothing, ...)`).

`test/policy.jl` enforces this, parsing `src/` with Julia's own parser and failing with the file
and function name of any offender. It is a test rather than a note because a policy nobody
verifies drifts exactly the way the two cases above did — and because adding a default to an
internal function breaks nothing and reads like a convenience, so nothing else would catch it.

**Second level, not yet enforced:** the mechanical rule is *visibility* (not exported by the
package), which is checkable but not quite the real rule. The real rule is *direction of call*:
a function should not carry a default if all its callers are our own code — which includes
submodule APIs that `SimilaritySearchServer` happens to call, and excludes nothing. That pass
comes after this one is secured, and it needs judgement per function rather than a script.
