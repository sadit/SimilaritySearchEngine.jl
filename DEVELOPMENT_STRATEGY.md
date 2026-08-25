# Development strategy (as of 2026-08-18)

Active development now happens **here first**, in `SimilaritySearchEngine.jl` — `Dataset`,
`Schema`, `IndexEngine`, `Persistence`, and the embedded API (`src/embedded.jl`). Fine-grained
work (bug fixes, API refinements, new engine-level functionality, test hardening) lands in
this package first.

Once a piece of work here is settled, it gets **ported over to `SimilaritySearchServer`**
(the sibling package at `../SimilaritySearchServer`, which depends on this one via
`Pkg.develop`) — updating its own call sites, HTTP handlers, and CLI commands to match
whatever changed here, and re-running its full test suite to confirm nothing broke.

This reverses the direction development had been going in until now: previously, every
chunk landed directly in `SimilaritySearchServer`, and this package only came to exist
(2026-08-18) as an extraction *out of* it (PLAN.md §8.5 in `SimilaritySearchServer`). Going
forward, this package is the primary workspace, and `SimilaritySearchServer` is the
downstream consumer that gets updated afterward, not the other way around.

**Why:** the user made this call explicitly (`"vamos a cambiar un poco la estrategia de
desarrollo; nos enfocaremos en SimilaritySearchEngine con detalles finos, y luego
portaremos a SimilaritySearchServer"`) after the initial extraction + embedded-API chunk
landed. No specific reason was given beyond wanting fine-detail work to happen at the
engine level before flowing downstream — treat this as the standing default until told
otherwise.

**How to apply:** when picking up new work with no other explicit target, default to
scoping it inside `SimilaritySearchEngine.jl` first. Only touch `SimilaritySearchServer`
once the corresponding engine-level change is done, to port it through (update its
`Pkg.develop`-linked dependency if the version/API shape changed, fix any call sites,
re-run its suite). Don't restart both packages' work in lockstep by default — the engine
leads, the server follows.

## Audience: `SimilaritySearchServer` is a consumer, not the audience (as of 2026-08-25)

`SimilaritySearchEngine.jl` **also expects people who reach for it as a Julia package** — a
script or an application that does `using SimilaritySearchEngine`, indexes its own data and
searches it, with no HTTP server and no CLI anywhere in the picture. That is what
`src/embedded.jl` is for, and the paragraph-search tutorial is written for exactly that reader.

The section above says the engine leads and the server follows, and that stays true about
*where work lands*. It should not be read as saying the server is who the engine is for.

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
