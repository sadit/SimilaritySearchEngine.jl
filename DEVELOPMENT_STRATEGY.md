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
