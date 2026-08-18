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
