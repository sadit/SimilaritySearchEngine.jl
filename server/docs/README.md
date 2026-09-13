# Documentation for `SimilaritySearchServer`

The server's manual and tutorial are pages of the project site, not files in this directory:

- Manual — <https://sadit.github.io/SimilaritySearchEngine.jl/server/manual.html>
- Tutorial — <https://sadit.github.io/SimilaritySearchEngine.jl/server/tutorial.html>
- API reference — <https://sadit.github.io/SimilaritySearchEngine.jl/api/server.html>

Their sources are `manual/server/manual.qmd` and `manual/server/tutorial.qmd` in the
repository root, alongside the engine's own pages. One repository with two packages gets one
site: GitHub Pages serves one site per repository, and documenting both halves in a single
manual — same navigation, same search, same vocabulary — is how the seam between them stays
visible. See `publish-docs.sh` for how the site is assembled and published.

`PLAN.md`, next to this directory, is the original design document: a historical record of
intent, not a description of what the package does today. The manual is that.
