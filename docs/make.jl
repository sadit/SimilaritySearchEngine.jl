using Documenter, SimilaritySearchEngine, SimilaritySearchServer
using SimilaritySearchEngine: Errors, Schema, Project, IndexEngine, Persistence

# One site for one repository, assembled by publish-docs.sh:
#
#     /       the Quarto manual (manual/), prose for both packages
#     /api/   this reference, generated from the docstrings of both packages
#
# GitHub Pages serves one site per repository, so the two packages share it; the structure
# above is what makes that a feature rather than an accident. Hence `canonical` pointing at
# the subdirectory this build is mounted under, and hence no `deploydocs()` call: Documenter's
# deployment writes its own redirect at the branch root, which is where the manual's landing
# page lives now. publish-docs.sh is the one publishing path.
makedocs(;
    modules=[SimilaritySearchEngine, Errors, Schema, Project, IndexEngine, Persistence,
             SimilaritySearchServer, SimilaritySearchServer.Server, SimilaritySearchServer.Jobs,
             SimilaritySearchServer.Executors, SimilaritySearchServer.Tokens,
             SimilaritySearchServer.Telemetry, SimilaritySearchServer.Cursors],
    authors="Eric S. Tellez",
    repo="https://github.com/sadit/SimilaritySearchEngine.jl/blob/{commit}{path}#L{line}",
    sitename="SimilaritySearchEngine.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://sadit.github.io/SimilaritySearchEngine.jl/api",
        assets=String[],
        size_threshold=500_000,
        size_threshold_warn=300_000,
    ),
    pages=[
        "Home" => "index.md",
        "Engine API" => "engine.md",
        "Server API" => "server.md",
        # No entry for the manual: Documenter validates `pages` as file paths and rejects a
        # URL ("'https:/…/manual/' is not an existing page!"), so the links to it live in
        # index.md instead, which the sidebar always reaches as "Home".
    ],
    doctest=false,
    warnonly=true
)
