using Documenter, SimilaritySearchEngine
using SimilaritySearchEngine: Schema, Project, IndexEngine, Persistence

makedocs(;
    modules=[SimilaritySearchEngine, Schema, Project, IndexEngine, Persistence],
    authors="Eric S. Tellez",
    repo="https://github.com/sadit/SimilaritySearchEngine.jl/blob/{commit}{path}#L{line}",
    sitename="SimilaritySearchEngine.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://sadit.github.io/SimilaritySearchEngine.jl",
        assets=String[],
        size_threshold=500_000,
        size_threshold_warn=300_000,
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md"
    ],
    doctest=false,
    warnonly=true
)

deploydocs(;
    repo="github.com/sadit/SimilaritySearchEngine.jl",
    devbranch="master",
    devurl="dev",
    versions=["stable" => "v^", "v#.#", "dev" => "dev"],
    push_preview=true,
)
