include("selection.jl")
include("testutils.jl")

# ---------------------------------------------------------------------------------------
# A light run on every change, a run of the files a change touches while working, and the full
# run before a release.
#
#     julia --project -e 'using Pkg; Pkg.test()'                             # light (the default)
#     julia --project -e 'using Pkg; Pkg.test(test_args=["text", "search"])' # these files only
#     julia --project -e 'using Pkg; Pkg.test(test_args=["full"])'           # everything
#     SSE_TEST_LEVEL=full julia --project -e 'using Pkg; Pkg.test()'         # same, from CI
#
# `selection.jl` says how a name or a tag selects a file. Every file of the engine is light: the
# light run includes all of them and skips their heavy testsets. A run that names its files
# runs the heavy testsets of those files.
#
# The heavy testsets are chosen by their duration. Measured 2026-09-12, the whole suite took
# 90 s, and five testsets took about 50 s of it: the whole-dataset algorithms (29.2 s), the
# concurrency test, and the text testsets that fit a profile from a corpus (7.2 s, 5.7 s,
# 2.5 s). The light run covers the same code on smaller inputs, in about one second each. The
# full run adds larger inputs, real corpora, and concurrent access.
#
# A skipped testset is named at the end of the run. A light run must not look identical to a
# full one.
const SELECTION = TestSelection(ARGS)
const HEAVY = run_heavy(SELECTION)
const SKIPPED_HEAVY = String[]

macro heavy(name, body)
    # The whole quote is escaped rather than each argument: `@testset` wants its block as a
    # literal `begin ... end`, and an `esc(...)`-wrapped one is not something it recognizes.
    esc(quote
        if HEAVY
            @testset $name $body
        else
            push!(SKIPPED_HEAVY, $name)
        end
    end)
end

const TEST_FILES = [
    TestFile("policy", "policy.jl"; tags=["api"]),
    TestFile("dense", "dense.jl"; tags=["dense", "calibration"]),
    TestFile("whole_dataset", "whole_dataset.jl"; tags=["dense", "jobs", "concurrency"]),
    TestFile("sparse", "sparse.jl"; tags=["sparse"]),
    TestFile("projects", "projects.jl"; tags=["dense", "sparse", "text"]),
    TestFile("text", "text.jl"; tags=["text", "vocabulary"]),
    TestFile("persistence", "persistence.jl"; tags=["persistence", "dense", "text"]),
    TestFile("doc_ids", "doc_ids.jl"; tags=["doc_ids"]),
    TestFile("filters_and_schema", "filters_and_schema.jl"; tags=["filter", "schema"]),
    TestFile("search", "search.jl"; tags=["search", "dense"]),
    TestFile("text_partitions", "text_partitions.jl"; tags=["partitions", "text"]),
    TestFile("dense_partitions", "dense_partitions.jl"; tags=["partitions", "dense"]),
]

const CHOSEN = selected_files(SELECTION, TEST_FILES)
report_selection("SimilaritySearchEngine", SELECTION, TEST_FILES, CHOSEN)

@testset verbose = true "SimilaritySearchEngine.jl" begin
    for f in CHOSEN
        include(f.path)
    end
    isempty(SKIPPED_HEAVY) || @info "light run: $(length(SKIPPED_HEAVY)) heavy testset(s) skipped; run with test_args=[\"full\"] before a release" SKIPPED_HEAVY
end
