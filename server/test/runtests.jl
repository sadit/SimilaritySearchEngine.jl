using Test
using SimilaritySearchServer
using SimilaritySearchEngine
using TOML

# The selection of files is the engine's (`test/selection.jl` at the root of the repository).
include(joinpath(@__DIR__, "..", "..", "test", "selection.jl"))

# ---------------------------------------------------------------------------------------
# A light run on every change, a run of the files a change touches while working, and the full
# run before a release.
#
#     julia --project=server -e 'using Pkg; Pkg.test()'                                # light
#     julia --project=server -e 'using Pkg; Pkg.test(test_args=["jobs", "client"])'    # these
#     julia --project=server -e 'using Pkg; Pkg.test(test_args=["api/17", "cli/06"])'  # these
#     julia --project=server -e 'using Pkg; Pkg.test(test_args=["full"])'              # everything
#
# A file is selected by its name or by one of its tags. The word `light` adds the light files.
#
# The light files start nothing: no HTTP server, no subprocess. They check configuration and
# the functions between this package and the engine: which kind of project a wire name means,
# which status code and exit code an engine error becomes, what an incoming item turns into.
# An engine change breaks these functions first, so the light run fails on the day of that
# change. This is the reason both packages are in one repository.
#
# The other files are end-to-end. The `api/` files share one HTTP server, started by the first
# file that needs it, and the job routes start Julia subprocesses. Each file creates its own
# datasets, so any subset of the files can run alone. Measured 2026-11-23, the light run took
# 26 s (288 assertions) and the full run 37 min (1,273 assertions).
const SELECTION = TestSelection(ARGS)

const TEST_FILES = [
    TestFile("config", "test_config.jl"; tags=["config"]),
    TestFile("boundary", "test_boundary.jl"; tags=["errors", "auth"]),
    TestFile("dispatcher", "test_dispatcher.jl"; tags=["jobs"]),
    TestFile("e2e", "test_e2e.jl"; tags=["api", "jobs"], light=false),
    TestFile("api/01", "api/test_01_metric_search.jl"; tags=["api", "dense", "search"], light=false),
    TestFile("api/02", "api/test_02_full_text_search.jl"; tags=["api", "text", "search"], light=false),
    TestFile("api/03", "api/test_03_hybrid_and_meta.jl"; tags=["api", "hybrid", "filter", "search"], light=false),
    TestFile("api/04", "api/test_04_jobs_and_access.jl"; tags=["api", "jobs", "fetch"], light=false),
    TestFile("api/05", "api/test_05_dataset_admin_and_health.jl"; tags=["api", "admin", "health"], light=false),
    TestFile("api/06", "api/test_06_field_datasets.jl"; tags=["api", "text"], light=false),
    TestFile("api/07", "api/test_07_responses.jl"; tags=["api", "responses", "max_k", "gzip"], light=false),
    TestFile("api/08", "api/test_08_calibration_and_safety.jl"; tags=["api", "calibration", "dense"], light=false),
    TestFile("api/09", "api/test_09_rebuild_and_describe.jl"; tags=["api", "rebuild", "cli"], light=false),
    TestFile("api/10", "api/test_10_dataset_reload.jl"; tags=["api", "reload", "admin"], light=false),
    TestFile("api/11", "api/test_11_admin_jobs_and_dataset_reload.jl"; tags=["api", "jobs", "admin", "reload"], light=false),
    TestFile("api/12", "api/test_12_dump_load_via_admin.jl"; tags=["api", "dump", "admin", "jobs"], light=false),
    TestFile("api/13", "api/test_13_ctl_cli.jl"; tags=["api", "ctl", "jobs", "admin"], light=false),
    TestFile("api/14", "api/test_14_auth.jl"; tags=["api", "auth"], light=false),
    TestFile("api/15", "api/test_15_meta_schema.jl"; tags=["api", "schema", "filter"], light=false),
    TestFile("api/16", "api/test_16_partitions.jl"; tags=["api", "partitions", "text"], light=false),
    TestFile("api/17", "api/test_17_julia_client.jl"; tags=["api", "client", "jobs", "max_k"], light=false),
    TestFile("api/18", "api/test_18_followers.jl"; tags=["api", "followers", "partitions", "text", "dense", "ctl"], light=false),
    TestFile("api/19", "api/test_19_plugins.jl"; tags=["api", "followers", "plugins"], light=false),
    TestFile("api/20", "api/test_20_storage.jl"; tags=["api", "dense", "storage", "client"], light=false),
    TestFile("api/21", "api/test_21_node_client.jl"; tags=["api", "client", "node"], light=false),
    TestFile("cli/01", "cli/test_01_build.jl"; tags=["cli", "build"], light=false),
    TestFile("cli/02", "cli/test_02_search.jl"; tags=["cli", "search"], light=false),
    TestFile("cli/03", "cli/test_03_describe_rebuild_closestpair.jl"; tags=["cli", "rebuild"], light=false),
    TestFile("cli/04", "cli/test_04_dump_load.jl"; tags=["cli", "dump"], light=false),
    TestFile("cli/05", "cli/test_05_interactive.jl"; tags=["cli", "interactive"], light=false),
    TestFile("cli/06", "cli/test_06_partitions.jl"; tags=["cli", "partitions"], light=false),
    TestFile("cli/07", "cli/test_07_followers.jl"; tags=["cli", "followers", "partitions", "dump"], light=false),
    TestFile("cli/08", "cli/test_08_storage.jl"; tags=["cli", "storage", "rebuild", "dump"], light=false),
]

const CHOSEN = selected_files(SELECTION, TEST_FILES)
report_selection("SimilaritySearchServer", SELECTION, TEST_FILES, CHOSEN)

# The shared test server, started by the first file that calls `ensure_test_server`.
all(f -> f.light, CHOSEN) || include("test_server_helpers.jl")

@testset verbose = true "SimilaritySearchServer.jl" begin
    for f in CHOSEN
        include(f.path)
    end
end
