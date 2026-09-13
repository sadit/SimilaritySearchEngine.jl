using Test
using SimilaritySearchServer
using SimilaritySearchEngine
using TOML

# ---------------------------------------------------------------------------------------
# Two levels, the same switch the engine's own suite uses (see ../../test/runtests.jl):
#
#     julia --project=server -e 'using Pkg; Pkg.test()'                     # light
#     julia --project=server -e 'using Pkg; Pkg.test(test_args=["full"])'   # everything
#
# The split here is sharper than the engine's, because almost everything in this package's
# suite is end-to-end: the API and CLI testsets start a real HTTP server, spawn Julia
# subprocesses per job, and poll them to completion -- minutes, and the only tests in either
# package that depend on ports and process scheduling. What stays in the light run is what
# can be checked without any of that: configuration, and the pure functions that make up the
# boundary between this package and the engine (which kind of project a wire name means, which
# status code and exit code an engine error becomes, what an incoming item turns into).
#
# That boundary is exactly what breaks when the engine changes, so a light run is not a token
# gesture: it fails on the same day an engine rename lands, which is the reason both packages
# live in one repository.
const FULL = "full" in ARGS || lowercase(get(ENV, "SSE_TEST_LEVEL", "light")) == "full"

@testset verbose = true "SimilaritySearchServer.jl" begin
    @testset "configuration" begin
        temp_config = tempname() * ".toml"
        config = SimilaritySearchServer.load_config(temp_config)

        @test haskey(config, "paths")
        @test config["paths"]["workdir"] == "./workdir"
        @test config["resources"]["query_threads_pct"] == 80
        @test config["resources"]["batch_threads_pct"] == 20

        rm(temp_config, force=true)
    end

    include("test_boundary.jl")

    if FULL
        include("test_server_helpers.jl")
        include("test_e2e.jl")

        @testset "API Endpoints" begin
            include("api/test_01_metric_search.jl")
            include("api/test_02_full_text_search.jl")
            include("api/test_03_hybrid_and_meta.jl")
            include("api/test_04_jobs_and_access.jl")
            include("api/test_05_dataset_admin_and_health.jl")
            include("api/test_06_join_group_text_search.jl")
            include("api/test_07_cursors_and_pagination.jl")
            include("api/test_08_calibration_and_safety.jl")
            include("api/test_09_rebuild_and_describe.jl")
            include("api/test_10_dataset_reload.jl")
            include("api/test_11_admin_jobs_and_dataset_reload.jl")
            include("api/test_12_dump_load_via_admin.jl")
            include("api/test_13_ctl_cli.jl")
            include("api/test_14_auth.jl")
        end

        @testset "CLI Commands" begin
            include("cli/test_01_build.jl")
            include("cli/test_02_search.jl")
            include("cli/test_03_describe_rebuild_closestpair.jl")
            include("cli/test_04_dump_load.jl")
            include("cli/test_05_interactive.jl")
        end
    else
        @info "light run: the HTTP, job and CLI end-to-end suites were skipped; run with test_args=[\"full\"] before a release"
    end
end
