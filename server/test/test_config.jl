@testset "configuration" begin
    temp_config = tempname() * ".toml"
    config = SimilaritySearchServer.load_config(temp_config)

    @test haskey(config, "paths")
    @test config["paths"]["workdir"] == "./workdir"
    @test config["resources"]["query_threads_pct"] == 80
    @test config["resources"]["batch_threads_pct"] == 20

    rm(temp_config, force=true)
end
