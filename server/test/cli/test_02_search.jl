using Test
using JSON3

@testset "CLI Search Tests" begin
    workdir = mktempdir()
    data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")

    cli_script = joinpath(@__DIR__, "..", "..", "src", "apps", "similarity-search.jl")
    proj = joinpath(@__DIR__, "..", "..")

    cmd_build = `julia --project=$proj $cli_script build --dataset test_ds --input $data_path --index-kind searchgraph --distance L2 --workdir $workdir`
    @test success(cmd_build)

    out_file = tempname()
    cmd_search = `julia --project=$proj $cli_script searchbatch --dataset test_ds --queries $data_path --output $out_file --k 5 --workdir $workdir`
    @test success(cmd_search)
    @test isfile(out_file)

    lines = readlines(out_file)
    @test length(lines) > 0
    parsed = JSON3.read(lines[1])
    @test haskey(parsed, :ids)
    @test haskey(parsed, :dists)
    @test length(parsed.ids) > 0

    rm(out_file, force=true)
    rm(workdir, force=true, recursive=true)
end
