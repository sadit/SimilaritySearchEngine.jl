using Test
using JSON3

@testset "CLI Build Tests" begin
    workdir = mktempdir()
    data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")

    cli_script = joinpath(@__DIR__, "..", "..", "src", "apps", "similarity-search.jl")
    proj = joinpath(@__DIR__, "..", "..")

    # 1. Build Dense Index
    cmd_dense = `julia --project=$proj $cli_script build --dataset frank_dense --input $data_path --index-kind searchgraph --distance L2 --workdir $workdir`
    @test success(cmd_dense)
    @test isdir(joinpath(workdir, "frank_dense"))
    @test isfile(joinpath(workdir, "frank_dense", "CURRENT"))   # the project persists itself

    # 2. Build Lexical Index (BM25 indexing works; only BM25 *search* is broken
    # upstream in TextSearch.jl, see PLAN.md checkpoint notes)
    cmd_lex = `julia --project=$proj $cli_script build --dataset frank_lex --input $data_path --index-kind bm25_invfile --workdir $workdir`
    @test success(cmd_lex)
    @test isdir(joinpath(workdir, "frank_lex"))
    @test isfile(joinpath(workdir, "frank_lex", "CURRENT"))

    rm(workdir, force=true, recursive=true)
end
