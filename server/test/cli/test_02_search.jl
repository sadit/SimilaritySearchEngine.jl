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

    # --edit-correction, on a text dataset: `Frankenstien` exchanges two letters of
    # `Frankenstein`, which the vocabulary holds and no case or diacritic fold reaches.
    cmd_build_lex = `julia --project=$proj $cli_script build --dataset lex_ds --input $data_path --index-kind bm25_invfile --workdir $workdir`
    @test success(cmd_build_lex)
    typo_queries = tempname()
    write(typo_queries, JSON3.write(Dict("text" => "Frankenstien")) * "\n")
    searched(flags...) = begin
        out = tempname()
        ok = success(`julia --project=$proj $cli_script searchbatch --dataset lex_ds --queries $typo_queries --output $out --k 5 --workdir $workdir $flags`)
        ok ? JSON3.read(only(readlines(out))).ids : nothing
    end
    @test isempty(searched())
    @test !isempty(searched("--edit-correction"))

    # A dataset without text refuses it with the exit code of an invalid request.
    proc = run(ignorestatus(`julia --project=$proj $cli_script searchbatch --dataset test_ds --queries $data_path --output $(tempname()) --k 5 --workdir $workdir --edit-correction`))
    @test proc.exitcode == 2

    rm(typo_queries, force=true)
    rm(workdir, force=true, recursive=true)
end
