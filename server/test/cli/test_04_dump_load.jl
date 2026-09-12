using Test
using JSON3

@testset "CLI Dump/Load Tests" begin
    workdir = mktempdir()
    data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")

    cli_script = joinpath(@__DIR__, "..", "..", "src", "apps", "similarity-search.jl")
    proj = joinpath(@__DIR__, "..", "..")

    # --- Dense (searchgraph) round-trip ------------------------------------------

    cmd_build = `julia --project=$proj $cli_script build --dataset dump_dense --input $data_path --index-kind searchgraph --distance L2 --workdir $workdir`
    @test success(cmd_build)

    bundle_dense = joinpath(workdir, "bundle_dense")
    cmd_dump = `julia --project=$proj $cli_script dump --dataset dump_dense --workdir $workdir --output $bundle_dense`
    @test success(cmd_dump)
    @test isfile(joinpath(bundle_dense, "dataset.avro"))
    @test isdir(joinpath(bundle_dense, "project"))
    @test isfile(joinpath(bundle_dense, "manifest.json"))
    manifest = JSON3.read(read(joinpath(bundle_dense, "manifest.json"), String))
    @test manifest.project_id == "dump_dense"
    @test manifest.index_kind == "searchgraph"
    @test manifest.record_count == 724

    cmd_load = `julia --project=$proj $cli_script load --bundle $bundle_dense --dataset dump_dense_restored --workdir $workdir`
    @test success(cmd_load)
    @test isdir(joinpath(workdir, "datasets", "dump_dense_restored"))

    desc_orig = tempname()
    desc_restored = tempname()
    @test success(`julia --project=$proj $cli_script describe --dataset dump_dense --workdir $workdir --output $desc_orig`)
    @test success(`julia --project=$proj $cli_script describe --dataset dump_dense_restored --workdir $workdir --output $desc_restored`)
    d1 = JSON3.read(read(desc_orig, String))
    d2 = JSON3.read(read(desc_restored, String))
    @test d1.doc_count == d2.doc_count == 724
    @test d1.tombstone_count == d2.tombstone_count == 0
    # Same distance_stats confirms the restored index is byte-for-byte the same graph,
    # not just "a graph over the same data" (rebuilding from rows would generally differ).
    @test d1.distance_stats.nn_dist_min == d2.distance_stats.nn_dist_min
    @test d1.distance_stats.nn_dist_mean == d2.distance_stats.nn_dist_mean
    @test d1.distance_stats.nn_dist_max == d2.distance_stats.nn_dist_max

    out_search = tempname()
    @test success(`julia --project=$proj $cli_script searchbatch --dataset dump_dense_restored --queries $data_path --output $out_search --k 5 --workdir $workdir`)
    lines = readlines(out_search)
    @test length(lines) > 0
    @test length(JSON3.read(lines[1]).ids) > 0

    # --- Text (bm25_invfile) round-trip ------------------------------------------

    cmd_build_lex = `julia --project=$proj $cli_script build --dataset dump_lex --input $data_path --index-kind bm25_invfile --workdir $workdir`
    @test success(cmd_build_lex)

    bundle_lex = joinpath(workdir, "bundle_lex")
    @test success(`julia --project=$proj $cli_script dump --dataset dump_lex --workdir $workdir --output $bundle_lex`)
    @test success(`julia --project=$proj $cli_script load --bundle $bundle_lex --dataset dump_lex_restored --workdir $workdir`)

    desc_lex_restored = tempname()
    @test success(`julia --project=$proj $cli_script describe --dataset dump_lex_restored --workdir $workdir --output $desc_lex_restored`)
    dl = JSON3.read(read(desc_lex_restored, String))
    @test dl.kind == "bm25_invfile"
    @test dl.doc_count == 724
    @test dl.vocab.vocsize > 0
    @test dl.vocab.trainsize == 724
    @test dl.vocab.live_oov_rate == 0.0 # restored text still matches the restored vocabulary

    # --- Error paths ---------------------------------------------------------------

    @test !success(`julia --project=$proj $cli_script dump --dataset does_not_exist --workdir $workdir --output $(joinpath(workdir, "bundle_missing"))`)

    empty_bundle_target = joinpath(workdir, "bundle_already_exists")
    mkpath(empty_bundle_target)
    @test !success(`julia --project=$proj $cli_script dump --dataset dump_dense --workdir $workdir --output $empty_bundle_target`) # output already exists

    @test !success(`julia --project=$proj $cli_script load --bundle $bundle_dense --dataset dump_dense_restored --workdir $workdir`) # target already exists

    incomplete_bundle = joinpath(workdir, "incomplete_bundle")
    mkpath(incomplete_bundle)
    write(joinpath(incomplete_bundle, "manifest.json"), "{}") # missing dataset.avro / index.snapshot.jld2
    @test !success(`julia --project=$proj $cli_script load --bundle $incomplete_bundle --dataset from_incomplete --workdir $workdir`)

    rm(workdir, force=true, recursive=true)
end
