using Test
using JSON3

@testset "CLI Describe/Rebuild/Closestpair Tests" begin
    workdir = mktempdir()
    data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")

    cli_script = joinpath(@__DIR__, "..", "..", "src", "apps", "similarity-search.jl")
    proj = joinpath(@__DIR__, "..", "..")

    # --- Dense (searchgraph) ---------------------------------------------------

    cmd_build = `julia --project=$proj $cli_script build --dataset desc_dense --input $data_path --index-kind searchgraph --distance L2 --workdir $workdir`
    @test success(cmd_build)

    desc_out = tempname()
    cmd_describe = `julia --project=$proj $cli_script describe --dataset desc_dense --workdir $workdir --output $desc_out`
    @test success(cmd_describe)
    desc = JSON3.read(read(desc_out, String))
    @test desc.id == "desc_dense"
    # `describe` now names the kind in the same vocabulary `--index-kind` takes
    @test desc.kind == "searchgraph"
    @test desc.doc_count == 724
    @test desc.tombstone_count == 0
    @test desc.tombstone_ratio == 0.0
    @test !isempty(desc.distance)
    @test desc.distance_stats.sample_size > 0
    @test desc.distance_stats.nn_dist_min >= 0
    @test desc.distance_stats.nn_dist_max >= desc.distance_stats.nn_dist_min
    @test desc.beamsearch_baseline.bsize > 0

    cp_out = tempname()
    cmd_cp = `julia --project=$proj $cli_script closestpair --dataset desc_dense --workdir $workdir --output $cp_out --min-k 8`
    @test success(cmd_cp)
    cp = JSON3.read(read(cp_out, String))
    @test cp.i != cp.j
    @test cp.dist >= 0

    cmd_rebuild = `julia --project=$proj $cli_script rebuild --dataset desc_dense --workdir $workdir`
    @test success(cmd_rebuild)

    desc_out2 = tempname()
    cmd_describe2 = `julia --project=$proj $cli_script describe --dataset desc_dense --workdir $workdir --output $desc_out2`
    @test success(cmd_describe2)
    desc2 = JSON3.read(read(desc_out2, String))
    @test desc2.doc_count == 724 # nothing was ever tombstoned -- rebuild is content-preserving here
    @test desc2.tombstone_count == 0

    # Rebuilt index is still independently searchable (fresh CLI process, on-disk snapshot only).
    out_file = tempname()
    cmd_search = `julia --project=$proj $cli_script searchbatch --dataset desc_dense --queries $data_path --output $out_file --k 5 --workdir $workdir`
    @test success(cmd_search)
    lines = readlines(out_file)
    @test length(lines) > 0
    parsed = JSON3.read(lines[1])
    @test length(parsed.ids) > 0

    # --- Text (bm25_invfile) ----------------------------------------------------

    cmd_build_lex = `julia --project=$proj $cli_script build --dataset desc_lex --input $data_path --index-kind bm25_invfile --workdir $workdir`
    @test success(cmd_build_lex)

    desc_lex_out = tempname()
    cmd_describe_lex = `julia --project=$proj $cli_script describe --dataset desc_lex --workdir $workdir --output $desc_lex_out`
    @test success(cmd_describe_lex)
    desc_lex = JSON3.read(read(desc_lex_out, String))
    @test desc_lex.kind == "bm25_invfile"
    @test desc_lex.doc_count == 724
    @test desc_lex.vocab.vocsize > 0
    @test desc_lex.vocab.trainsize == 724
    @test length(desc_lex.vocab.top_tokens) > 0
    # Same corpus trained AND described -- nothing should be out-of-vocabulary yet.
    @test desc_lex.vocab.live_oov_rate == 0.0

    cmd_rebuild_lex = `julia --project=$proj $cli_script rebuild --dataset desc_lex --workdir $workdir`
    @test success(cmd_rebuild_lex)

    desc_lex_out2 = tempname()
    cmd_describe_lex2 = `julia --project=$proj $cli_script describe --dataset desc_lex --workdir $workdir --output $desc_lex_out2`
    @test success(cmd_describe_lex2)
    desc_lex2 = JSON3.read(read(desc_lex_out2, String))
    @test desc_lex2.doc_count == 724

    # closestpair is meaningless against a text (non-dense) index -- must fail cleanly.
    cp_lex_out = tempname()
    cmd_cp_lex = `julia --project=$proj $cli_script closestpair --dataset desc_lex --workdir $workdir --output $cp_lex_out`
    @test !success(cmd_cp_lex)

    # --- Error paths --------------------------------------------------------------

    cmd_describe_missing = `julia --project=$proj $cli_script describe --dataset does_not_exist --workdir $workdir --output $(tempname())`
    @test !success(cmd_describe_missing)

    cmd_rebuild_missing = `julia --project=$proj $cli_script rebuild --dataset does_not_exist --workdir $workdir`
    @test !success(cmd_rebuild_missing)

    rm(workdir, force=true, recursive=true)
end
