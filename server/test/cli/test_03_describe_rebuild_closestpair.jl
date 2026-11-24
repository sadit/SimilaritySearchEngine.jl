using Test
using JSON

@testset "CLI Describe/Rebuild/Closestpair Tests" begin
    workdir = mktempdir()
    data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")

    # Every command runs in this process (`_cli_ok`, `_cli_fails`). Each command opens the
    # project from its files and closes it.

    # --- Dense (searchgraph) ---------------------------------------------------

    @test _cli_ok("build", "--dataset", "desc_dense", "--input", data_path, "--index-kind", "searchgraph", "--distance", "L2", "--workdir", workdir)

    desc_out = tempname()
    @test _cli_ok("describe", "--dataset", "desc_dense", "--workdir", workdir, "--output", desc_out)
    desc = JSON.parse(read(desc_out, String))
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
    @test _cli_ok("closestpair", "--dataset", "desc_dense", "--workdir", workdir, "--output", cp_out, "--min-k", 8)
    cp = JSON.parse(read(cp_out, String))
    @test cp.i != cp.j
    @test cp.dist >= 0

    @test _cli_ok("rebuild", "--dataset", "desc_dense", "--workdir", workdir)

    desc_out2 = tempname()
    @test _cli_ok("describe", "--dataset", "desc_dense", "--workdir", workdir, "--output", desc_out2)
    desc2 = JSON.parse(read(desc_out2, String))
    @test desc2.doc_count == 724 # nothing was ever tombstoned -- rebuild is content-preserving here
    @test desc2.tombstone_count == 0

    # The rebuilt index is searchable: `searchbatch` opens it from its files.
    out_file = tempname()
    @test _cli_ok("searchbatch", "--dataset", "desc_dense", "--queries", data_path, "--output", out_file, "--k", 5, "--workdir", workdir)
    lines = readlines(out_file)
    @test length(lines) > 0
    parsed = JSON.parse(lines[1])
    @test length(parsed.ids) > 0

    # --- Text (bm25_invfile) ----------------------------------------------------

    @test _cli_ok("build", "--dataset", "desc_lex", "--input", data_path, "--index-kind", "bm25_invfile", "--workdir", workdir)

    desc_lex_out = tempname()
    @test _cli_ok("describe", "--dataset", "desc_lex", "--workdir", workdir, "--output", desc_lex_out)
    desc_lex = JSON.parse(read(desc_lex_out, String))
    @test desc_lex.kind == "bm25_invfile"
    @test desc_lex.doc_count == 724
    @test desc_lex.vocab.vocsize > 0
    @test desc_lex.vocab.trainsize == 724
    @test length(desc_lex.vocab.top_tokens) > 0
    # Same corpus trained AND described -- nothing should be out-of-vocabulary yet.
    @test desc_lex.vocab.live_oov_rate == 0.0

    @test _cli_ok("rebuild", "--dataset", "desc_lex", "--workdir", workdir)

    desc_lex_out2 = tempname()
    @test _cli_ok("describe", "--dataset", "desc_lex", "--workdir", workdir, "--output", desc_lex_out2)
    desc_lex2 = JSON.parse(read(desc_lex_out2, String))
    @test desc_lex2.doc_count == 724

    # closestpair is meaningless against a text (non-dense) index -- must fail cleanly.
    cp_lex_out = tempname()
    @test _cli_fails("closestpair", "--dataset", "desc_lex", "--workdir", workdir, "--output", cp_lex_out)

    # --- Error paths --------------------------------------------------------------

    @test _cli_fails("describe", "--dataset", "does_not_exist", "--workdir", workdir, "--output", tempname())
    @test _cli_fails("rebuild", "--dataset", "does_not_exist", "--workdir", workdir)

    rm(workdir, force=true, recursive=true)
end
