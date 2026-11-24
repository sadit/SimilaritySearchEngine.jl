using Test
using JSON
using SimilaritySearchEngine

@testset "CLI Dump/Load Tests" begin
    workdir = mktempdir()
    data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")

    cli_script = joinpath(@__DIR__, "..", "..", "src", "apps", "similarity-search.jl")
    proj = joinpath(@__DIR__, "..", "..")

    # --- Dense (searchgraph) round-trip ------------------------------------------

    cmd_build = `julia --project=$proj $cli_script build --dataset dump_dense --input $data_path --index-kind searchgraph --distance L2 --workdir $workdir --max-k 250`
    @test success(cmd_build)

    bundle_dense = joinpath(workdir, "bundle_dense")
    cmd_dump = `julia --project=$proj $cli_script dump --dataset dump_dense --workdir $workdir --output $bundle_dense`
    @test success(cmd_dump)
    @test isfile(joinpath(bundle_dense, "dataset.jsonl"))
    @test isdir(joinpath(bundle_dense, "project"))
    @test isfile(joinpath(bundle_dense, "manifest.json"))
    manifest = JSON.parse(read(joinpath(bundle_dense, "manifest.json"), String))
    @test manifest.project_id == "dump_dense"
    @test manifest.index_kind == "searchgraph"
    @test manifest.record_count == 724
    # `max_k` (CHANGELOG.md §5.16) goes from `build` to the bundle, and from the bundle to the server.
    @test manifest.max_k == 250

    cmd_load = `julia --project=$proj $cli_script load --bundle $bundle_dense --dataset dump_dense_restored --workdir $workdir`
    @test success(cmd_load)
    @test isdir(joinpath(workdir, "datasets", "dump_dense_restored"))
    @test JSON.parse(read(joinpath(workdir, "datasets", "dump_dense_restored", "descriptor.json"))).max_k == 250

    desc_orig = tempname()
    desc_restored = tempname()
    @test success(`julia --project=$proj $cli_script describe --dataset dump_dense --workdir $workdir --output $desc_orig`)
    @test success(`julia --project=$proj $cli_script describe --dataset dump_dense_restored --workdir $workdir --output $desc_restored`)
    d1 = JSON.parse(read(desc_orig, String))
    d2 = JSON.parse(read(desc_restored, String))
    @test d1.doc_count == d2.doc_count == 724
    @test d1.max_k == d2.max_k == 250
    @test d1.tombstone_count == d2.tombstone_count == 0
    # Same distance_stats confirms the restored index is byte-for-byte the same graph,
    # not just "a graph over the same data" (rebuilding from rows would generally differ).
    @test d1.distance_stats.nn_dist_min == d2.distance_stats.nn_dist_min
    @test d1.distance_stats.nn_dist_mean == d2.distance_stats.nn_dist_mean
    @test d1.distance_stats.nn_dist_max == d2.distance_stats.nn_dist_max

    # --- The metadata of a bundle are JSON lines (CHANGELOG.md §5.16, phase 9c) -----------

    lines = readlines(joinpath(bundle_dense, "dataset.jsonl"))
    @test length(lines) == 724 && !isfile(joinpath(bundle_dense, "dataset.avro"))
    first_item = JSON.parse(lines[1])
    @test first_item._id == 1 && haskey(first_item, :meta) && first_item.deleted == false
    first_restored_item(id) = begin
        h = SimilaritySearchEngine.open_project(joinpath(workdir, "datasets"), id; read_only=true)
        it = only(SimilaritySearchEngine.fetch_items(h, [1]))
        SimilaritySearchEngine.close_project!(h)
        it
    end

    # A line edited with a new `meta` is restored by `load`.
    edited_bundle = joinpath(workdir, "bundle_edited")
    cp(bundle_dense, edited_bundle)
    edited = JSON.parse(lines[1]; dicttype=Dict{String,Any})
    edited["meta"] = Dict("note" => "edited")
    write(joinpath(edited_bundle, "dataset.jsonl"), join(vcat(JSON.json(edited), lines[2:end]), "\n") * "\n")
    code, out = _run_cli(["load", "--bundle", edited_bundle, "--dataset", "dump_edited", "--workdir", workdir])
    @test code == 0 && occursin("724 line(s)", out)
    @test first_restored_item("dump_edited").meta == Dict("note" => "edited")

    # A bundle without `dataset.jsonl`, and a bundle with only `dataset.avro` (written before
    # 2026-11-23), are loaded from `project/`, which holds the same metadata.
    no_lines = joinpath(workdir, "bundle_no_lines")
    cp(bundle_dense, no_lines)
    rm(joinpath(no_lines, "dataset.jsonl"))
    code, out = _run_cli(["load", "--bundle", no_lines, "--dataset", "dump_no_lines", "--workdir", workdir])
    @test code == 0 && occursin("0 line(s)", out)
    @test first_restored_item("dump_no_lines").doc_id == first_item.doc_id
    old_bundle = joinpath(workdir, "bundle_old")
    cp(no_lines, old_bundle)
    write(joinpath(old_bundle, "dataset.avro"), "Obj")
    code, out = _run_cli(["load", "--bundle", old_bundle, "--dataset", "dump_old", "--workdir", workdir])
    @test code == 0 && occursin("dataset.avro", out) && occursin("ignored", out)
    @test first_restored_item("dump_old").doc_id == first_item.doc_id

    # A line that cannot be read stops `load` before anything is copied.
    bad_bundle = joinpath(workdir, "bundle_bad")
    cp(no_lines, bad_bundle)
    write(joinpath(bad_bundle, "dataset.jsonl"), "not json\n")
    code, out = _run_cli(["load", "--bundle", bad_bundle, "--dataset", "dump_bad", "--workdir", workdir])
    @test code == 2 && occursin("line 1", out) && !ispath(joinpath(workdir, "datasets", "dump_bad"))

    # A dataset without items is dumped and loaded: its `dataset.jsonl` is empty.
    empty_input = joinpath(workdir, "empty.jsonl")
    touch(empty_input)
    @test first(_run_cli(["build", "--dataset", "dump_empty", "--input", empty_input, "--index-kind", "exhaustive_search",
                          "--workdir", workdir])) == 0
    bundle_empty = joinpath(workdir, "bundle_empty")
    @test first(_run_cli(["dump", "--dataset", "dump_empty", "--workdir", workdir, "--output", bundle_empty])) == 0
    @test filesize(joinpath(bundle_empty, "dataset.jsonl")) == 0
    @test JSON.parse(read(joinpath(bundle_empty, "manifest.json"), String)).record_count == 0
    @test first(_run_cli(["load", "--bundle", bundle_empty, "--dataset", "dump_empty_restored", "--workdir", workdir])) == 0
    @test isdir(joinpath(workdir, "datasets", "dump_empty_restored"))

    out_search = tempname()
    @test success(`julia --project=$proj $cli_script searchbatch --dataset dump_dense_restored --queries $data_path --output $out_search --k 5 --workdir $workdir`)
    lines = readlines(out_search)
    @test length(lines) > 0
    @test length(JSON.parse(lines[1]).ids) > 0

    # --- Text (bm25_invfile) round-trip ------------------------------------------

    cmd_build_lex = `julia --project=$proj $cli_script build --dataset dump_lex --input $data_path --index-kind bm25_invfile --workdir $workdir`
    @test success(cmd_build_lex)

    bundle_lex = joinpath(workdir, "bundle_lex")
    @test success(`julia --project=$proj $cli_script dump --dataset dump_lex --workdir $workdir --output $bundle_lex`)
    @test JSON.parse(read(joinpath(bundle_lex, "manifest.json"), String)).max_k == 100
    @test success(`julia --project=$proj $cli_script load --bundle $bundle_lex --dataset dump_lex_restored --workdir $workdir`)

    desc_lex_restored = tempname()
    @test success(`julia --project=$proj $cli_script describe --dataset dump_lex_restored --workdir $workdir --output $desc_lex_restored`)
    dl = JSON.parse(read(desc_lex_restored, String))
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
    write(joinpath(incomplete_bundle, "manifest.json"), "{}") # missing project/
    @test !success(`julia --project=$proj $cli_script load --bundle $incomplete_bundle --dataset from_incomplete --workdir $workdir`)

    rm(workdir, force=true, recursive=true)
end
