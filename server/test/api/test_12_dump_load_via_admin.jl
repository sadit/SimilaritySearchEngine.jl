using Test
using HTTP
using JSON3

@testset "12. CLI dump/load composed with admin unload/reload (§4.4)" begin
    with_test_server() do base_url, workdir
        resp = HTTP.post("$base_url/datasets", [], JSON3.write(Dict("id" => "dump_admin_ds", "index_type" => "searchgraph", "distance" => "L2")))
        @test resp.status == 201

        data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")
        docs = [JSON3.read(line) for line in readlines(data_path)[1:10]]
        resp = HTTP.post("$base_url/datasets/dump_admin_ds/append", [], JSON3.write(Dict("items" => docs)))
        @test resp.status == 200

        resp = HTTP.post("$base_url/datasets/dump_admin_ds/delete", [], JSON3.write(Dict("_id" => 6)))
        @test resp.status == 200

        # dump/load need exclusive access to write a fresh target and (for the source) a
        # consistent read -- unload the source first, same technique as chunk 7/9's tests.
        resp = HTTP.post("$base_url/admin/datasets/dump_admin_ds/unload", [], "")
        @test resp.status == 200

        proj = joinpath(@__DIR__, "..", "..")
        cli_script = joinpath(@__DIR__, "..", "..", "src", "apps", "similarity-search.jl")
        bundle_dir = joinpath(workdir, "dump_admin_bundle")

        cmd_dump = `julia --project=$proj $cli_script dump --dataset dump_admin_ds --workdir $workdir --output $bundle_dir`
        @test success(cmd_dump)

        cmd_load = `julia --project=$proj $cli_script load --bundle $bundle_dir --dataset dump_admin_ds_restored --workdir $workdir`
        @test success(cmd_load)

        # The restored dataset already has its own descriptor.json (written by `load`) --
        # a plain admin `reload` picks it up into this same live server with no restart.
        resp = HTTP.post("$base_url/admin/datasets/dump_admin_ds_restored/reload", [], "")
        @test resp.status == 200

        resp = HTTP.get("$base_url/datasets/dump_admin_ds_restored")
        detail = JSON3.read(String(resp.body))
        @test detail.loaded == true
        @test detail.doc_count == 10
        @test detail.tombstone_count == 1
        @test detail.index_kind == "searchgraph"
        @test detail.distance == "L2" # round-tripped from the original dataset's descriptor.json

        # k covers every originally-appended item -- if the tombstone hadn't survived the
        # dump/load/reload round trip, doc_id 6 would show up here too.
        query_vec = docs[1].vector
        resp = HTTP.post("$base_url/datasets/dump_admin_ds_restored/search", [], JSON3.write(Dict("vector" => query_vec, "k" => 10)))
        @test resp.status == 200
        results = JSON3.read(String(resp.body)).results
        @test length(results) == 9
        @test !any(r -> r._id == 6, results)

        # A text dataset's `edit_correction` lives in its descriptor, and the bundle carries it.
        resp = HTTP.post("$base_url/datasets", [], JSON3.write(Dict("id" => "dump_edits_ds",
                         "index_type" => "bm25_invfile", "edit_correction" => true)))
        @test resp.status == 201
        @test HTTP.post("$base_url/datasets/dump_edits_ds/append", [], JSON3.write(Dict("items" => docs))).status == 200
        @test HTTP.post("$base_url/admin/datasets/dump_edits_ds/unload", [], "").status == 200
        edits_bundle = joinpath(workdir, "dump_edits_bundle")
        @test success(`julia --project=$proj $cli_script dump --dataset dump_edits_ds --workdir $workdir --output $edits_bundle`)
        @test success(`julia --project=$proj $cli_script load --bundle $edits_bundle --dataset dump_edits_restored --workdir $workdir`)
        @test HTTP.post("$base_url/admin/datasets/dump_edits_restored/reload", [], "").status == 200
        detail = JSON3.read(String(HTTP.get("$base_url/datasets/dump_edits_restored").body))
        @test detail.loaded == true
        @test detail.edit_correction == true
    end
end
