using Test
using HTTP
using JSON3

@testset "9. CLI describe/rebuild against an HTTP-created dataset with real soft-deletes" begin
    with_test_server() do base_url, workdir
        resp = HTTP.post("$base_url/datasets", [], JSON3.write(Dict("id" => "rebuild_http_ds", "index_type" => "searchgraph", "distance" => "L2")))
        @test resp.status == 201

        data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")
        docs = [JSON3.read(line) for line in readlines(data_path)[1:20]]
        resp = HTTP.post("$base_url/simsearch/rebuild_http_ds/append", [], JSON3.write(Dict("items" => docs)))
        @test resp.status == 200
        @test JSON3.read(String(resp.body)).inserted == 20

        # Soft-delete 3 of the 20 items over HTTP -- this is the only way to create a
        # tombstone at all (there's no CLI delete command). handle_delete_item now resaves
        # the snapshot precisely so a separate CLI process (describe/rebuild below) can see it.
        for doc_id in (2, 5, 11)
            resp = HTTP.post("$base_url/simsearch/rebuild_http_ds/delete", [], JSON3.write(Dict("doc_id" => doc_id)))
            @test resp.status == 200
        end

        proj = joinpath(@__DIR__, "..", "..")
        cli_script = joinpath(@__DIR__, "..", "..", "src", "apps", "similarity-search.jl")

        # `describe` opens the dataset read-only (Project.open_project(...; read_only=true))
        # precisely so it can inspect a dataset the live server still has open for writing,
        # without hitting RocksDB's exclusive write lock -- exercise exactly that: the
        # server hasn't been touched yet, "rebuild_http_ds" is still loaded in it. Sees
        # exactly the 3 tombstones this test just made durable via HTTP delete.
        desc_out = tempname()
        cmd_describe = `julia --project=$proj $cli_script describe --dataset rebuild_http_ds --workdir $workdir --output $desc_out`
        @test success(cmd_describe)
        desc = JSON3.read(read(desc_out, String))
        @test desc.doc_count == 20
        @test desc.tombstone_count == 3
        @test desc.tombstone_ratio ≈ 0.15

        # `rebuild` writes, and a write genuinely cannot run concurrently with a server that
        # still has the same dataset open (RocksDB's exclusive write lock) -- PLAN.md's own
        # admin note already establishes that live-dataset write access goes through the
        # server's HTTP API, not a second process touching the files directly. Since nothing
        # else in this suite touches "rebuild_http_ds" again, unload it from the live server
        # first to simulate the "server not currently serving this dataset" case `rebuild`
        # actually requires.
        srv = ensure_test_server()
        Project.close_project(srv.app.handles["rebuild_http_ds"].project)
        delete!(srv.app.handles, "rebuild_http_ds")

        cmd_rebuild = `julia --project=$proj $cli_script rebuild --dataset rebuild_http_ds --workdir $workdir`
        @test success(cmd_rebuild)

        desc_out2 = tempname()
        cmd_describe2 = `julia --project=$proj $cli_script describe --dataset rebuild_http_ds --workdir $workdir --output $desc_out2`
        @test success(cmd_describe2)
        desc2 = JSON3.read(read(desc_out2, String))
        @test desc2.doc_count == 17
        @test desc2.tombstone_count == 0

        # The rebuilt index is independently searchable (fresh CLI process, rebuilt on-disk
        # snapshot only) with a plausible result count against the now-smaller live set.
        out_file = tempname()
        cmd_search = `julia --project=$proj $cli_script searchbatch --dataset rebuild_http_ds --queries $data_path --output $out_file --k 5 --workdir $workdir`
        @test success(cmd_search)
        lines = readlines(out_file)
        @test length(lines) > 0
        parsed = JSON3.read(lines[1])
        @test 0 < length(parsed.ids) <= 5
    end
end
