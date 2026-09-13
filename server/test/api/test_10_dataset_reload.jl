using Test
using HTTP
using JSON3

@testset "10. Server-restart dataset reload (Server.reload_datasets!)" begin
    with_test_server() do base_url, workdir
        # --- Dataset 1: dense, schema-declared indexed field, some appended items, one soft-delete ---

        resp = HTTP.post("$base_url/datasets", [], JSON3.write(Dict(
            "id" => "reload_ds1", "index_type" => "searchgraph", "distance" => "L2",
            "meta_schema" => [Dict("name" => "category", "type" => "string", "indexed" => true)],
        )))
        @test resp.status == 201

        data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")
        docs = [JSON3.read(line) for line in readlines(data_path)[1:10]]
        resp = HTTP.post("$base_url/simsearch/reload_ds1/append", [], JSON3.write(Dict("items" => docs)))
        @test resp.status == 200
        @test JSON3.read(String(resp.body)).inserted == 10

        resp = HTTP.post("$base_url/simsearch/reload_ds1/delete", [], JSON3.write(Dict("doc_id" => 3)))
        @test resp.status == 200

        # --- Dataset 2: text (bm25_invfile), created but never appended to -- no snapshot exists yet ---

        resp = HTTP.post("$base_url/datasets", [], JSON3.write(Dict("id" => "reload_ds2", "index_type" => "bm25_invfile")))
        @test resp.status == 201

        # --- Simulate "server just restarted": unload both from the live in-memory app ---
        # (mirrors test_09's technique -- RocksDB's exclusive write lock means a *second*
        # AppState pointed at the same workdir can't safely coexist with this one, so this
        # test exercises the real `reload_datasets!` function against the same live `app`
        # after clearing exactly the two entries it should repopulate, rather than spinning
        # up a genuinely second server process.)

        srv = ensure_test_server()
        for id in ("reload_ds1", "reload_ds2")
            Project.close_project(srv.app.handles[id].project)
            delete!(srv.app.handles, id)
        end
        @test !haskey(srv.app.handles, "reload_ds1")
        @test !haskey(srv.app.handles, "reload_ds2")

        reloaded = Server.reload_datasets!(srv.app)
        @test "reload_ds1" in reloaded
        @test "reload_ds2" in reloaded

        # --- reload_ds1: schema, doc_count, and the tombstone all round-tripped from disk ---

        @test haskey(srv.app.handles, "reload_ds1")
        engine1 = srv.app.handles["reload_ds1"].engine
        @test length(engine1.backend.index) == 10
        @test length(engine1.deleted_ids) == 1
        @test 3 in engine1.deleted_ids

        # --- reload_ds2: never indexed -- reopens as a freshly-created, untrained text engine ---

        engine2 = srv.app.handles["reload_ds2"].engine
        @test engine2.backend.index === nothing
        @test payload_kind(engine2) === :text

        # --- Both are genuinely live again: exercise them over HTTP exactly like a normal request ---

        # k=10 covers every one of the 10 originally-appended items -- if the tombstone
        # hadn't survived the reload, doc_id 3 would show up here too.
        query_vec = docs[1].vector
        resp = HTTP.post("$base_url/simsearch/reload_ds1/search", [], JSON3.write(Dict("vector" => query_vec, "k" => 10)))
        @test resp.status == 200
        results = JSON3.read(String(resp.body)).results
        @test length(results) == 9
        @test !any(r -> r.doc_id == 3, results)

        resp = HTTP.post("$base_url/simsearch/reload_ds2/append", [], JSON3.write(Dict("items" => docs[1:3])))
        @test resp.status == 200
        @test JSON3.read(String(resp.body)).inserted == 3

        resp = HTTP.post("$base_url/simsearch/reload_ds2/ftsearch", [], JSON3.write(Dict("text" => "letter", "k" => 3)))
        @test resp.status == 200

        # Descriptor-level view is consistent post-reload too (doc_count/tombstone now come
        # from the freshly reloaded live engine, not a stale pre-restart snapshot of stats).
        resp = HTTP.get("$base_url/datasets/reload_ds1")
        detail = JSON3.read(String(resp.body))
        @test detail.doc_count == 10
        @test detail.tombstone_count == 1
        @test detail.loaded == true
    end
end
