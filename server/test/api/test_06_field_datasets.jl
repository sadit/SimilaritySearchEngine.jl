using Test
using HTTP
using JSON3

# Join groups were removed on 2026-11-10 (PLAN.md §0, item 11). A record with several text
# fields is one dataset per field, each appended with the record's doc_id. These tests check
# that pattern, and that the removed fields and routes are refused instead of ignored.
@testset "6. One text dataset per field, without join groups" begin
    with_test_server() do base_url, workdir
        data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")
        docs = [JSON3.read(line) for line in readlines(data_path)[1:20]]
        post(path, body) = try
            HTTP.post("$base_url$path", [], JSON3.write(body))
        catch e
            e isa HTTP.StatusError || rethrow()
            e.response
        end

        # 1. The three removed fields are refused at creation, each one alone.
        for (field, value) in (("join_group", "books"), ("key", "title"), ("holds_metadata", true))
            resp = post("/datasets", Dict("id" => "fd_bad", "index_type" => "bm25_invfile", field => value))
            @test resp.status == 400
            @test occursin("join groups were removed", JSON3.read(String(resp.body)).error)
        end
        @test HTTP.get("$base_url/datasets/fd_bad"; status_exception=false).status == 404

        # 2. Two field datasets, appended with the same doc_ids.
        for id in ("fd_title", "fd_body")
            @test post("/datasets", Dict("id" => id, "index_type" => "bm25_invfile")).status == 201
        end
        title = [Dict("doc_id" => d.doc_id, "text" => first(d.text, 60), "meta" => Dict("n" => i))
                 for (i, d) in enumerate(docs)]
        body = [Dict("doc_id" => d.doc_id, "text" => d.text) for d in docs]
        @test post("/datasets/fd_title/append", Dict("items" => title)).status == 200
        @test post("/datasets/fd_body/append", Dict("items" => body)).status == 200

        # 3. Each field is searched with its own request, and the results carry the doc_id.
        results = Dict{String, Any}()
        for (name, id) in ("title" => "fd_title", "body" => "fd_body")
            resp = post("/datasets/$id/ftsearch", Dict("text" => docs[3].text, "k" => 3))
            @test resp.status == 200
            results[name] = JSON3.read(String(resp.body)).results
        end
        @test results["body"][1].doc_id == docs[3].doc_id
        @test any(r -> r.doc_id == docs[3].doc_id, results["title"])

        # 4. The metadata of a result of one field comes from a fetch to another field's
        # dataset, by doc_id.
        resp = post("/datasets/fd_title/fetch", Dict("ids" => [r.doc_id for r in results["body"]]))
        @test resp.status == 200
        fetched = JSON3.read(String(resp.body)).results
        @test [f.doc_id for f in fetched] == [r.doc_id for r in results["body"]]
        @test fetched[1].n == 3

        # 5. The two routes of the join groups no longer exist.
        @test post("/search/group", Dict("join_group" => "books", "key" => "*", "text" => "x")).status in (404, 405)
        @test HTTP.get("$base_url/datasets/fd_title/join_group"; status_exception=false).status in (404, 405)

        # 6. A descriptor written before the removal loses the three fields when it is read.
        dpath = joinpath(workdir, "datasets", "fd_title", "descriptor.json")
        desc = JSON3.read(read(dpath), Dict{String, Any})
        merge!(desc, Dict("join_group" => "books", "holds_metadata" => false, "key" => "title"))
        write(dpath, JSON3.write(desc))
        detail = JSON3.read(String(HTTP.get("$base_url/datasets/fd_title").body), Dict{String, Any})
        @test !any(f -> haskey(detail, f), ("join_group", "holds_metadata", "key"))
        @test detail["index_kind"] == "bm25_invfile"
    end
end
