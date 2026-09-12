using Test
using HTTP
using JSON3

@testset "1. Metric Search Tests" begin
    with_test_server() do base_url, workdir

        data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")

        # 1. Create a Dense Semantic Dataset (Metric Search)
        req_body = JSON3.write(Dict("id" => "metric_ds", "index_type" => "searchgraph", "distance" => "L2"))
        resp = HTTP.post("$base_url/datasets", [], req_body)
        @test resp.status == 201

        # 2. Append Data from JSONL
        # Leemos los primeros 200 párrafos generados
        lines = readlines(data_path)[1:200]
        docs = [JSON3.read(line) for line in lines]
        append_body = JSON3.write(Dict("items" => docs))

        resp = HTTP.post("$base_url/simsearch/metric_ds/append", [], append_body)
        @test resp.status == 200
        @test JSON3.read(String(resp.body)).inserted == 200

        # 3. Query (Metric Search)
        # Usamos el vector del primer documento como query. Debe ser el top-1 con distancia ~0.
        query_vec = docs[1].vector
        search_req = JSON3.write(Dict(
            "vector" => query_vec,
            "k" => 5
        ))

        resp = HTTP.post("$base_url/simsearch/metric_ds/search", [], search_req)
        @test resp.status == 200
        parsed = JSON3.read(String(resp.body))
        @test length(parsed.results) == 5
        @test parsed.results[1].id == docs[1].doc_id
        @test parsed.results[1].distance ≈ 0.0 atol=1e-6

        # 4. Fetch by original id round-trips the appended vector/metadata.
        fetch_req = JSON3.write(Dict("ids" => [docs[1].doc_id]))
        resp = HTTP.post("$base_url/simsearch/metric_ds/fetch", [], fetch_req)
        @test resp.status == 200
        fetched = JSON3.read(String(resp.body)).results
        @test length(fetched) == 1
        @test fetched[1].id == docs[1].doc_id

        # 5. Soft delete: the deleted doc must no longer be returned by search.
        del_req = JSON3.write(Dict("doc_id" => 1))
        resp = HTTP.post("$base_url/simsearch/metric_ds/delete", [], del_req)
        @test resp.status == 200

        resp = HTTP.post("$base_url/simsearch/metric_ds/search", [], search_req)
        @test resp.status == 200
        parsed = JSON3.read(String(resp.body))
        @test !any(r -> r.doc_id == 1, parsed.results)
    end
end
