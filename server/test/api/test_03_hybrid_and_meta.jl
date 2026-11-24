using Test
using HTTP
using JSON

@testset "3. Hybrid Search and Metadata Filtering Tests" begin
    with_test_server() do base_url, workdir

        data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")
        lines = readlines(data_path)[1:200]
        docs = [JSON.parse(line) for line in lines]
        append_body = JSON.json(Dict("items" => docs))

        # Create Both Indexes. `word_count` is declared+indexed so the metadata filter
        # below can be evaluated against it (§4.5's declared/typed field split).
        meta_schema = [Dict("name" => "word_count", "type" => "int64", "indexed" => true)]
        resp = HTTP.post("$base_url/datasets", [], JSON.json(Dict("id" => "hybrid_dense_ds", "index_type" => "searchgraph", "distance" => "L2", "meta_schema" => meta_schema,
                                                                   "max_k" => 200)))
        @test resp.status == 201
        resp = HTTP.post("$base_url/datasets", [], JSON.json(Dict("id" => "hybrid_lexical_ds", "index_type" => "bm25_invfile")))
        @test resp.status == 201

        resp = HTTP.post("$base_url/datasets/hybrid_dense_ds/append", [], append_body)
        @test resp.status == 200
        resp = HTTP.post("$base_url/datasets/hybrid_lexical_ds/append", [], append_body)
        @test resp.status == 200

        # 1. Hybrid search, dense-only (omit "text" so the request never touches the
        # broken BM25 search path) -- exercises RRF fusion for real.
        query_vec = docs[1].vector
        hybrid_req = JSON.json(Dict(
            "dense_index" => "hybrid_dense_ds",
            "lexical_index" => "hybrid_lexical_ds",
            "vector" => query_vec,
            "k" => 10
        ))

        resp = HTTP.post("$base_url/search/hybrid", [], hybrid_req)
        @test resp.status == 200
        parsed = JSON.parse(String(resp.body))
        @test haskey(parsed, :results)
        @test length(parsed.results) > 0
        @test parsed.results[1].doc_id == docs[1].doc_id

        # 1b. Hybrid search with both modalities.
        hybrid_req_both = JSON.json(Dict(
            "dense_index" => "hybrid_dense_ds",
            "lexical_index" => "hybrid_lexical_ds",
            "vector" => query_vec,
            "text" => "Frankenstein",
            "k" => 10
        ))
        resp_both = HTTP.post("$base_url/search/hybrid", [], hybrid_req_both)
        @test resp_both.status == 200
        parsed_both = JSON.parse(String(resp_both.body))
        @test haskey(parsed_both, :results)
        @test length(parsed_both.results) > 0

        # 2. Metadata Filtering (on dense search): equality/range filter over a declared,
        # indexed field (§5.4 syntax: {"field": {"gte": ..., "lte": ...}}).
        filter_req = JSON.json(Dict(
            "vector" => query_vec,
            "k" => 200,
            "filter" => Dict("word_count" => Dict("gte" => 10))
        ))
        resp = HTTP.post("$base_url/datasets/hybrid_dense_ds/search", [], filter_req)
        @test resp.status == 200
        parsed = JSON.parse(String(resp.body))
        @test length(parsed.results) > 0
        @test all(r -> begin
            fetched = JSON.parse(String(HTTP.post("$base_url/datasets/hybrid_dense_ds/fetch", [], JSON.json(Dict("ids" => [r.doc_id]))).body)).results
            !isempty(fetched) && fetched[1].word_count >= 10
        end, parsed.results)
    end
end
