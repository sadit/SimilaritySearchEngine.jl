using Test
using HTTP
using JSON3

# RESOLVED (was a KNOWN UPSTREAM BUG): TextSearch.jl 1.0.0 now defines its own local
# `pairiterator` in `TextSearch.jl/src/bm25/BM25.jl` instead of importing it from
# `SimilaritySearch.InvertedFiles` (which removed that name in `7cce0cd`) -- the cross-repo
# race documented in PLAN.md §5.3's former "Open issue" note is fixed now that both packages
# landed stable releases (SimilaritySearch v1.1.1, TextSearch v1.0.0). Verified via direct
# CLI reproduction (`build --index-kind bm25_invfile` + `searchbatch`) before flipping these
# back to real `@test`s -- see PLAN.md §5.3's updated note.
#
# `invfile`/`weighted_invfile` (§4 below) was a separate, application-side issue (a raw
# bag-of-words no longer scores under `NormCosine` since TextSearch.jl dropped its
# `Dict`-based `dot`/`norm`, `191103c`), already fixed in `IndexEngine` (vectorize through a
# trained `VectorModel` instead of `bagofwords` for this index kind).
@testset "2. Full Text Search Tests" begin
    with_test_server() do base_url, workdir

        data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")

        # 1. Create a Lexical Dataset (BM25)
        req_body = JSON3.write(Dict("id" => "lexical_ds", "index_type" => "bm25_invfile"))
        resp = HTTP.post("$base_url/datasets", [], req_body)
        @test resp.status == 201

        # 2. Append Data
        lines = readlines(data_path)[1:200]
        docs = [JSON3.read(line) for line in lines]

        append_body = JSON3.write(Dict("items" => docs))
        resp = HTTP.post("$base_url/datasets/lexical_ds/append", [], append_body)
        @test resp.status == 200

        # 3. Full Text Search
        ftsearch_req = JSON3.write(Dict(
            "text" => "Frankenstein",
            "k" => 10
        ))
        resp = HTTP.post("$base_url/datasets/lexical_ds/ftsearch", [], ftsearch_req)
        @test resp.status == 200
        results = JSON3.read(resp.body).results
        @test !isempty(results)

        # 4. The other text index kind (plain weighted `InvertedFile`, NormCosine-scored).
        req_body2 = JSON3.write(Dict("id" => "lexical_ds_inv", "index_type" => "invfile"))
        resp2 = HTTP.post("$base_url/datasets", [], req_body2)
        @test resp2.status == 201

        resp2 = HTTP.post("$base_url/datasets/lexical_ds_inv/append", [], append_body)
        @test resp2.status == 200

        ftsearch_req2 = JSON3.write(Dict(
            "text" => "Frankenstein",
            "k" => 10
        ))
        resp3 = HTTP.post("$base_url/datasets/lexical_ds_inv/ftsearch", [], ftsearch_req2)
        @test resp3.status == 200
        results = JSON3.read(resp3.body).results
        @test !isempty(results)
    end
end
