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

        # 5. What the vocabulary covers, and what it does not (PLAN.md §5.8). The searches
        # above already went through it, so the counters are not empty.
        vocab = JSON3.read(String(HTTP.get("$base_url/datasets/lexical_ds/vocab").body))
        @test vocab.vocsize > 0
        @test vocab.trainsize > 0
        @test length(vocab.top_tokens) > 0
        @test vocab.queries > 0
        @test vocab.query_tokens > 0
        @test 0.0 <= vocab.query_oov_rate <= 1.0
        # The reading over the queries costs nothing and is always there; the one over the
        # documents reads all of them and is asked for explicitly.
        @test !haskey(vocab, :live_oov_rate)

        # A query of words this corpus does not have moves the rate up
        before = vocab.query_oov_tokens
        HTTP.post("$base_url/datasets/lexical_ds/ftsearch", [], JSON3.write(Dict("text" => "criptomoneda blockchain", "k" => 3)))
        vocab = JSON3.read(String(HTTP.get("$base_url/datasets/lexical_ds/vocab?scan=true").body))
        @test vocab.query_oov_tokens > before
        # The documents themselves are covered: this vocabulary was fitted from them
        @test vocab.live_oov_rate == 0.0
        @test vocab.live_tokens > 0

        # A dataset that holds no text has no vocabulary to report on
        resp = try
            HTTP.get("$base_url/datasets/metric_ds/vocab")
        catch e
            e.response
        end
        @test resp.status == 409
        @test JSON3.read(String(resp.body)).kind == "NotTrained"

        # 6. edit_correction: a query token that the vocabulary lacks is corrected to the only
        # vocabulary token at edit distance 1 from it. `Frankenstien` exchanges two letters of
        # `Frankenstein`, which no case or diacritic fold reaches.
        status(f) = try f().status catch e; e.response.status end
        typo = JSON3.write(Dict("text" => "Frankenstien", "k" => 10))
        hits(id) = JSON3.read(HTTP.post("$base_url/datasets/$id/ftsearch", [], typo).body).results

        # Off by default, and reported as off.
        @test isempty(hits("lexical_ds"))
        @test JSON3.read(HTTP.get("$base_url/datasets/lexical_ds").body).edit_correction == false

        resp = HTTP.post("$base_url/datasets", [],
                         JSON3.write(Dict("id" => "lexical_edits", "index_type" => "bm25_invfile",
                                          "edit_correction" => true)))
        @test resp.status == 201
        @test HTTP.post("$base_url/datasets/lexical_edits/append", [], append_body).status == 200
        @test !isempty(hits("lexical_edits"))
        @test JSON3.read(HTTP.get("$base_url/datasets/lexical_edits").body).edit_correction == true

        # The descriptor keeps it, so a plain reload opens the dataset with it again.
        reload(id, body="") = HTTP.post("$base_url/admin/datasets/$id/reload", [], body)
        @test reload("lexical_edits").status == 200
        @test !isempty(hits("lexical_edits"))

        # A reload with a body changes it.
        @test reload("lexical_edits", JSON3.write(Dict("edit_correction" => false))).status == 200
        @test isempty(hits("lexical_edits"))
        @test JSON3.read(HTTP.get("$base_url/datasets/lexical_edits").body).edit_correction == false
        @test reload("lexical_edits", JSON3.write(Dict("edit_correction" => true))).status == 200
        @test !isempty(hits("lexical_edits"))

        # Refused for a dataset without text, at creation and at reload. The refused reload
        # leaves the dataset loaded.
        @test status(() -> HTTP.post("$base_url/datasets", [],
                JSON3.write(Dict("id" => "dense_edits", "index_type" => "searchgraph",
                                 "edit_correction" => true)))) == 400
        @test HTTP.post("$base_url/datasets", [],
                JSON3.write(Dict("id" => "dense_plain", "index_type" => "searchgraph"))).status == 201
        @test status(() -> reload("dense_plain", JSON3.write(Dict("edit_correction" => true)))) == 400
        @test JSON3.read(HTTP.get("$base_url/datasets/dense_plain").body).loaded == true
        # Only a boolean is accepted.
        @test status(() -> HTTP.post("$base_url/datasets", [],
                JSON3.write(Dict("id" => "lexical_bad", "index_type" => "bm25_invfile",
                                 "edit_correction" => "yes")))) == 400
        @test status(() -> reload("lexical_edits", JSON3.write(Dict("edit_correction" => 1)))) == 400
    end
end
