# The seam between this package and the engine, tested without starting anything: no HTTP
# server, no subprocess, no dataset on disk. These are the functions that translate between
# the wire's vocabulary and the engine's, and they are what an engine-side change breaks
# first -- which is why they run on every change while the end-to-end suites do not.

using SimilaritySearchEngine
import SimilaritySearchEngine as SSE
using SimilaritySearchServer.Server: parse_index_kind, parse_distance, default_textmodel,
                                     engine_error_response, _typed_item
using SimilaritySearchServer: cli_exit_code, wire_kind
using JSON3

@testset "the wire's index kinds map onto engine/backend pairs" begin
    @test parse_index_kind("searchgraph") == (DenseEngine, SearchGraph)
    @test parse_index_kind("exhaustive_search") == (DenseEngine, ExhaustiveSearch)
    @test parse_index_kind("bm25_invfile") == (FullTextEngine, BM25InvertedFile)
    @test parse_index_kind("invfile") == (FullTextEngine, TextInvertedFile)
    @test parse_index_kind("sparse_invfile") == (SparseEngine, InvertedFile)
    # An unknown name is an invalid request, not an opaque failure: the handler turns this
    # very type into a 400.
    @test_throws SSE.UnknownBackend parse_index_kind("nope")

    # Text projects must state a text model; nothing else may carry one.
    @test default_textmodel(FullTextEngine) isa FitFromCorpus
    @test default_textmodel(DenseEngine) === nothing
    @test default_textmodel(SparseEngine) === nothing
end

@testset "engine errors become status codes by category, not by message" begin
    cases = [
        (SSE.ProfileNotInstalled("es", "no profile"), 404),
        (SSE.PendingBacklog(:allknn, 10, 4, "staged backlog"), 409),
        (SSE.NothingStaged("nothing staged"), 409),
        (SSE.PayloadMismatch("wrong item"), 400),
        (SSE.InvalidOption(:dimension, "needs a dimension"), 400),
        (SSE.WrongDimension(8, 4, "wrong dimension"), 400),
        (SSE.CorruptedStorage("unreadable"), 500),
    ]
    for (err, status) in cases
        resp = engine_error_response(err)
        @test resp.status == status
        body = JSON3.read(String(resp.body), Dict{String, Any})
        # The concrete type travels too, for a client that wants to branch without parsing prose
        @test body["kind"] == string(nameof(typeof(err)))
        @test !isempty(body["error"])
    end
end

@testset "engine errors become exit codes by the same categories" begin
    @test cli_exit_code(SSE.ProfileNotInstalled("es", "x")) == 4
    @test cli_exit_code(SSE.PendingBacklog(:fft, 3, 1, "x")) == 5
    @test cli_exit_code(SSE.NoIndex("x")) == 5
    @test cli_exit_code(SSE.PayloadMismatch("x")) == 2
    @test cli_exit_code(SSE.UnsupportedOperation(:calibrate!, "x")) == 2
    @test cli_exit_code(SSE.CorruptedStorage("x")) == 70
end

@testset "a wire item becomes the engine's own item type" begin
    dense = _typed_item(:dense, Dict{String, Any}("vector" => [1.0, 2.0], "doc_id" => "d1",
                                                  "keywords" => ["a"], "year" => 2020))
    @test dense isa DenseItem
    @test dense.doc_id == "d1"
    @test dense.keywords == ["a"]
    # Anything that is not a reserved field is free-form metadata
    @test dense.meta["year"] == 2020

    text = _typed_item(:text, Dict{String, Any}("text" => "hola", "meta" => Dict("lang" => "es")))
    @test text isa TextItem
    @test text.meta["lang"] == "es"
    @test text.doc_id === nothing

    sparse = _typed_item(:sparse, Dict{String, Any}("indices" => [1, 5], "values" => [0.5, 0.25],
                                                    "dimension" => 8))
    @test sparse isa SparseItem

    # An item carrying no payload for this kind of project is skipped, not an error
    @test _typed_item(:dense, Dict{String, Any}("text" => "no vector here")) === nothing
    @test _typed_item(:text, Dict{String, Any}("vector" => [1.0])) === nothing
end
