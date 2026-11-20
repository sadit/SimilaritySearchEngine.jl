using Test
using SimilaritySearchClient
using SimilaritySearchClient: response_error, wire_item, wire_meta_schema, ERROR_KINDS
using SparseArrays
using JSON3

# These tests need no server. The tests against a running server are in the server's suite
# (server/test/api/test_17_julia_client.jl), which can start one; this package cannot depend on
# the server, not even for its tests, because its users install it without the server.

body(x) = Vector{UInt8}(JSON3.write(x))

@testset "SimilaritySearchClient.jl" begin
    @testset "an error response becomes the type its kind names" begin
        e = response_error(404, body(Dict("error" => "no 'nortee'; closest: norte", "kind" => "UnknownPartition")))
        @test e isa UnknownPartition && e isa NotFound && e isa EngineError
        @test e.status == 404 && e.msg == "no 'nortee'; closest: norte"
        @test response_error(409, body(Dict("error" => "x", "kind" => "NotTrained"))) isa NotTrained
        @test response_error(400, body(Dict("error" => "x", "kind" => "WrongDimension"))) isa WrongDimension
        @test response_error(500, body(Dict("error" => "x", "kind" => "CorruptedStorage"))) isa StorageFailure
        for (kind, T) in ERROR_KINDS
            @test response_error(400, body(Dict("error" => "x", "kind" => kind))) isa T
        end
    end

    @testset "without a known kind, the status selects the category" begin
        @test response_error(400, body(Dict("error" => "bad"))) isa RequestRejected
        @test response_error(400, body(Dict("error" => "bad", "kind" => "InvalidRequest"))) isa RequestRejected
        @test response_error(401, body(Dict("error" => "a valid token is required"))) isa Unauthenticated
        @test response_error(403, body(Dict("error" => "no"))) isa Forbidden
        @test response_error(404, body(Dict("error" => "dataset_not_found"))) isa ResourceNotFound
        @test response_error(409, body(Dict("error" => "busy"))) isa StateConflict
        @test response_error(500, body(Dict("error" => "boom"))) isa ServerFailure
        e = response_error(502, Vector{UInt8}("<html>Bad Gateway</html>"))
        @test e isa UnexpectedResponse && e.msg == "<html>Bad Gateway</html>"
        @test !(e isa EngineError) && e isa RemoteError
        @test sprint(showerror, response_error(404, body(Dict("error" => "dataset_not_found")))) ==
              "ResourceNotFound (HTTP 404): dataset_not_found"
    end

    @testset "items are sent in the shape of the append route" begin
        d = wire_item(DenseItem([1, 2, 3]; doc_id=7, keywords=[:a], meta=Dict(:year => 2020)))
        @test d["vector"] == Float32[1, 2, 3] && d["doc_id"] == "7" && d["keywords"] == ["a"]
        @test d["meta"] == Dict("year" => 2020) && d["refs"] == String[]
        @test !haskey(wire_item(TextItem("hola")), "doc_id")
        @test wire_item(TextItem("hola"))["text"] == "hola"
        s = wire_item(SparseItem(sparsevec([2, 5], [1.0, 0.5], 9)))
        @test s["indices"] == Int32[2, 5] && s["values"] == Float32[1.0, 0.5] && s["dimension"] == 9
    end

    @testset "options are checked before a request is sent" begin
        @test wire_meta_schema(["year" => :int64, ("lang", :string)]) ==
              [Dict("name" => "year", "type" => "int64"), Dict("name" => "lang", "type" => "string")]
        @test_throws ArgumentError wire_meta_schema(["year"])
        # Nothing listens on this port: each call must fail before it connects.
        h = RemoteProject(Connection("http://127.0.0.1:1"), "x")
        @test_throws ArgumentError search(h, [1.0, 2.0]; filter=(r, m) -> true)
        @test_throws ArgumentError ftsearch(h, "x"; filter=(r, m) -> true)
        @test_throws ArgumentError exists(h, ["a,b"])
        @test_throws ArgumentError hybrid_search(h, RemoteProject(Connection("http://127.0.0.1:2"), "y"); text="x")
    end

    @testset "a failed job names the category of its exit code" begin
        exit_category = SimilaritySearchClient.exit_category
        @test exit_category(2) === InvalidRequest && exit_category(4) === NotFound
        @test exit_category(5) === ConflictingState && exit_category(70) === StorageFailure
        @test exit_category(1) === nothing && exit_category(nothing) === nothing
        e = JobFailed("j1", "fft", 2, InvalidRequest, "job process exited with status 2")
        @test e isa RemoteError
        @test sprint(showerror, e) == "JobFailed: fft job j1: job process exited with status 2 (InvalidRequest)"
        @test occursin("still on the server", sprint(showerror, JobTimeout("j2", 5.0)))
        h = RemoteProject(Connection("http://127.0.0.1:1"), "x")
        @test_throws ArgumentError search_pages(h, [1.0], 10; page_size=0)
        @test_throws ArgumentError search_pages(h, [1.0], 10; page_size=5, filter=(r, m) -> true)
        @test FFTResult === CenterSelectionResult
    end

    @testset "a connection holds the url and the token" begin
        c = Connection("http://host:8080/"; token="abc")
        @test c.url == "http://host:8080"
        @test ("Authorization" => "Bearer abc") in c.headers
        @test !any(p -> first(p) == "Authorization", Connection().headers)
        @test close_project!(RemoteProject(c, "x")) === nothing
        @test index!(RemoteProject(c, "x")) === nothing
    end
end
