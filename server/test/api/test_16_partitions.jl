using Test
using HTTP
using JSON3

# Partitioned text datasets over HTTP (PLAN.md §5.9). The engine's own tests check that a
# partition search equals the global search restricted to the partition; this file checks that
# the API carries the field, the partition and the errors through.
@testset "16. Partitioned text datasets" begin
    with_test_server() do base_url, workdir
        data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")
        repo_of(i) = i % 5 == 0 ? ["norte", "sur"] : i % 2 == 0 ? "norte" : i % 31 == 1 ? "isla" : "sur"
        docs = map(enumerate(readlines(data_path)[1:150])) do (i, line)
            d = JSON3.read(line, Dict{String,Any})
            Dict{String,Any}("doc_id" => d["doc_id"], "text" => d["text"], "repo" => repo_of(i))
        end
        in_part(i, p) = (v = repo_of(i); v isa AbstractString ? v == p : p in v)

        @test HTTP.post("$base_url/datasets", [], JSON3.write(Dict(
            "id" => "part_ds", "index_type" => "bm25_invfile", "partition_by" => "repo"))).status == 201
        @test HTTP.post("$base_url/datasets/part_ds/append", [], JSON3.write(Dict("items" => docs))).status == 200

        # The field is reported from the project, and the partitions with their sizes.
        @test JSON3.read(String(HTTP.get("$base_url/datasets/part_ds").body)).partition_by == "repo"
        listing = JSON3.read(String(HTTP.get("$base_url/datasets/part_ds/partitions").body))
        @test listing.partition_by == "repo"
        @test Dict(p.name => p.items for p in listing.partitions) ==
              Dict(p => count(i -> in_part(i, p), 1:150) for p in ("isla", "norte", "sur"))

        # A partition search returns the global results restricted to the partition.
        doc_index = Dict(d["doc_id"] => i for (i, d) in enumerate(docs))
        query = docs[12]["text"]
        glob = JSON3.read(String(HTTP.post("$base_url/datasets/part_ds/ftsearch", [],
                                           JSON3.write(Dict("text" => query, "k" => 150))).body)).results
        for p in ("isla", "norte", "sur")
            want = [r.doc_id for r in glob if in_part(doc_index[r.doc_id], p)]
            got = JSON3.read(String(HTTP.post("$base_url/datasets/part_ds/ftsearch", [],
                                              JSON3.write(Dict("text" => query, "k" => 5, "partition" => p))).body)).results
            @test [r.doc_id for r in got] == want[1:min(5, end)]
        end

        failed(f) = try f(); nothing catch e; e.response end

        # An unknown name is a 404 that names the closest ones, not an empty result.
        resp = failed(() -> HTTP.post("$base_url/datasets/part_ds/ftsearch", [],
                                      JSON3.write(Dict("text" => query, "k" => 5, "partition" => "nortee"))))
        @test resp.status == 404
        body = JSON3.read(String(resp.body))
        @test body.kind == "UnknownPartition"
        @test occursin("\"norte\"", body.error)

        # A partition value that is not a string is refused with the whole batch.
        resp = failed(() -> HTTP.post("$base_url/datasets/part_ds/append", [], JSON3.write(Dict("items" => [
            Dict("doc_id" => "x1", "text" => "uno dos", "repo" => "sur"),
            Dict("doc_id" => "x2", "text" => "tres", "repo" => 7)]))))
        @test resp.status == 400
        @test JSON3.read(String(resp.body)).kind == "PayloadMismatch"

        # The request must name the partition with a string.
        resp = failed(() -> HTTP.post("$base_url/datasets/part_ds/ftsearch", [],
                                      JSON3.write(Dict("text" => query, "partition" => 3))))
        @test resp.status == 400

        # A dataset without a partition field refuses `partition` and the listing.
        @test HTTP.post("$base_url/datasets", [], JSON3.write(Dict("id" => "plain_ds", "index_type" => "bm25_invfile"))).status == 201
        @test HTTP.post("$base_url/datasets/plain_ds/append", [], JSON3.write(Dict("items" => docs[1:20]))).status == 200
        @test JSON3.read(String(HTTP.get("$base_url/datasets/plain_ds").body)).partition_by === nothing
        resp = failed(() -> HTTP.post("$base_url/datasets/plain_ds/ftsearch", [],
                                      JSON3.write(Dict("text" => query, "partition" => "sur"))))
        @test resp.status == 400
        @test JSON3.read(String(resp.body)).kind == "InvalidOption"
        @test failed(() -> HTTP.get("$base_url/datasets/plain_ds/partitions")).status == 400

        # `partition_by` on a dense dataset is refused at creation.
        resp = failed(() -> HTTP.post("$base_url/datasets", [], JSON3.write(Dict(
            "id" => "dense_part", "index_type" => "searchgraph", "partition_by" => "repo"))))
        @test resp.status == 400
    end
end
