using Test
using HTTP
using JSON3

# Partitioned text datasets over HTTP (CHANGELOG.md §5.9). The engine's own tests check that a
# partition search equals the global search restricted to the partition; this file checks that
# the API carries the field, the partition and the errors through.
@testset "16. Partitioned text datasets" begin
    with_test_server() do base_url, workdir
        data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")
        repo_of(i) = i % 5 == 0 ? ["norte", "sur"] : i % 2 == 0 ? "norte" : i % 31 == 1 ? "isla" : "sur"
        docs = map(enumerate(readlines(data_path)[1:150])) do (i, line)
            d = JSON3.read(line, Dict{String,Any})
            Dict{String,Any}("doc_id" => d["doc_id"], "text" => d["text"], "repo" => repo_of(i), "n" => i)
        end
        in_part(i, p) = (v = repo_of(i); v isa AbstractString ? v == p : p in v)

        @test HTTP.post("$base_url/datasets", [], JSON3.write(Dict(
            "id" => "part_ds", "index_type" => "bm25_invfile", "partition_by" => "repo", "max_k" => 150))).status == 201
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

        # A filter, alone and inside a partition, keeps the order of the global results. The
        # server reads a bounded number of candidates, so the page can be shorter than `k`, and
        # what it returns is a prefix of the filtered global results.
        for p in (nothing, "sur")
            req = Dict{String,Any}("text" => query, "k" => 4, "filter" => Dict("n" => Dict("gte" => 75)))
            p === nothing || (req["partition"] = p)
            got = [r.doc_id for r in JSON3.read(String(HTTP.post("$base_url/datasets/part_ds/ftsearch", [],
                                                                  JSON3.write(req)).body)).results]
            want = [r.doc_id for r in glob if doc_index[r.doc_id] >= 75 && (p === nothing || in_part(doc_index[r.doc_id], p))]
            @test !isempty(got)
            @test got == want[1:length(got)]
        end
        @test failed(() -> HTTP.post("$base_url/datasets/part_ds/ftsearch", [],
                                     JSON3.write(Dict("text" => ["the", "monster"])))).status == 400

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

        # A dense dataset accepts `partition_by` since CHANGELOG.md §5.10. A sparse one is refused
        # at creation.
        @test HTTP.post("$base_url/datasets", [], JSON3.write(Dict(
            "id" => "dense_part", "index_type" => "searchgraph", "partition_by" => "repo"))).status == 201
        @test JSON3.read(String(HTTP.get("$base_url/datasets/dense_part").body)).partition_by == "repo"
        resp = failed(() -> HTTP.post("$base_url/datasets", [], JSON3.write(Dict(
            "id" => "sparse_part", "index_type" => "sparse_invfile", "dimension" => 8, "partition_by" => "repo"))))
        @test resp.status == 400
    end
end

# Partitioned dense datasets over HTTP (CHANGELOG.md §5.10). The engine's tests check the scan, the
# graphs and the reopen; this file checks that the API carries the partition, the threshold,
# the calibration of one partition and the errors through.
@testset "16b. Partitioned dense datasets" begin
    with_test_server() do base_url, workdir
        data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")
        rows = [JSON3.read(l, Dict{String,Any}) for l in readlines(data_path)[1:300]]
        repo_of(i) = i % 3 == 0 ? "tres" : "otro"
        docs = [Dict{String,Any}("doc_id" => d["doc_id"], "vector" => d["vector"], "repo" => repo_of(i))
                for (i, d) in enumerate(rows)]
        failed(f) = try f(); nothing catch e; e.response end
        post(path, body) = JSON3.read(String(HTTP.post("$base_url/$path", [], JSON3.write(body)).body))

        # "otro" (200 items) reaches the threshold and has a graph; "tres" (100) is scanned.
        @test HTTP.post("$base_url/datasets", [], JSON3.write(Dict(
            "id" => "dpart", "index_type" => "searchgraph", "partition_by" => "repo",
            "partition_graph_min" => 150))).status == 201
        @test HTTP.post("$base_url/datasets/dpart/append", [], JSON3.write(Dict("items" => docs))).status == 200
        detail = JSON3.read(String(HTTP.get("$base_url/datasets/dpart").body))
        @test detail.partition_by == "repo" && detail.partition_graph_min == 150
        listing = JSON3.read(String(HTTP.get("$base_url/datasets/dpart/partitions").body))
        @test listing.partition_graph_min == 150
        @test Dict(p.name => (p.items, p.graph) for p in listing.partitions) == Dict("otro" => (200, true), "tres" => (100, false))

        # The scanned partition returns the exact nearest members, in order of distance.
        q = rows[7]["vector"]
        dist(i) = sum(abs2, Float32.(rows[i]["vector"]) .- Float32.(q))
        tres = [i for i in eachindex(rows) if repo_of(i) == "tres"]
        want = [rows[i]["doc_id"] for i in tres[sortperm(dist.(tres))][1:5]]
        got = post("datasets/dpart/search", Dict("vector" => q, "k" => 5, "partition" => "tres")).results
        @test [r.doc_id for r in got] == want
        otro = Set(rows[i]["doc_id"] for i in eachindex(rows) if repo_of(i) == "otro")
        got = post("datasets/dpart/search", Dict("vector" => q, "k" => 5, "partition" => "otro")).results
        @test length(got) == 5 && all(r -> r.doc_id in otro, got)
        @test got[1].doc_id == rows[7]["doc_id"]

        # Calibrating a partition leaves the global baseline of the descriptor alone.
        baseline = detail.beamsearch_baseline
        cal = post("datasets/dpart/calibrate", Dict("minrecall" => 0.9, "numqueries" => 16, "partition" => "otro"))
        @test cal.status == "calibrated" && cal.partition == "otro"
        @test JSON3.read(String(HTTP.get("$base_url/datasets/dpart").body)).beamsearch_baseline == baseline
        resp = failed(() -> HTTP.post("$base_url/datasets/dpart/calibrate", [], JSON3.write(Dict("partition" => "tres"))))
        @test resp.status == 400 && JSON3.read(String(resp.body)).kind == "InvalidOption"

        # Errors: an unknown name, a name that is not a string, overrides with a partition.
        resp = failed(() -> HTTP.post("$base_url/datasets/dpart/search", [], JSON3.write(Dict("vector" => q, "partition" => "tress"))))
        @test resp.status == 404 && JSON3.read(String(resp.body)).kind == "UnknownPartition"
        @test failed(() -> HTTP.post("$base_url/datasets/dpart/search", [], JSON3.write(Dict("vector" => q, "partition" => 3)))).status == 400
        @test failed(() -> HTTP.post("$base_url/datasets/dpart/search", [], JSON3.write(Dict(
            "vector" => q, "partition" => "otro", "beamsearch_overrides" => Dict("bsize" => 8))))).status == 400

        # `partition_graph_min` needs a dense dataset with `partition_by`, and an integer.
        @test failed(() -> HTTP.post("$base_url/datasets", [], JSON3.write(Dict(
            "id" => "t_min", "index_type" => "bm25_invfile", "partition_by" => "repo", "partition_graph_min" => 10)))).status == 400
        @test failed(() -> HTTP.post("$base_url/datasets", [], JSON3.write(Dict(
            "id" => "d_min", "index_type" => "searchgraph", "partition_graph_min" => 10)))).status == 400
        @test failed(() -> HTTP.post("$base_url/datasets", [], JSON3.write(Dict(
            "id" => "d_min2", "index_type" => "searchgraph", "partition_by" => "repo", "partition_graph_min" => "10")))).status == 400

        # A dense dataset without a partition field refuses `partition`.
        @test HTTP.post("$base_url/datasets", [], JSON3.write(Dict("id" => "dplain", "index_type" => "searchgraph"))).status == 201
        @test HTTP.post("$base_url/datasets/dplain/append", [], JSON3.write(Dict("items" => docs[1:20]))).status == 200
        @test failed(() -> HTTP.post("$base_url/datasets/dplain/search", [], JSON3.write(Dict("vector" => q, "partition" => "otro")))).status == 400
    end
end
