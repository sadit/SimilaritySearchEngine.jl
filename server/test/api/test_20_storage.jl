using Test
using HTTP
using JSON3
import SimilaritySearchClient as C

# The storage of a dense dataset over HTTP (PLAN.md §5.14, phase 7b): `storage` of
# `POST /api/v1/datasets`, `storage` and `encoder` of `GET /api/v1/datasets/{id}`, and the
# `storage` keyword of the Julia client.
@testset "20. Storage of dense datasets" begin
    with_test_server() do base_url, workdir
        post(path, body) = HTTP.post("$base_url/$path", [], JSON3.write(body))
        get_(path) = JSON3.read(String(HTTP.get("$base_url/$path").body))
        failed(f) = try f(); nothing catch e; e.response end
        # `String(body)` empties the vector, and a response is read twice here.
        kind_of(resp) = JSON3.read(String(copy(resp.body))).kind
        search(id, i, k) = JSON3.read(String(post("datasets/$id/search", Dict("vector" => rows[i]["vector"], "k" => k)).body)).results
        data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")
        rows = [JSON3.read(l, Dict{String,Any}) for l in readlines(data_path)[1:300]]
        items(r) = [Dict("doc_id" => rows[i]["doc_id"], "vector" => rows[i]["vector"]) for i in r]

        # A quantized dataset: the encoder is fitted at the first append, which indexes.
        @test post("datasets", Dict("id" => "st_q", "index_type" => "exhaustive_search", "distance" => "L2",
                                    "storage" => Dict("kind" => "quantized", "bits" => 8))).status == 201
        @test post("datasets/st_q/append", Dict("items" => items(1:200))).status == 200
        d = get_("datasets/st_q")
        @test d.storage.kind == "quantized" && d.storage.bits == 8 && d.storage.range == "global" &&
              d.storage.rotation == "none" && d.storage.keep_vectors == false && d.storage.samplesize == 4096
        @test d.encoder.fitted && d.encoder.dimension == 128 && d.encoder.bytes_per_vector == 128 && d.encoder.staged == 0
        @test d.encoder.min < d.encoder.max
        @test search("st_q", 7, 5)[1]._id == 7

        # A sketched dataset keeps its first 64 vectors staged, and answers NotTrained (409).
        sk = Dict("kind" => "sketched", "hyperplanes" => 128, "bits" => 8, "rerank" => 4, "keep_vectors" => true)
        @test post("datasets", Dict("id" => "st_s", "index_type" => "exhaustive_search", "distance" => "L2",
                                    "storage" => sk)).status == 201
        @test post("datasets/st_s/append", Dict("items" => items(1:50))).status == 200
        d = get_("datasets/st_s")
        @test d.encoder.fitted == false && d.encoder.staged == 50 && d.encoder.dimension === nothing
        resp = failed(() -> post("datasets/st_s/search", Dict("vector" => rows[1]["vector"], "k" => 5)))
        @test resp.status == 409 && kind_of(resp) == "NotTrained"
        @test post("datasets/st_s/append", Dict("items" => items(51:300))).status == 200
        d = get_("datasets/st_s")
        @test d.storage.hyperplanes == 128 && d.storage.rerank == 4 && d.storage.hsel == 16 * 128 &&
              d.storage.henc == 1024 && d.storage.entquantile == 0.5 && d.storage.anchorpolicy == "random"
        @test d.encoder.fitted && d.encoder.staged == 0 && d.encoder.bytes_per_vector == 128 &&
              d.encoder.bytes_on_disk_per_vector == 128 + 4 * 128
        # With rerank, the results have the distance of the dataset.
        res = search("st_s", 9, 5)
        @test res[1]._id == 9 && res[1].distance == 0 && issorted([r.distance for r in res])

        # The storage lives in the project: a reload reads it back.
        before = copy(get_("datasets/st_s").storage)
        @test post("admin/datasets/st_s/unload", Dict()).status == 200
        @test get_("datasets/st_s").storage === nothing
        @test post("admin/datasets/st_s/reload", Dict()).status == 200
        @test copy(get_("datasets/st_s").storage) == before
        @test search("st_s", 9, 5)[1]._id == 9

        # A dataset of Float32 vectors reports its storage, and no encoder.
        @test post("datasets", Dict("id" => "st_f", "index_type" => "exhaustive_search")).status == 201
        d = get_("datasets/st_f")
        @test d.storage.kind == "float32" && d.encoder === nothing

        # Refused with 400, and no dataset is created.
        for (id, body) in (("st_bad1", Dict("index_type" => "exhaustive_search", "storage" => Dict("kind" => "pq"))),
                           ("st_bad2", Dict("index_type" => "exhaustive_search", "storage" => Dict("kind" => "quantized", "bits" => 3))),
                           ("st_bad3", Dict("index_type" => "exhaustive_search", "storage" => Dict("kind" => "sketched", "rerank" => 2))),
                           ("st_bad4", Dict("index_type" => "bm25_invfile", "storage" => Dict("kind" => "quantized"))),
                           ("st_bad5", Dict("index_type" => "exhaustive_search", "distance" => "angle",
                                            "storage" => Dict("kind" => "quantized"))),
                           ("st_bad6", Dict("index_type" => "exhaustive_search", "distance" => "jaccard")))
            resp = failed(() -> post("datasets", merge(Dict{String,Any}("id" => id), body)))
            @test resp.status == 400 && kind_of(resp) == "InvalidOption"
            @test !isdir(joinpath(workdir, "datasets", id))
        end

        # The Julia client sends `storage` as it is, and reads it back in `project_info`.
        conn = C.Connection(ensure_test_server().root_url)
        h = C.create_project(conn, "st_jc"; index_type="exhaustive_search", distance="L2",
                             storage=(kind="quantized", bits=4, keep_vectors=true))
        @test C.append_items!(h, [C.DenseItem(Float32.(rows[i]["vector"]); doc_id=rows[i]["doc_id"]) for i in 1:100]) == 100
        info = C.project_info(h)
        @test info["storage"]["kind"] == "quantized" && info["storage"]["bits"] == 4 && info["storage"]["keep_vectors"]
        @test info["encoder"]["fitted"] && info["encoder"]["bytes_per_vector"] == 64
        @test first(C.search(h, Float32.(rows[12]["vector"]), 3))._id == 12
    end
end
