using Test
using HTTP
using JSON3
using CodecZlib

# Responses without cursors (PLAN.md §5.16): a search answers its `k` results in one response,
# `k` is limited by the `max_k` of the dataset, a large response is compressed with gzip when
# the request accepts it, and the listings keep `offset` and `limit`.
@testset "7. Responses: max_k, no cursors, gzip and listings" begin
    with_test_server() do base_url, workdir
        post(path, body) = HTTP.post("$base_url/$path", [], JSON3.write(body); status_exception=false)
        # A copy: `String(v)` takes the bytes of `v` and leaves it empty, and a response is read twice.
        parsed(resp) = JSON3.read(String(copy(resp.body)))
        info(id) = parsed(HTTP.get("$base_url/datasets/$id"))

        vectors = [Float32[i, i % 7, i % 3, 1] for i in 1:150]
        docs = [Dict("doc_id" => "r$i", "vector" => vectors[i]) for i in 1:150]
        q = vectors[1]

        # A dataset created without `max_k` has 100.
        @test post("datasets", Dict("id" => "resp_ds", "index_type" => "exhaustive_search", "distance" => "L2")).status == 201
        @test post("datasets/resp_ds/append", Dict("items" => docs)).status == 200
        @test info("resp_ds").max_k == 100

        r = post("datasets/resp_ds/search", Dict("vector" => q, "k" => 100))
        @test r.status == 200 && length(parsed(r).results) == 100 && !haskey(parsed(r), :cursor_id)
        r = post("datasets/resp_ds/search", Dict("vector" => q, "k" => 101))
        @test r.status == 400 && parsed(r).kind == "InvalidOption"
        @test occursin("max_k = 100", parsed(r).error) && occursin("reload", parsed(r).error)
        for bad in (0, -3, "ten")
            r = post("datasets/resp_ds/search", Dict("vector" => q, "k" => bad))
            @test r.status == 400 && parsed(r).kind == "InvalidOption"
        end

        # `page_size` and the route of the cursors were removed on 2026-11-23.
        r = post("datasets/resp_ds/search", Dict("vector" => q, "k" => 10, "page_size" => 5))
        @test r.status == 400 && parsed(r).kind == "InvalidOption" && occursin("page_size", parsed(r).error)
        @test HTTP.get("$base_url/cursors/abc"; status_exception=false).status == 404
        @test !isdir(joinpath(workdir, "cursors"))

        # `reload` changes `max_k` and saves it in `descriptor.json`.
        @test post("admin/datasets/resp_ds/reload", Dict("max_k" => 150)).status == 200
        @test info("resp_ds").max_k == 150
        @test JSON3.read(read(joinpath(workdir, "datasets", "resp_ds", "descriptor.json"))).max_k == 150
        r = post("datasets/resp_ds/search", Dict("vector" => q, "k" => 150))
        @test r.status == 200 && length(parsed(r).results) == 150
        for bad in (0, "many", true)
            @test post("admin/datasets/resp_ds/reload", Dict("max_k" => bad)).status == 400
        end
        @test info("resp_ds").max_k == 150

        # A dataset created with `max_k`. `ftsearch`, the searches of a follower and the hybrid
        # search are limited too; the hybrid search by the `max_k` of each dataset.
        @test post("datasets", Dict("id" => "resp_bad", "max_k" => -1)).status == 400
        @test post("datasets", Dict("id" => "resp_txt", "index_type" => "bm25_invfile", "max_k" => 3)).status == 201
        texts = [Dict("doc_id" => "t$i", "text" => "gato perro casa número $i", "keywords" => ["k$(i % 2)"]) for i in 1:10]
        @test post("datasets/resp_txt/append", Dict("items" => texts)).status == 200
        @test info("resp_txt").max_k == 3
        @test post("datasets/resp_txt/ftsearch", Dict("text" => "gato", "k" => 3)).status == 200
        r = post("datasets/resp_txt/ftsearch", Dict("text" => "gato", "k" => 4))
        @test r.status == 400 && occursin("max_k = 3", parsed(r).error)
        @test post("datasets/resp_txt/followers", Dict("name" => "kw", "kind" => "keyword", "source" => "keywords")).status == 201
        @test post("datasets/resp_txt/followers/kw/search", Dict("keys" => ["k1"], "k" => 3)).status == 200
        r = post("datasets/resp_txt/followers/kw/search", Dict("keys" => ["k1"], "k" => 4))
        @test r.status == 400 && occursin("max_k = 3", parsed(r).error)
        r = post("datasets/resp_txt/followers/kw/search", Dict("keys" => ["k1"], "k" => 2, "page_size" => 1))
        @test r.status == 400 && occursin("page_size", parsed(r).error)
        @test HTTP.delete("$base_url/datasets/resp_txt/followers/kw").status == 200
        hybrid(k) = post("search/hybrid", Dict("dense_index" => "resp_ds", "lexical_index" => "resp_txt",
                                               "vector" => q, "text" => "gato", "k" => k))
        @test hybrid(3).status == 200
        r = hybrid(4)
        @test r.status == 400 && occursin("resp_txt", parsed(r).error)

        # A `descriptor.json` written before `max_k` existed gives 100.
        dpath = joinpath(workdir, "datasets", "resp_txt", "descriptor.json")
        d = JSON3.read(read(dpath), Dict{String,Any})
        delete!(d, "max_k")
        write(dpath, JSON3.write(d))
        @test post("admin/datasets/resp_txt/reload", Dict()).status == 200
        @test info("resp_txt").max_k == 100
        @test post("datasets/resp_txt/ftsearch", Dict("text" => "gato", "k" => 10)).status == 200

        # Compression. A response of at least `gzip_min_bytes` bytes is compressed when the
        # request accepts gzip, and decompressed it equals the response without compression.
        req_body = JSON3.write(Dict("vector" => q, "k" => 150))
        gz = HTTP.post("$base_url/datasets/resp_ds/search", ["Accept-Encoding" => "gzip"], req_body; decompress=false)
        plain = HTTP.post("$base_url/datasets/resp_ds/search", ["Accept-Encoding" => "identity"], req_body; decompress=false)
        @test HTTP.header(gz, "Content-Encoding") == "gzip" && HTTP.header(gz, "Vary") == "Accept-Encoding"
        @test !HTTP.hasheader(plain, "Content-Encoding")
        @test length(gz.body) < length(plain.body)
        @test transcode(GzipDecompressor, Vector{UInt8}(gz.body)) == Vector{UInt8}(plain.body)
        # `q=0` refuses gzip, and a small body is sent as it is.
        refused = HTTP.post("$base_url/datasets/resp_ds/search", ["Accept-Encoding" => "gzip;q=0"], req_body; decompress=false)
        @test !HTTP.hasheader(refused, "Content-Encoding")
        small = HTTP.get("$(replace(base_url, "/api/v1" => ""))/healthz", ["Accept-Encoding" => "gzip"]; decompress=false)
        @test !HTTP.hasheader(small, "Content-Encoding")
        # HTTP.jl asks for gzip and decompresses it by default, as the Julia client does.
        @test length(parsed(HTTP.post("$base_url/datasets/resp_ds/search", [], req_body)).results) == 150

        # The result of a job is sent compressed when the request accepts gzip, and it equals
        # the file. `max_k` does not limit a job.
        r = post("jobs/allknn", Dict("dataset" => "resp_ds", "k" => 3))
        @test r.status == 202
        job_id = parsed(r).job_id
        status = ""
        for _ in 1:600
            status = string(parsed(HTTP.get("$base_url/jobs/$job_id")).status)
            status in ("completed", "failed") && break
            sleep(0.3)
        end
        @test status == "completed"
        gz = HTTP.get("$base_url/jobs/$job_id/result", ["Accept-Encoding" => "gzip"]; decompress=false)
        plain = HTTP.get("$base_url/jobs/$job_id/result", ["Accept-Encoding" => "identity"]; decompress=false)
        @test HTTP.header(gz, "Content-Encoding") == "gzip" && !HTTP.hasheader(plain, "Content-Encoding")
        @test transcode(GzipDecompressor, Vector{UInt8}(gz.body)) == Vector{UInt8}(plain.body)
        @test !isempty(plain.body)
        @test Vector{UInt8}(HTTP.get("$base_url/jobs/$job_id/result").body) == Vector{UInt8}(plain.body)

        # The listings keep `offset` and `limit`, which need no state in the server.
        @test post("datasets", Dict("id" => "resp_extra_ds")).status == 201
        all_listed = parsed(HTTP.get("$base_url/datasets"))
        total_datasets = all_listed.total
        @test total_datasets >= 3
        first_dataset_page = parsed(HTTP.get("$base_url/datasets?limit=1"))
        @test length(first_dataset_page.datasets) == 1 && first_dataset_page.total == total_datasets
        @test isempty(parsed(HTTP.get("$base_url/datasets?offset=$(total_datasets)&limit=10")).datasets)

        # GET /api/v1/jobs, with status and kind filters.
        @test any(j -> j.id == job_id, parsed(HTTP.get("$base_url/jobs?kind=allknn")).jobs)
        @test !any(j -> j.id == job_id, parsed(HTTP.get("$base_url/jobs?kind=neardup")).jobs)
        @test HTTP.get("$base_url/jobs?status=not_a_real_status"; status_exception=false).status == 400
    end
end
