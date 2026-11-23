using Test
using JSON3

# The storage of a dense dataset in `similarity-search` (PLAN.md §5.14, phase 7b), run in this
# process: `build --storage`, `describe`, `searchbatch`, `rebuild`, `dump` and `load`.
@testset "CLI storage of dense datasets" begin
    workdir = mktempdir()
    w = ("--workdir", workdir)
    data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")
    rows = [JSON3.read(l, Dict{String,Any}) for l in readlines(data_path)[1:300]]
    input = tempname() * ".jsonl"
    write(input, join([JSON3.write(Dict("doc_id" => r["doc_id"], "vector" => r["vector"])) for r in rows], "\n"))
    queries = tempname() * ".jsonl"
    picked = (3, 40, 200)
    write(queries, join([JSON3.write(Dict("vector" => rows[i]["vector"])) for i in picked], "\n"))

    cli(args...) = _run_cli(String[string(a) for a in args])
    function search_ids(dataset, k)
        out = tempname()
        code, printed = cli("searchbatch", "--dataset", dataset, "--queries", queries, "--output", out, "--k", k, w...)
        code == 0 || return (code, printed)
        return [collect(JSON3.read(l).ids) for l in readlines(out)]
    end
    function describe(dataset)
        out = tempname()
        code, printed = cli("describe", "--dataset", dataset, "--output", out, w...)
        code == 0 || return (code, printed)
        return JSON3.read(read(out, String))
    end
    build(name, storage) = cli("build", "--dataset", name, "--input", input, "--index-kind", "exhaustive_search",
                               "--storage", storage, w...)

    @test first(build("q", """{"kind": "quantized", "bits": 4, "keep_vectors": true}""")) == 0
    @test first(build("s", """{"kind": "sketched", "hyperplanes": 128, "rerank": 4, "keep_vectors": true}""")) == 0
    @test first(build("n", """{"kind": "sketched", "hyperplanes": 128}""")) == 0
    # A storage that is not JSON, or that the dataset does not take, exits with 2.
    code, out = build("bad", "{kind}")
    @test code == 2 && occursin("--storage is a JSON object", out)
    code, out = build("bad", """{"kind": "quantized", "bits": 5}""")
    @test code == 2 && occursin("8, 4 or 2", out)
    code, out = cli("build", "--dataset", "bad", "--input", input, "--index-kind", "bm25_invfile",
                    "--storage", """{"kind": "quantized"}""", w...)
    @test code == 2 && occursin("applies to a dense dataset", out)
    @test !isdir(joinpath(workdir, "bad"))

    # describe reports the distance of the dataset, its storage and its encoder.
    d = describe("q")
    @test d.distance == "SqL2" && d.storage.kind == "quantized" && d.storage.bits == 4 && d.storage.keep_vectors
    @test d.encoder.fitted && d.encoder.dimension == 128 && d.encoder.bytes_per_vector == 64
    @test d.distance_stats !== nothing && d.distance_stats.sample_size == 100
    d = describe("s")
    @test d.storage.kind == "sketched" && d.storage.rerank == 4 && d.encoder.bytes_on_disk_per_vector == 128 + 512
    @test d.distance_stats !== nothing
    # Without rerank, a sketched dataset answers with distances between sketches: no statistics.
    d = describe("n")
    @test d.encoder.fitted && d.storage.keep_vectors == false && d.distance_stats === nothing

    for name in ("q", "s", "n")
        @test [ids[1] for ids in search_ids(name, 5)] == collect(picked)
    end

    # rebuild encodes the kept vectors again, with a new encoder and the same storage.
    for name in ("q", "s")
        before = copy(describe(name).storage)
        @test first(cli("rebuild", "--dataset", name, w...)) == 0
        d = describe(name)
        @test copy(d.storage) == before && d.doc_count == 300 && d.encoder.fitted
        @test [ids[1] for ids in search_ids(name, 5)] == collect(picked)
    end
    # Without the kept vectors there is nothing to encode: rebuild refuses, and the dataset stays.
    code, out = cli("rebuild", "--dataset", "n", w...)
    @test code == 2 && occursin("keep_vectors=false", out)
    @test describe("n").doc_count == 300

    # dump and load copy the project, with its storage and its encoder.
    bundle = joinpath(workdir, "bundle_n")
    @test first(cli("dump", "--dataset", "n", "--output", bundle, w...)) == 0
    @test first(cli("load", "--bundle", bundle, "--dataset", "n2", w...)) == 0
    @test copy(describe("n2").storage) == copy(describe("n").storage)
    @test copy(describe("n2").encoder) == copy(describe("n").encoder)
    @test search_ids("n2", 5) == search_ids("n", 5)
end
