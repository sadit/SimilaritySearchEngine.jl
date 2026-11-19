using Test
using JSON3

# `build --partition-by`, `searchbatch --partition`, and a `rebuild` that keeps the partition
# field (PLAN.md §5.9).
@testset "CLI partitions" begin
    workdir = mktempdir()
    data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")
    cli_script = joinpath(@__DIR__, "..", "..", "src", "apps", "similarity-search.jl")
    proj = joinpath(@__DIR__, "..", "..")

    input = tempname() * ".jsonl"
    open(input, "w") do io
        for (i, line) in enumerate(readlines(data_path)[1:200])
            d = JSON3.read(line, Dict{String,Any})
            println(io, JSON3.write(Dict("doc_id" => d["doc_id"], "text" => d["text"],
                                         "repo" => i % 3 == 0 ? "tres" : "otro")))
        end
    end
    queries = tempname() * ".jsonl"
    write(queries, join([JSON3.write(Dict("text" => JSON3.read(l)["text"])) for l in readlines(data_path)[1:5]], "\n"))

    @test success(`julia --project=$proj $cli_script build --dataset part_cli --input $input --index-kind bm25_invfile --partition-by repo --workdir $workdir`)

    function partition_ids()
        out = tempname()
        @test success(`julia --project=$proj $cli_script searchbatch --dataset part_cli --queries $queries --output $out --k 5 --partition tres --workdir $workdir`)
        [collect(JSON3.read(l).ids) for l in readlines(out)]
    end
    before = partition_ids()
    @test all(ids -> !isempty(ids) && all(id -> id % 3 == 0, ids), before)

    # An unknown partition is an engine error, and the command fails.
    @test !success(`julia --project=$proj $cli_script searchbatch --dataset part_cli --queries $queries --output $(tempname()) --partition cuatro --workdir $workdir`)

    # Nothing was deleted, so the rebuilt project has the same ids, and it is still partitioned.
    @test success(`julia --project=$proj $cli_script rebuild --dataset part_cli --workdir $workdir`)
    after = partition_ids()
    @test all(ids -> all(id -> id % 3 == 0, ids), after)
    @test [length(x) for x in after] == [length(x) for x in before]
end

# A dense dataset: `build --partition-by --partition-graph-min`, `searchbatch --partition`, and
# a `rebuild` that keeps both (PLAN.md §5.10).
@testset "CLI dense partitions" begin
    workdir = mktempdir()
    data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")
    cli_script = joinpath(@__DIR__, "..", "..", "src", "apps", "similarity-search.jl")
    proj = joinpath(@__DIR__, "..", "..")
    rows = [JSON3.read(l, Dict{String,Any}) for l in readlines(data_path)[1:300]]

    input = tempname() * ".jsonl"
    open(input, "w") do io
        for (i, d) in enumerate(rows)
            println(io, JSON3.write(Dict("doc_id" => d["doc_id"], "vector" => d["vector"],
                                         "repo" => i % 3 == 0 ? "tres" : "otro")))
        end
    end
    queries = tempname() * ".jsonl"
    write(queries, join([JSON3.write(Dict("vector" => d["vector"])) for d in rows[1:5]], "\n"))

    # "otro" (200 items) reaches the threshold and has a graph; "tres" (100) is scanned.
    @test success(`julia --project=$proj $cli_script build --dataset dense_cli --input $input --index-kind searchgraph --partition-by repo --partition-graph-min 150 --workdir $workdir`)
    function partition_ids(p)
        out = tempname()
        @test success(`julia --project=$proj $cli_script searchbatch --dataset dense_cli --queries $queries --output $out --k 5 --partition $p --workdir $workdir`)
        [collect(JSON3.read(l).ids) for l in readlines(out)]
    end
    for (p, inp) in (("tres", id -> id % 3 == 0), ("otro", id -> id % 3 != 0))
        ids = partition_ids(p)
        @test all(x -> length(x) == 5 && all(inp, x), ids)
    end
    @test !success(`julia --project=$proj $cli_script searchbatch --dataset dense_cli --queries $queries --output $(tempname()) --partition cuatro --workdir $workdir`)

    @test success(`julia --project=$proj $cli_script rebuild --dataset dense_cli --workdir $workdir`)
    @test all(x -> all(id -> id % 3 == 0, x), partition_ids("tres"))
end
