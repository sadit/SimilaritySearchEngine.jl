using Test
using SimilaritySearchEngine
using JSON
using RocksDB

const FRANKENSTEIN_PATH = joinpath(@__DIR__, "data", "frankenstein.jsonl")

function mktempworkdir(f)
    dir = mktempdir()
    try
        f(dir)
    finally
        rm(dir; recursive=true, force=true)
    end
end

@testset "SimilaritySearchEngine.jl" begin

    @testset "dense dataset: create, append, search, calibrate, allknn, delete, fetch" begin
        mktempworkdir() do workdir
            items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:100]]

            h = create_dataset(workdir, "dense_ds")
            inserted = append_items!(h, items)
            @test inserted == 100

            res = search(h, items[1]["vector"]; k=5)
            @test length(res) == 5
            @test res[1].id == "frankenstein_1"
            @test res[1].doc_id == 1
            @test res[1].distance ≈ 0.0 atol=1e-6

            bs = calibrate!(h; numqueries=20)
            @test bs.bsize isa Integer

            ak = allknn(h; k=5)
            @test length(ak) == 100
            @test ak[1].id == 1
            @test ak[1].neighbors[1] == 1
            @test ak[1].dists[1] ≈ 0.0 atol=1e-6

            e_before = exists(h, ["frankenstein_2", "does-not-exist"])
            @test e_before[1].exists && !e_before[1].deleted
            @test !e_before[2].exists

            delete_item!(h, 2)
            e_after = exists(h, ["frankenstein_2"])
            @test e_after[1].exists && e_after[1].deleted

            res_after_delete = search(h, items[1]["vector"]; k=100)
            @test !(2 in [r.doc_id for r in res_after_delete])

            fetched = fetch_items(h, ["frankenstein_3"])
            @test length(fetched) == 1
            @test fetched[1]["id"] == "frankenstein_3"

            filtered = search(h, items[1]["vector"]; k=5, filter=Dict("id" => "frankenstein_1"))
            @test length(filtered) == 1
            @test filtered[1].id == "frankenstein_1"

            close_dataset!(h)
        end
    end

    @testset "text (bm25) dataset: ftsearch" begin
        mktempworkdir() do workdir
            items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:50]]

            h = create_dataset(workdir, "text_ds"; index_type="bm25_invfile")
            inserted = append_items!(h, items)
            @test inserted == 50

            res = ftsearch(h, items[1]["text"]; k=3)
            @test length(res) == 3
            @test res[1].id == "frankenstein_1"

            close_dataset!(h)
        end
    end

    @testset "persistence round-trip: close + reopen preserves search and tombstones" begin
        mktempworkdir() do workdir
            items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:30]]

            h = create_dataset(workdir, "roundtrip_ds")
            append_items!(h, items)
            before = search(h, items[1]["vector"]; k=5)
            delete_item!(h, 2)
            close_dataset!(h)

            h2 = open_dataset(workdir, "roundtrip_ds")
            after = search(h2, items[1]["vector"]; k=5)
            @test [r.id for r in before if r.doc_id != 2] == [r.id for r in after]
            @test !(2 in [r.doc_id for r in after])

            e = exists(h2, ["frankenstein_2"])
            @test e[1].exists && e[1].deleted

            close_dataset!(h2)
        end
    end

    @testset "read_only open avoids the write lock; a second writer collides" begin
        mktempworkdir() do workdir
            items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:10]]

            h = create_dataset(workdir, "lock_ds")
            append_items!(h, items)

            h_ro = open_dataset(workdir, "lock_ds"; read_only=true)
            @test length(search(h_ro, items[1]["vector"]; k=3)) == 3
            close_dataset!(h_ro)

            @test_throws RocksDB.RocksDBException open_dataset(workdir, "lock_ds")

            close_dataset!(h)
        end
    end

end
