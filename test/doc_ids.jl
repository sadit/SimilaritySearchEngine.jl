@testset "an item without a doc_id is stored, searched and fetched by _id" begin
    mktempworkdir() do workdir
        # `doc_id` is optional on every item type; the doc_id index used to call
        # `isempty(nothing)` on such an item and fail the whole append.
        h = create_project(workdir, "no_docid_ds"; engine=DenseEngine, backend=ExhaustiveSearch)
        @test append_items!(h, [DenseItem(Float32[i, 0, 0, 0]) for i in 1:5]) == 5
        hits = search(h, Float32[3, 0, 0, 0], 1)
        @test hits[1]._id == 3 && hits[1].doc_id === nothing
        @test only(fetch_items(h, [3])).doc_id === nothing
        close_project!(h)
        # the startup backfill of the doc_id index reads such records too
        close_project!(open_project(workdir, "no_docid_ds"))
    end
end

@testset "doc_id resolves through an index, and a shared doc_id returns every item" begin
    mktempworkdir() do workdir
        items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:30]]
        h = create_project(workdir, "docid_ds"; engine=DenseEngine, backend=SearchGraph)
        append_items!(h, dense_items(items))
        index!(h)

        # Exact match: "frankenstein_2" is not "frankenstein_20", even though the index
        # keys of the latter start with the former's bytes.
        one = fetch_items(h, ["frankenstein_2"])
        @test length(one) == 1 && one[1].doc_id == "frankenstein_2"

        # Nothing requires doc_id to be unique, so a repeated one brings back every item
        # carrying it, ascending by _id -- the old scan returned whichever it reached first.
        append_items!(h, [DenseItem(Float32.(items[5]["vector"]); doc_id="frankenstein_2")])
        index!(h)
        twice = fetch_items(h, ["frankenstein_2"])
        @test [it._id for it in twice] == Int32[2, 31]

        # A numeric argument still means the internal _id, not a doc_id.
        @test [it._id for it in fetch_items(h, [3])] == Int32[3]

        @test isempty(fetch_items(h, ["nope"]))
        @test exists(h, ["nope"])[1].exists == false

        # exists() summarizes the matches: found if any, deleted only when all of them are.
        @test exists(h, ["frankenstein_2"])[1].exists
        delete_item!(h, 2)
        @test exists(h, ["frankenstein_2"])[1].deleted == false   # _id 31 is still live
        delete_item!(h, 31)
        @test exists(h, ["frankenstein_2"])[1].deleted == true
        close_project!(h)
    end
end

@testset "a project written before the doc_id index gets it backfilled on the next open" begin
    mktempworkdir() do workdir
        P = SimilaritySearchEngine.Project
        items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:10]]
        h = create_project(workdir, "backfill_ds"; engine=DenseEngine, backend=SearchGraph)
        append_items!(h, dense_items(items))
        index!(h)

        # Wipe the column family down to what a project written before it existed holds.
        keys_to_drop = [k for (k, _) in RocksDB.DBIterator(h.project.db; cf=h.project.cf_idx_docid)]
        for k in keys_to_drop
            delete!(h.project.db, k, cf=h.project.cf_idx_docid)
        end
        @test isempty([k for (k, _) in RocksDB.DBIterator(h.project.db; cf=h.project.cf_idx_docid)])
        close_project!(h)

        h2 = open_project(workdir, "backfill_ds")
        @test [it.doc_id for it in fetch_items(h2, ["frankenstein_4"])] == ["frankenstein_4"]
        # ... and it is a one-time migration: the marker stops the next open from rescanning.
        @test P.backfill_docid_index!(h2.project) == 0
        close_project!(h2)
    end
end
