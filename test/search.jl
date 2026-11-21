@testset "a search borrows a context sized for one query, not for a batch" begin
    mktempworkdir() do workdir
        h = create_project(workdir, "ctx_ds", DenseVectors())
        append_items!(h, [DenseItem(Float32[i, i % 5, 1, 0]; doc_id="c$i") for i in 1:64])
        index!(h)

        # The context a search borrows carries one batch slot: a single query is one
        # element and uses one. The context the batch operations run on keeps the
        # library's default, which is what `@BATCHES` divides work over.
        pooled = IndexEngine.checkout!(h.engine.search_ctx_pool)
        @test pooled.maxbatches == IndexEngine.SEARCH_CONTEXT_MAXBATCHES == 1
        @test h.engine.backend.ctx.maxbatches > 1
        # A `SearchGraphContext` holds a beam and a set of visited vertices per slot (the
        # set is allocated when a search first uses the slot), so the difference is the
        # memory a concurrent search holds
        @test Base.summarysize(pooled) < Base.summarysize(h.engine.backend.ctx)
        IndexEngine.checkin!(h.engine.search_ctx_pool, pooled)

        # ... and the answers are the same ones
        @test length(search(h, Float32[1, 1, 1, 0], 5)) == 5
        close_project!(h)

        # The cap on batch work is a property of the process, not of the project: it is
        # passed when the project is opened, not stored in it.
        h2 = open_project(workdir, "ctx_ds"; runtime=Runtime(maxbatches=4))
        @test h2.engine.backend.ctx.maxbatches == 4
        @test IndexEngine.checkout!(h2.engine.search_ctx_pool).maxbatches == 1
        @test length(search(h2, Float32[1, 1, 1, 0], 5)) == 5
        close_project!(h2)

        h3 = open_project(workdir, "ctx_ds")
        @test h3.engine.backend.ctx.maxbatches > 4      # no cap given, library default
        close_project!(h3)
    end
end

@testset "SearchStats reports the cost of the search it was passed to" begin
    mktempworkdir() do workdir
        h = create_project(workdir, "stats_ds", DenseVectors(; index=Exhaustive()))
        append_items!(h, [DenseItem(Float32[i, i % 5, 1, 0]; doc_id="s$i") for i in 1:32])
        q = Float32[3, 3, 1, 0]

        stats = SearchStats()
        @test stats.distance_evaluations == 0
        hits = search(h, q, 4; stats)
        @test length(hits) == 4
        # An exhaustive backend compares the query against every item, so the count is
        # not merely positive, it is known.
        @test stats.distance_evaluations == 32

        # Each call overwrites rather than accumulates (the filtered path reports the cost
        # of the candidate search it ran underneath, which here is again the whole
        # collection -- an exhaustive backend has no cheaper way to answer either query).
        search(h, q, 2; filter=(record, meta) -> true, stats)
        @test stats.distance_evaluations == 32

        # Passing no stats is the default and changes nothing about the answer.
        @test [r.doc_id for r in search(h, q, 4)] == [r.doc_id for r in hits]
        close_project!(h)
    end
end

@testset "batch search answers exactly what one-at-a-time search answers" begin
    mktempworkdir() do workdir
        h = create_project(workdir, "batch_ds", DenseVectors(; index=Exhaustive()))
        vectors = [Float32[i, i % 7, i % 3, 1] for i in 1:40]
        append_items!(h, [DenseItem(v; doc_id="b$i") for (i, v) in enumerate(vectors)])
        queries = vectors[1:2:20]

        batched = search(h, queries, 5)
        @test length(batched) == length(queries)
        for (q, hits) in zip(queries, batched)
            one = search(h, q, 5)
            @test [r._id for r in hits] == [r._id for r in one]
            @test [r.doc_id for r in hits] == [r.doc_id for r in one]
            @test [r.distance for r in hits] ≈ [r.distance for r in one]
        end

        ids, dists = searchbatch(h, queries, 5)
        @test size(ids) == (5, length(queries))
        @test size(dists) == (5, length(queries))
        @test all(>(0), ids)
        @test [Int32(ids[i, 1]) for i in 1:5] == [r._id for r in batched[1]]

        # A soft-deleted hit comes back flagged and without a doc_id, same as single search.
        delete_item!(h, Int(ids[1, 1]))
        flagged = search(h, queries[1:1], 5)[1][1]
        @test flagged.deleted && flagged.doc_id === nothing
        close_project!(h)
    end
end

@testset "batch search is a whole-dataset operation: no backlog, no text projects" begin
    mktempworkdir() do workdir
        h = create_project(workdir, "batchguard_ds", DenseVectors())
        vectors = [Float32[i, 1, 2, 3] for i in 1:30]
        append_items!(h, [DenseItem(v; doc_id="g$i") for (i, v) in enumerate(vectors)])
        index!(h)
        @test size(searchbatch(h, vectors[1:3], 4)[1]) == (4, 3)

        # Staged but not indexed: refused rather than read past the adjacency list.
        append_items!(h, [DenseItem(Float32[99, 1, 2, 3]; doc_id="g_new")])
        @test_throws PendingBacklog searchbatch(h, vectors[1:3], 4)
        index!(h)
        @test size(searchbatch(h, vectors[1:3], 4)[1]) == (4, 3)
        close_project!(h)

        ht = create_project(workdir, "batchtext_ds", TextDocuments(FitFromCorpus()))
        append_items!(ht, [TextItem("some text number $i"; doc_id="t$i") for i in 1:20])
        index!(ht)
        @test_throws UnsupportedOperation searchbatch(ht, ["some text", "other text"], 3)
        close_project!(ht)
    end
end
