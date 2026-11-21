@testset "dense dataset: append (stage), index!, search, calibrate, allknn, delete, fetch" begin
    mktempworkdir() do workdir
        items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:100]]

        h = create_project(workdir, "dense_ds", DenseVectors())
        inserted = append_items!(h, dense_items(items))
        @test inserted == 100

        # append_items! only stages a SearchGraph's vectors -- nothing is searchable
        # until an explicit index! call catches up the graph-connection backlog.
        @test isempty(search(h, items[1]["vector"], 5))

        index!(h)

        res = search(h, items[1]["vector"], 5)
        @test length(res) == 5
        @test res[1].doc_id == "frankenstein_1"
        @test res[1]._id == 1
        @test res[1].distance ≈ 0.0 atol=1e-6

        # calibrate! reports one CalibrationLevel per level: the BeamSearch it stored for that
        # level, and the recall it reaches on 20 queries that took no part in tuning it.
        # No warning: the sampled queries are declared internal (`queries_identifiers`), and
        # SimilaritySearch warns when a view of the index's own database arrives undeclared.
        cal = @test_logs min_level=Base.CoreLogging.Warn calibrate!(h; numqueries=20, rng=Xoshiro(1))
        @test [c.minrecall for c in cal] == IndexEngine.DEFAULT_MINRECALL_LEVELS
        table = h.engine.backend.opt_beamsearch
        @test all(c -> table[c.minrecall] == c.beamsearch, cal)
        @test all(c -> 0 <= c.recall_lo <= c.recall <= c.recall_hi <= 1, cal)
        @test all(c -> c.heldout && c.numqueries == 20 && c.confidence == 0.95, cal)
        @test all(c -> c.distance_evaluations > 0, cal)

        # The same rng draws the same samples and resamples.
        again = calibrate!(h; levels=0.9, numqueries=20, rng=Xoshiro(1), tradeoff=3.0, transition=(-2, 0))
        @test only(again).minrecall == 0.9f0
        @test only(again).numqueries == 20

        # Explicit queries both tune and measure, and the report says so.
        ext = calibrate!(h; levels=0.9, queries=[it["vector"] for it in items[1:30]], nboot=200)
        @test !only(ext).heldout && only(ext).numqueries == 30

        # The goal's and the bootstrap's parameters are checked here, with the engine's own error.
        @test_throws InvalidOption calibrate!(h; levels=0.9, tradeoff=1.0)
        @test_throws InvalidOption calibrate!(h; levels=0.9, transition=(1, -1))
        @test_throws InvalidOption calibrate!(h; levels=0.9, confidence=1.0)
        @test_throws InvalidOption calibrate!(h; levels=0.9, nboot=1)

        # neardup's epsilon is a distance: a negative or NaN one is refused before the library sees it.
        @test_throws InvalidOption neardup(h, -0.1)
        @test_throws InvalidOption neardup(h, NaN)

        ak = allknn(h; k=5)
        @test length(ak) == 100
        @test ak[1]._id == 1
        @test ak[1].neighbors[1] == 1
        @test ak[1].dists[1] ≈ 0.0 atol=1e-6

        e_before = exists(h, ["frankenstein_2", "does-not-exist"])
        @test e_before[1].exists && !e_before[1].deleted
        @test !e_before[2].exists

        delete_item!(h, 2)
        e_after = exists(h, ["frankenstein_2"])
        @test e_after[1].exists && e_after[1].deleted

        # A soft-deleted id is reported with deleted=true, not hidden/backfilled.
        res_after_delete = search(h, items[1]["vector"], 100)
        deleted_idx = findfirst(r -> r._id == 2, res_after_delete)
        @test deleted_idx !== nothing
        @test res_after_delete[deleted_idx].deleted
        # a soft-deleted hit reports no doc_id at all, rather than the internal id dressed
        # up as one
        @test res_after_delete[deleted_idx].doc_id === nothing

        fetched = fetch_items(h, ["frankenstein_3"])
        @test length(fetched) == 1
        @test fetched[1].doc_id == "frankenstein_3"
        @test fetched[1].keywords == items[3]["keywords"]
        @test fetched[1].meta["word_count"] == items[3]["word_count"]
        # the payload comes back too -- the dictionary form could not return it at all
        @test fetched[1].payload isa Vector{Float32}
        @test fetched[1].payload ≈ Float32.(items[3]["vector"])

        filtered = search(h, items[1]["vector"], 5; filter=(record, meta) -> record.doc_id == "frankenstein_1")
        @test length(filtered) == 1
        @test filtered[1].doc_id == "frankenstein_1"

        close_project!(h)
    end
end

@testset "calibrate! tunes with its whole sample, above the library's default of 64 too" begin
    mktempworkdir() do workdir
        # Since SimilaritySearch 1.6, `optimize_index!` draws `numqueries` (64 by default)
        # queries out of a larger set it is given; calibrate! passes the size of its own
        # sample. The library reports how many queries it used only on a verbose context.
        h = create_project(workdir, "calib_pool", DenseVectors())
        rng = Xoshiro(7)
        append_items!(h, [DenseItem(rand(rng, Float32, 8)) for _ in 1:400])
        index!(h)
        log = IOBuffer()
        h.engine.backend.ctx = SearchGraphContext(h.engine.backend.ctx; verbose=true,
                                                  reporters=InformativeLog(log; dt=0))
        cal = calibrate!(h; levels=0.9, numqueries=150, rng=Xoshiro(1))
        @test only(cal).numqueries == 150 && only(cal).heldout
        @test occursin("using 150 of a pool of 150", String(take!(log)))
        close_project!(h)
    end
end

@testset "dense dataset: a staged-but-not-yet-indexed backlog survives close + reopen" begin
    mktempworkdir() do workdir
        items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:20]]
        indexed, backlog = items[1:12], items[13:20]

        # The dense counterpart of the text test below, and it earns its own case because
        # its durability comes from somewhere else entirely: staged vectors live in a
        # `MMapMatrixDatabase`, whose persistence is opt-in -- `push_item!`/`append_items!`
        # advance the in-memory count and write the mapped bytes without msyncing them or
        # persisting the advanced header. `append_items!` flushing once per batch is what
        # makes this survive, and nothing else in this suite would notice if it stopped.
        h = create_project(workdir, "dense_backlog_ds", DenseVectors())
        append_items!(h, dense_items(indexed))
        index!(h)
        append_items!(h, dense_items(backlog))   # staged, deliberately left un-indexed
        close_project!(h)

        h2 = open_project(workdir, "dense_backlog_ds")
        # the backlog is not searchable yet -- what comes back is the indexed prefix
        ids_before = [r.doc_id for r in search(h2, backlog[1]["vector"], 5)]
        @test !(backlog[1]["doc_id"] in ids_before)

        index!(h2)
        hit = search(h2, backlog[1]["vector"], 3)
        @test hit[1].doc_id == backlog[1]["doc_id"]
        @test hit[1].distance ≈ 0.0 atol=1e-6
        close_project!(h2)
    end
end
