@testset "persistence round-trip: close + reopen preserves search and tombstones" begin
    mktempworkdir() do workdir
        items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:30]]

        h = create_project(workdir, "roundtrip_ds", DenseVectors())
        append_items!(h, dense_items(items))
        index!(h)
        before = search(h, items[1]["vector"], 5)
        delete_item!(h, 2)
        close_project!(h)

        h2 = open_project(workdir, "roundtrip_ds")
        after = search(h2, items[1]["vector"], 5)
        @test [r.doc_id for r in before if r._id != 2] == [r.doc_id for r in after]
        @test !(2 in [r._id for r in after])

        e = exists(h2, ["frankenstein_2"])
        @test e[1].exists && e[1].deleted

        close_project!(h2)
    end
end

@testset "the beam search of a graph survives a reopen" begin
    # CHANGELOG.md §0, item 13: the graph used to be rebuilt with `BeamSearch()` and no hints, so
    # a search without `minrecall` took 350 distance evaluations instead of 2,534 after a
    # reopen, and started from random points. With both saved, a search after the reopen
    # costs about what it cost before.
    mktempworkdir() do workdir
        rng = Xoshiro(13)
        X = [randn(rng, Float32, 16) for _ in 1:4000]
        beam(h) = IndexEngine.current_beamsearch(h.engine)
        fields(bs) = (bs.bsize, bs.Δ, bs.maxvisits)
        evals(h) = [(st = SearchStats(); search(h, X[i], 10; stats=st); st.distance_evaluations) for i in 1:20]
        results(h) = [[r._id for r in search(h, X[i], 10)] for i in 1:20]
        hints(h) = copy(h.engine.backend.index.hints)

        h = create_project(workdir, "beam_ds", DenseVectors())
        append_items!(h, [DenseItem(x) for x in X])
        index!(h)
        tuned = beam(h)
        # The test is only meaningful if `index!` moved the beam away from the default.
        @test fields(tuned) != fields(BeamSearch())
        @test !isempty(hints(h))
        before = (evals(h), results(h), hints(h))
        close_project!(h)

        h = open_project(workdir, "beam_ds")
        @test fields(beam(h)) == fields(tuned)
        @test hints(h) == before[3]
        # Not equal query by query: the reverse links are rebuilt in one pass on open, so the
        # order of a vertex's neighbors can differ from the graph that was built in batches.
        # On 2026-11-18 one query of 20 took 394 evaluations instead of 384.
        @test abs(sum(evals(h)) - sum(before[1])) <= 0.1 * sum(before[1])
        @test count(results(h) .== before[2]) >= 18
        # `calibrate!` leaves the last level it tuned in the graph, and that is kept too.
        calibrate!(h; levels=0.95, numqueries=16, nboot=10, rng=Xoshiro(1))
        calibrated = beam(h)
        close_project!(h)

        h = open_project(workdir, "beam_ds"; read_only=true)
        @test fields(beam(h)) == fields(calibrated)
        close_project!(h)
    end
end

@testset "read_only open avoids the write lock; a second writer collides" begin
    mktempworkdir() do workdir
        items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:10]]

        h = create_project(workdir, "lock_ds", DenseVectors())
        append_items!(h, dense_items(items))
        index!(h)

        h_ro = open_project(workdir, "lock_ds"; read_only=true)
        @test length(search(h_ro, items[1]["vector"], 3)) == 3
        close_project!(h_ro)

        @test_throws RocksDB.RocksDBException open_project(workdir, "lock_ds")

        close_project!(h)
    end
end

@testset "a read-only handle of an exact index closes without writing" begin
    mktempworkdir() do workdir
        items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:10]]
        h = create_project(workdir, "exact_ro", DenseVectors(; index=Exhaustive()))
        append_items!(h, dense_items(items))
        index!(h)
        # A second process reads while the first one writes, as a job of the server does.
        h_ro = open_project(workdir, "exact_ro"; read_only=true)
        @test length(search(h_ro, items[1]["vector"], 3)) == 3
        @test close_project!(h_ro) === nothing
        close_project!(h)
    end
end

@testset "an exact project stores its vectors in the dense-vector file (issue #3)" begin
    P = SimilaritySearchEngine.Persistence
    mktempworkdir() do workdir
        items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:40]]
        q = items[3]["vector"]
        for (name, spec) in ("ex" => DenseVectors(; index=Exhaustive()),
                             "pex" => DenseVectors(Dist.SqL2(); index=Exhaustive(parallel=true)))
            h = create_project(workdir, name, spec)
            # Several calls: until issue #3, each inserted item rewrote the whole index.
            for chunk in Iterators.partition(dense_items(items), 10)
                append_items!(h, collect(chunk))
            end
            path = joinpath(workdir, name, "dense_vectors.mmapdb")
            @test isfile(path)
            @test !P.has_field(h.store, :index)
            before = [(r._id, r.distance) for r in search(h, q, 5)]
            @test length(before) == 5
            close_project!(h)

            h2 = open_project(workdir, name)
            @test [(r._id, r.distance) for r in search(h2, q, 5)] == before
            @test fetch_items(h2, [items[7]["doc_id"]])[1].payload == Vector{Float32}(items[7]["vector"])
            append_items!(h2, DenseItem(Float32.(q); doc_id="again"))
            @test length(search(h2, q, 41)) == 41
            close_project!(h2)

            h3 = open_project(workdir, name; read_only=true)
            @test search(h3, q, 2)[1].distance ≈ 0 atol=1e-5
            @test length(search(h3, q, 50)) == 41
            close_project!(h3)
        end

        # A project saved before 2026-11-22 holds its whole index in `:index` and has no file.
        h = create_project(workdir, "legacy", DenseVectors(; index=Exhaustive()))
        append_items!(h, dense_items(items))
        before = [(r._id, r.distance) for r in search(h, q, 5)]
        P.save_field!(h.store, :index, h.engine.backend.index)
        close_project!(h)
        path = joinpath(workdir, "legacy", "dense_vectors.mmapdb")
        rm(path)

        h_ro = open_project(workdir, "legacy"; read_only=true)
        @test [(r._id, r.distance) for r in search(h_ro, q, 5)] == before
        @test !isfile(path)                    # a read-only open does not move the vectors
        close_project!(h_ro)

        h2 = open_project(workdir, "legacy")
        @test isfile(path)
        @test !isfile(path * ".tmp")
        @test !P.has_field(h2.store, :index)
        @test [(r._id, r.distance) for r in search(h2, q, 5)] == before
        close_project!(h2)

        h3 = open_project(workdir, "legacy")
        @test [(r._id, r.distance) for r in search(h3, q, 5)] == before
        close_project!(h3)
    end
end

@testset "compaction runs for a session that wrote and is skipped for one that did not" begin
    mktempworkdir() do workdir
        h = create_project(workdir, "compact_ds", DenseVectors())
        vectors = [Float32[i, 2, 3, 4] for i in 1:40]
        append_items!(h, [DenseItem(v; doc_id="c$i") for (i, v) in enumerate(vectors)])
        index!(h)

        @test compact_project!(h) == true       # this session wrote
        @test compact_project!(h) == false      # ... and nothing changed since
        @test compact_project!(h; force=true) == true
        before = [r._id for r in search(h, vectors[3], 5)]
        close_project!(h)

        h2 = open_project(workdir, "compact_ds")
        @test [r._id for r in search(h2, vectors[3], 5)] == before   # survived compaction
        @test compact_project!(h2) == false     # a read-only session in practice
        close_project!(h2; compact=false)

        h3 = open_project(workdir, "compact_ds"; read_only=true)
        @test compact_project!(h3; force=true) == false   # a read-only handle cannot write
        @test [r._id for r in search(h3, vectors[3], 5)] == before
        close_project!(h3)
    end
end

@testset "a text project persists its index and reopens without recomputing it" begin
    for textindex in (BM25(), VectorSpace())
        mktempworkdir() do workdir
            P = SimilaritySearchEngine.Persistence
            items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:120]]

            h = create_project(workdir, "lazy_ds", TextDocuments(FitFromCorpus(); index=textindex))
            append_items!(h, text_items(items))
            index!(h)
            before = [r.doc_id for r in ftsearch(h, items[7]["text"], 5)]
            @test before[1] == items[7]["doc_id"]
            # The building session keeps its posting lists on disk too: one index, one shape.
            @test h.engine.backend.index.adj isa P.LazyPostings
            close_project!(h)

            h2 = open_project(workdir, "lazy_ds")
            # Reopening assembles the index from storage rather than replaying every object
            # back through the indexer, and the answers are identical.
            @test h2.engine.backend.index.adj isa P.LazyPostings
            @test length(h2.engine.backend.index) == length(items)
            @test [r.doc_id for r in ftsearch(h2, items[7]["text"], 5)] == before
            close_project!(h2)
        end
    end
end

@testset "a second block reaches the persisted index, incrementally" begin
    for textindex in (BM25(), VectorSpace())
        mktempworkdir() do workdir
            items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:120]]
            h = create_project(workdir, "incr_ds", TextDocuments(FitFromCorpus(); index=textindex))
            # Everything is staged before the first index! so the whole vocabulary is fitted;
            # what the second index! call exercises is the incremental write, not the fit.
            append_items!(h, text_items(items))
            index!(h)
            first_len = length(h.engine.backend.index)

            append_items!(h, text_items(items[1:20]))   # same words, new documents
            index!(h)
            @test length(h.engine.backend.index) == first_len + 20
            # The duplicate of item 3 is retrievable under its new id: its postings are in.
            live = [r._id for r in ftsearch(h, items[3]["text"], 60)]
            @test any(id -> id > first_len, live)
            close_project!(h)

            h2 = open_project(workdir, "incr_ds")
            @test length(h2.engine.backend.index) == first_len + 20
            @test [r._id for r in ftsearch(h2, items[3]["text"], 60)] == live
            close_project!(h2)
        end
    end
end

@testset "a project with no persisted index is rebuilt by one index! call" begin
    mktempworkdir() do workdir
        P = SimilaritySearchEngine.Persistence
        items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:80]]

        h = create_project(workdir, "rebuild_ds", TextDocuments(FitFromCorpus()))
        append_items!(h, text_items(items))
        index!(h)
        before = [r.doc_id for r in ftsearch(h, items[5]["text"], 5)]

        # What a project written before the index was persisted looks like: profile and staged
        # text on disk, no index. Nothing in this version rebuilds one behind the caller's back.
        for cf in (P.INVFILE_POSTINGS_CF, P.INVFILE_DOCVECS_CF)
            handle = h.project.db.column_families[cf]
            for k in [k for (k, _) in RocksDB.DBIterator(h.project.db; cf=handle)]
                delete!(h.project.db, k, cf=handle)
            end
        end
        @test !P.has_inverted_index(P.open_inverted_index_store(h.project.db))
        close_project!(h)

        h2 = open_project(workdir, "rebuild_ds")
        @test h2.engine.backend.index === nothing
        # Searching says what to do instead of answering an empty result set.
        @test_throws NoIndex ftsearch(h2, items[5]["text"], 5)
        msg = try ftsearch(h2, items[5]["text"], 5); "" catch e; sprint(showerror, e) end
        @test occursin("index!", msg)

        index!(h2)          # the documented procedure, in one call
        @test [r.doc_id for r in ftsearch(h2, items[5]["text"], 5)] == before
        close_project!(h2)

        h3 = open_project(workdir, "rebuild_ds")
        @test [r.doc_id for r in ftsearch(h3, items[5]["text"], 5)] == before
        close_project!(h3)
    end
end

@testset "the posting cache evicts by use, and only between queries" begin
    mktempworkdir() do workdir
        P = SimilaritySearchEngine.Persistence
        items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:120]]
        h = create_project(workdir, "cache_ds", TextDocuments(FitFromCorpus()); runtime=Runtime(postings_cache_max=8, postings_cache_base=4))
        append_items!(h, text_items(items))
        index!(h)
        adj = h.engine.backend.index.adj
        @test adj.maxlists == 8 && adj.baselists == 4

        # One query asks for more lists than the cache may hold: they are all cached while it
        # runs (nothing is evicted mid-query), and trimmed back to `baselists` afterwards.
        ftsearch(h, join([it["text"] for it in items[1:12]], " "), 5)
        @test length(adj.cache) == adj.baselists

        # The surviving entries are the most used ones: a repeated query keeps its lists.
        empty!(adj.cache); empty!(adj.uses)
        for _ in 1:5
            ftsearch(h, items[3]["text"], 5)
        end
        hot = copy(adj.cache)
        ftsearch(h, join([it["text"] for it in items[1:12]], " "), 5)
        @test length(adj.cache) == adj.baselists
        @test any(t -> haskey(adj.cache, t), keys(hot))
        close_project!(h)
    end
end

@testset "integer storage keys sort the way their numbers do" begin
    P = SimilaritySearchEngine.Persistence
    S = SimilaritySearchEngine.Schema
    ids = [1, 2, 255, 256, 257, 1000, 65535, 65536, 2^20]
    keys = [P._be_key(i) for i in ids]
    @test [Int(P._decode_be_key(k)) for k in keys] == ids
    # One encoding for the whole package: the metadata column families key their records
    # the same way the index does (they used to differ, little-endian against big-endian).
    @test keys == [S.be_key(i) for i in ids]
    @test [SimilaritySearchEngine.Project._id_key(Int32(i)) for i in ids] == keys
    # RocksDB compares keys bytewise, so the encoding is what decides iteration order. This
    # is the property the loaders used to depend on implicitly (and that a little-endian
    # key silently breaks past 255 -- [65536, 256, 1, 257, ...] rather than sorted).
    @test sortperm(keys) == sortperm(ids)
end

@testset "documents past the first 256 keep their identity across a reopen" begin
    mktempworkdir() do workdir
        # 300 documents on purpose: a one-byte id is where a key-encoding mistake hides, and
        # the fixtures used elsewhere (80-120 documents) never reach it.
        #
        # Each document is made unique by a letter code, not by its number: `TextConfig`
        # normalizes every run of digits to a single "0", so "documento numero 7" and
        # "documento numero 200" tokenize identically and every document would be the same
        # bag of words -- which is exactly how the first version of this test failed.
        code(i) = String([Char('a' + d) for d in digits(i, base=26, pad=3)])
        texts = ["documento $(code(i)) sobre el tema comun y algun otro asunto" for i in 1:300]
        h = create_project(workdir, "wide_ds", TextDocuments(FitFromCorpus()))
        append_items!(h, [TextItem(t; doc_id="w$i") for (i, t) in enumerate(texts)])
        index!(h)
        close_project!(h)

        h2 = open_project(workdir, "wide_ds")
        @test length(h2.engine.backend.index) == 300
        # A document vector filed under the wrong id would not fail: it would quietly score
        # the wrong document. Asking each probe document for itself is what exposes that.
        for i in (1, 200, 255, 256, 257, 299, 300)
            hit = first(ftsearch(h2, texts[i], 1))
            @test hit.doc_id == "w$i"
            @test hit._id == i
        end
        @test [it.doc_id for it in fetch_items(h2, [256, 257])] == ["w256", "w257"]
        close_project!(h2)
    end
end

@testset "the metadata of a dump bundle are JSON lines" begin
    # CHANGELOG.md §5.16, phase 9c: `dataset.jsonl` replaced `dataset.avro` on 2026-11-23.
    P = SimilaritySearchEngine.Persistence
    mktempdir() do dir
        path = joinpath(dir, "dataset.jsonl")
        meta = JSON3.write(Dict("year" => 2001, "tags" => ["a", "b"], "title" => "línea \"uno\"\nsegunda"))
        records = [P.DumpRecord(Int32(1), 1, "d1", ["gato"], String[], meta, false),
                   P.DumpRecord(Int32(2), 1, nothing, String[], ["r1", "r2"], nothing, true)]
        P.write_dump_records(path, records)
        lines = readlines(path)
        @test length(lines) == 2
        # `meta` is an object in the file, not a string, and each line is a JSON object.
        first_line = JSON3.read(lines[1])
        @test first_line.meta.year == 2001 && first_line.meta.title == "línea \"uno\"\nsegunda"
        @test JSON3.read(lines[2]).doc_id === nothing && JSON3.read(lines[2]).deleted === true

        back = P.read_dump_records(path)
        @test [r._id for r in back] == Int32[1, 2] && back[2].doc_id === nothing && back[2].meta === nothing
        @test back[1].keywords == ["gato"] && back[2].refs == ["r1", "r2"] && back[2].deleted && !back[1].deleted
        @test JSON3.read(back[1].meta, Dict{String,Any}) == JSON3.read(meta, Dict{String,Any})

        # A line edited by hand, a blank line, and a line without `schema_version`.
        write(path, """{"_id":1,"doc_id":"d1","keywords":[],"refs":[],"meta":{"year":1999}}\n\n""")
        edited = only(P.read_dump_records(path))
        @test edited.schema_version == 1 && JSON3.read(edited.meta).year == 1999

        # An empty project writes an empty file, which reads as no records.
        P.write_dump_records(path, P.DumpRecord[])
        @test filesize(path) == 0 && isempty(P.read_dump_records(path))

        for bad in ("not json", """{"doc_id":"x"}""", """{"_id":1,"meta":5}""")
            write(path, bad * "\n")
            @test_throws InvalidOption P.read_dump_records(path)
        end
    end
end
