# Partitions (PLAN.md §5.9). Sizes are deliberately uneven, one value is a list (an item in
# two partitions), and some items have no partition at all.
_repo_of(i) = i % 7 == 0 ? ["a", "c"] : i % 3 == 0 ? "b" : i % 40 == 1 ? "tiny" : i % 11 == 0 ? nothing : "a"
function partitioned_items(rows)
    map(enumerate(rows)) do (i, r)
        meta = Dict{String,Any}()
        v = _repo_of(i)
        v === nothing || (meta["repo"] = v)
        TextItem(r["text"]; doc_id=r["doc_id"], meta)
    end
end
_in_partition(i, p) = (v = _repo_of(i); v isa AbstractString ? v == p : v !== nothing && p in v)

# The partition search must equal the global search restricted to the partition: same ids,
# same order, same scores. `k` larger than the corpus makes the global list complete.
function partition_mismatches(h, rows, nitems)
    bad = 0
    for q in (3, 17, 40, 77, 101), p in ("a", "b", "c", "tiny")
        glob = ftsearch(h, rows[q]["text"], nitems)
        want = [r for r in glob if _in_partition(r._id, p)]
        got = ftsearch(h, rows[q]["text"], 10; partition=p)
        n = min(10, length(want))
        ([r._id for r in got] == [r._id for r in want[1:n]] &&
         [r.distance for r in got] ≈ [r.distance for r in want[1:n]]) || (bad += 1)
    end
    bad
end

@testset "a partition search equals the global search restricted to the partition" begin
    for textindex in (BM25(), VectorSpace())
        mktempworkdir() do workdir
            rows = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:160]]
            h = create_project(workdir, "part_ds", TextDocuments(FitFromCorpus(); index=textindex);
                               partitions=PartitionBy("repo"))
            @test partition_field(h) == "repo"
            # Two blocks: the second appends to partition lists the first one wrote.
            append_items!(h, partitioned_items(rows)[1:100])
            index!(h)
            append_items!(h, partitioned_items(rows)[101:160])
            index!(h)
            @test partition_mismatches(h, rows, 160) == 0
            counts = Dict(p.name => p.items for p in partitions(h))
            @test counts == Dict(p => count(i -> _in_partition(i, p), 1:160) for p in ("a", "b", "c", "tiny"))

            # Every partition list is ascending, and an item in two partitions is in both.
            P = SimilaritySearchEngine.Persistence
            parts = h.partitions[]
            lists = [P._decode_ids(v) for (_, v) in RocksDB.DBIterator(parts.db; cf=parts.postings)]
            @test !isempty(lists) && all(issorted, lists)
            both = findfirst(i -> _repo_of(i) == ["a", "c"], 1:160)
            for p in ("a", "c")
                @test any(r -> r._id == both, ftsearch(h, rows[both]["text"], 5; partition=p))
            end
            close_project!(h)

            h2 = open_project(workdir, "part_ds")
            @test partition_field(h2) == "repo"
            @test partition_mismatches(h2, rows, 160) == 0
            close_project!(h2)
            h3 = open_project(workdir, "part_ds"; read_only=true)
            @test partition_mismatches(h3, rows, 160) == 0
            close_project!(h3)
        end
    end
end

@testset "declaring a partition field on an indexed project" begin
    mktempworkdir() do workdir
        rows = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:160]]
        h = create_project(workdir, "late_ds", TextDocuments(FitFromCorpus()))
        @test partition_field(h) === nothing
        @test_throws InvalidOption ftsearch(h, "x", 3; partition="a")
        @test_throws InvalidOption partitions(h)
        append_items!(h, partitioned_items(rows)[1:100])
        index!(h)
        # The last 60 are staged but not indexed: they reach the partition lists through the
        # next index! call, not through the split of the lists already on disk.
        append_items!(h, partitioned_items(rows)[101:160])
        declared = declare_partition!(h, "repo")
        @test Dict(p.name => p.items for p in declared) ==
              Dict(p => count(i -> _in_partition(i, p), 1:160) for p in ("a", "b", "c", "tiny"))
        index!(h)
        @test partition_mismatches(h, rows, 160) == 0
        @test_throws InvalidOption declare_partition!(h, "repo")
        close_project!(h)
        h2 = open_project(workdir, "late_ds")
        @test partition_mismatches(h2, rows, 160) == 0
        close_project!(h2)
    end
end

@testset "partition errors are raised, not returned as empty results" begin
    mktempworkdir() do workdir
        rows = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:40]]
        h = create_project(workdir, "err_ds", TextDocuments(FitFromCorpus()); partitions=PartitionBy("repo"))
        append_items!(h, partitioned_items(rows))
        index!(h)
        err = try ftsearch(h, rows[1]["text"], 3; partition="tny"); nothing catch e; e end
        @test err isa UnknownPartition && err isa NotFound
        @test err.name == "tny" && occursin("\"tiny\"", sprint(showerror, err))

        # A value that is not a partition name refuses the whole batch before anything is
        # written: no item is staged, and no partition changes.
        staged = length(h.engine.staged)
        before = partitions(h)
        bad = [TextItem("uno"; meta=Dict{String,Any}("repo" => "a")),
               TextItem("dos"; meta=Dict{String,Any}("repo" => 2025))]
        @test_throws PayloadMismatch append_items!(h, bad)
        @test_throws PayloadMismatch append_items!(h, [TextItem("tres"; meta=Dict{String,Any}("repo" => ""))])
        @test length(h.engine.staged) == staged
        @test partitions(h) == before

        # A name appended but not indexed yet is a partition with no indexed items: empty, not
        # an error.
        append_items!(h, [TextItem("cuatro"; meta=Dict{String,Any}("repo" => "nueva"))])
        @test isempty(ftsearch(h, rows[1]["text"], 3; partition="nueva"))
        close_project!(h)

        # A dense project can be partitioned since PLAN.md §5.10; a sparse one cannot yet.
        @test_throws InvalidOption create_project(workdir, "sparse_part_ds", SparseVectors(8); partitions=PartitionBy("repo"))
        @test_throws InvalidOption create_project(workdir, "int_ds", TextDocuments(FitFromCorpus()); partitions=PartitionBy("repo"), meta_schema=["repo" => :int64])
    end
end

@testset "a project written before partitions existed opens read-only" begin
    mktempworkdir() do workdir
        rows = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:40]]
        h = create_project(workdir, "old_ds", TextDocuments(FitFromCorpus()))
        append_items!(h, text_items(rows))
        index!(h)
        before = [r._id for r in ftsearch(h, rows[5]["text"], 5)]
        close_project!(h)

        # Remove the three partition column families, which a project written before them
        # does not have. A read-only open cannot create them, and must not ask for them.
        P = SimilaritySearchEngine.Persistence
        dir = joinpath(workdir, "old_ds")
        db = RocksDB.opendb(dir; column_families=SimilaritySearchEngine.Project.existing_column_families(dir))
        for name in (P.PARTITION_TABLE_CF, P.PARTITION_MEMBERS_CF, P.PARTITION_POSTINGS_CF)
            RocksDB.checked() do errptr
                RocksDB.Lib.rocksdb_drop_column_family(db.ptr, db.column_families[name].ptr, errptr)
            end
        end
        close(db)
        @test !(P.PARTITION_TABLE_CF in SimilaritySearchEngine.Project.existing_column_families(dir))

        h2 = open_project(workdir, "old_ds"; read_only=true)
        @test [r._id for r in ftsearch(h2, rows[5]["text"], 5)] == before
        @test partition_field(h2) === nothing
        close_project!(h2)

        # A writable open creates them, and the project can then be partitioned.
        h3 = open_project(workdir, "old_ds")
        @test P.PARTITION_TABLE_CF in SimilaritySearchEngine.Project.existing_column_families(dir)
        # `verb_count` holds integers: the declaration fails on the first one and is not
        # recorded. A field no item has is a valid declaration with no partitions.
        @test_throws PayloadMismatch declare_partition!(h3, "verb_count")
        @test partition_field(h3) === nothing
        @test isempty(declare_partition!(h3, "unused_field"))
        @test partition_field(h3) == "unused_field"
        close_project!(h3)
    end
end

@testset "ftsearch with a filter, a candidate budget and a list of tokens" begin
    for textindex in (BM25(), VectorSpace())
        mktempworkdir() do workdir
            rows = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:160]]
            items = partitioned_items(rows)
            for (i, it) in enumerate(items)
                it.meta["n"] = i
            end
            h = create_project(workdir, "filt_ds", TextDocuments(FitFromCorpus(); index=textindex);
                               partitions=PartitionBy("repo"))
            append_items!(h, items)
            index!(h)
            third(rec, meta) = meta["n"] % 3 == 0
            for q in (rows[7]["text"], rows[90]["text"], "the monster and the creature")
                # With a budget that covers every item, the filtered search is the global
                # search restricted to the items that satisfy the predicate.
                all_hits = ftsearch(h, q, 160)
                want = first([r for r in all_hits if r._id % 3 == 0], 5)
                stats = SearchStats()
                got = ftsearch(h, q, 5; filter=third, candidates=160, stats)
                @test [r._id for r in got] == [r._id for r in want]
                @test [r.distance for r in got] ≈ [r.distance for r in want]
                @test stats.candidates == length(all_hits)

                # A budget smaller than the matches stops the search early, and the stats
                # say so: the index returned exactly the candidates that were asked for.
                if length(all_hits) > 8
                    short = ftsearch(h, q, 5; filter=third, candidates=8, stats)
                    @test stats.candidates == 8
                    @test [r._id for r in short] == first([r._id for r in all_hits[1:8] if r._id % 3 == 0], 5)
                end

                # A filter inside a partition.
                in_a = [r for r in all_hits if _in_partition(r._id, "a") && r._id % 3 == 0]
                got_a = ftsearch(h, q, 4; partition="a", filter=third, candidates=160)
                @test [r._id for r in got_a] == [r._id for r in first(in_a, 4)]
            end

            # A list of tokens is searched as given. The tokens of a query, as the query
            # pipeline resolves it, give the same results as the query itself.
            off = QueryPolicy(correction=:off, expansion=false)
            voc = text_profile(h).model.voc
            q = rows[12]["text"]
            tokens = [t.token for t in query_tokens(voc, q, QueryPipeline(); policy=off).terms]
            @test !isempty(tokens)
            before = vocabulary_report(h)["queries"]
            @test [r._id for r in ftsearch(h, tokens, 5; policy=off)] == [r._id for r in ftsearch(h, q, 5; policy=off)]
            @test vocabulary_report(h)["queries"] == before + 2
            # A token the vocabulary does not hold matches nothing, and is not tokenized
            # into one it holds.
            @test isempty(ftsearch(h, ["NOT A TOKEN"], 5; policy=off))
            close_project!(h)
        end
    end
end
