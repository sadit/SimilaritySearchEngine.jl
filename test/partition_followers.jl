# Partition followers (PLAN.md §5.11, issue #5, phase 3). A partition search through a follower
# must equal the global search of the leader restricted to the partition, as the partitions
# inside a project do (test/text_partitions.jl, test/dense_partitions.jl).

const PF = SimilaritySearchEngine.Persistence

# Uneven sizes, an item in two partitions, and items without a partition.
pf_repo(i) = i % 7 == 0 ? ["a", "c"] : i % 3 == 0 ? "b" : i % 40 == 1 ? "tiny" : i % 11 == 0 ? nothing : "a"
pf_in(i, p) = (v = pf_repo(i); v isa AbstractString ? v == p : v !== nothing && p in v)
pf_text_items(rows, range) = [TextItem(rows[i]["text"]; doc_id=rows[i]["doc_id"],
                                       meta=pf_repo(i) === nothing ? Dict{String,Any}() : Dict{String,Any}("repo" => pf_repo(i)))
                              for i in range]

function pf_text_mismatches(h, f, rows, nitems)
    bad = 0
    for q in (3, 17, 40, 77, 101), p in ("a", "b", "c", "tiny")
        want = [r for r in ftsearch(h, rows[q]["text"], nitems) if pf_in(r._id, p)]
        got = ftsearch(open_follower(h, "$(f)=$(p)"), rows[q]["text"], 10)
        n = min(10, length(want))
        ([r._id for r in got] == [r._id for r in want[1:n]] &&
         [r.distance for r in got] ≈ [r.distance for r in want[1:n]]) || (bad += 1)
    end
    bad
end

# A copy of `dense_partition_corpus` of test/dense_partitions.jl, so that this file runs alone.
pf_dense_corpus(rng) = begin
    labels = vcat(fill("a", 40), fill("b", 300), fill("c", 1200))
    labels = labels[sortperm(rand(rng, length(labels)))]
    X = [randn(rng, Float32, 16) for _ in labels]
    part(i) = i == 1 ? ["a", "b"] : [labels[i]]
    items = [DenseItem(X[i]; doc_id="d$i", meta=Dict("p" => (i == 1 ? part(i) : labels[i]))) for i in eachindex(X)]
    members(p) = [i for i in eachindex(X) if p in part(i)]
    (; X, items, members)
end
pf_exact_ids(X, members, q, k) = members[sortperm([sum(abs2, X[i] .- q) for i in members])[1:min(k, end)]]

pf_lists(f) = Dict(copy(k) => copy(v) for (k, v) in RocksDB.DBIterator(f.state.parts.db; cf=f.state.parts.postings))

@testset "partition follower of a text leader equals the restricted global search" begin
    for textindex in (BM25(), VectorSpace())
        mktempworkdir() do workdir
            rows = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:160]]
            h = create_project(workdir, "txt", TextDocuments(FitFromCorpus(); index=textindex))
            append_items!(h, pf_text_items(rows, 1:100))
            index!(h)
            f = create_follower(h, "repo", PartitionFollower(MetaValue("repo"); disjoint=false))
            @test f.position == 100
            # The second block reaches the follower through `index!` of the leader (OnIndex).
            append_items!(h, pf_text_items(rows, 101:160))
            @test f.position == 100
            index!(h)
            @test f.position == 160
            @test pf_text_mismatches(h, "repo", rows, 160) == 0
            @test Dict(p.name => p.items for p in partitions(f)) ==
                  Dict(p => count(i -> pf_in(i, p), 1:160) for p in ("a", "b", "c", "tiny"))
            st = follower_status(f).stats
            @test st.partitions == 4 && st.graphs == 0 && st.largest == count(i -> pf_in(i, "a"), 1:160)

            # Processing the same items again leaves the lists and the counts as they were.
            before, counts = pf_lists(f), PF.partition_counts(f.state.parts)
            f.position = 90
            @test sync!(f; block=7) == 70
            @test pf_lists(f) == before && PF.partition_counts(f.state.parts) == counts

            # A filter and a list of tokens work as on the leader.
            third(record, meta) = record._id % 3 == 0
            v = open_follower(h, "repo=a")
            @test all(r -> r._id % 3 == 0 && pf_in(r._id, "a"), ftsearch(v, rows[5]["text"], 5; filter=third, candidates=200))
            @test_throws UnknownPartition open_follower(h, "repo=nada")
            @test_throws InvalidOption search(v, Float32[1, 2], 3)
            close_project!(h)

            h2 = open_project(workdir, "txt")
            @test pf_text_mismatches(h2, "repo", rows, 160) == 0
            close_project!(h2)
            h3 = open_project(workdir, "txt"; read_only=true)
            @test pf_text_mismatches(h3, "repo", rows, 160) == 0
            close_project!(h3)
        end
    end
end

@testset "partition follower: disjoint, refused leaders and options" begin
    mktempworkdir() do workdir
        rows = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:40]]
        h = create_project(workdir, "txt", TextDocuments(FitFromCorpus()))
        append_items!(h, pf_text_items(rows, 1:40))
        index!(h)
        err = try create_follower(h, "repo", PartitionFollower(MetaValue("repo"))); nothing catch e; e end
        @test err isa PayloadMismatch && occursin("disjoint=false", err.msg)
        @test isempty(followers(h))
        @test_throws InvalidOption create_follower(h, "g", PartitionFollower(MetaValue("repo"); graph_min=10))
        # One value per item: a disjoint follower over the doc_id.
        d = create_follower(h, "byid", PartitionFollower(DocId()))
        @test length(partitions(d)) == 40
        @test [r._id for r in ftsearch(open_follower(h, "byid=$(rows[9]["doc_id"])"), rows[9]["text"], 3)] == [9]
        close_project!(h)

        s = create_project(workdir, "sp", SparseVectors(16))
        append_items!(s, [SparseItem(sparsevec(Int32[1], Float32[1], 16); meta=Dict{String,Any}("repo" => "a"))])
        @test_throws InvalidOption create_follower(s, "repo", PartitionFollower(MetaValue("repo")))
        close_project!(s)
    end
end

@testset "partition follower of a dense leader: exact and graph" begin
    mktempworkdir() do workdir
        c = pf_dense_corpus(Xoshiro(41))
        Q = [randn(Xoshiro(42), Float32, 16) for _ in 1:20]
        inpart(p) = (record, meta) -> meta !== nothing && (meta["p"] isa AbstractString ? meta["p"] == p : p in meta["p"])

        # Exact leader: every partition is scanned, and equals the filtered global search.
        h = create_project(workdir, "ex", DenseVectors(; index=Exhaustive()))
        append_items!(h, c.items)
        f = create_follower(h, "p", PartitionFollower(MetaValue("p"); disjoint=false))
        same(h) = all(Iterators.product(Q, ("a", "b", "c"))) do (q, p)
            got = search(open_follower(h, "p=$p"), q, 10)
            want = search(h, q, 10; filter=inpart(p), candidates=10^6)
            [r._id for r in got] == [r._id for r in want] && [r.distance for r in got] == [r.distance for r in want]
        end
        @test same(h)
        @test all(p -> !p.graph, partitions(f))
        @test 1 in [r._id for r in search(open_follower(h, "p=a"), c.X[1], 3)]
        @test 1 in [r._id for r in search(open_follower(h, "p=b"), c.X[1], 3)]
        @test_throws InvalidOption ftsearch(open_follower(h, "p=a"), "text", 3)
        close_project!(h)
        h = open_project(workdir, "ex")
        @test same(h)
        close_project!(h)

        # Graph leader: a partition with at least `graph_min` members has its own graph.
        ids(v, q) = [r._id for r in search(v, q, 10)]
        recall(v, p) = sum(length(intersect(ids(v, q), pf_exact_ids(c.X, c.members(p), q, 10))) for q in Q) / (10 * length(Q))
        g = create_project(workdir, "gr", DenseVectors())
        append_items!(g, c.items)
        fg = create_follower(g, "p", PartitionFollower(MetaValue("p"); disjoint=false, graph_min=250))
        @test fg.position == 0                  # nothing is indexed yet
        index!(g)
        @test fg.position == length(c.items)
        @test Dict(p.name => p.graph for p in partitions(fg)) == Dict("a" => false, "b" => true, "c" => true)
        va, vb, vc = (open_follower(g, "p=$p") for p in ("a", "b", "c"))
        @test all(q -> ids(va, q) == pf_exact_ids(c.X, c.members("a"), q, 10), Q)
        @test recall(vb, "b") >= 0.8 && recall(vc, "c") >= 0.8
        @test_throws NotTrained search(vc, Q[1], 10; minrecall=0.9)
        @test_throws InvalidOption calibrate!(va)
        report = calibrate!(vc; levels=[0.9], numqueries=16, nboot=50, rng=Xoshiro(3))
        @test only(report).minrecall == 0.9f0
        @test length(search(vc, Q[1], 10; minrecall=0.9)) == 10

        # The same items processed again leave the members and the graphs as they were.
        members_before = Dict(n => copy(p.ids) for (n, p) in fg.state.dense.parts)
        fg.position = length(c.items) - 30
        @test sync!(fg; block=11) == 30
        @test Dict(n => p.ids for (n, p) in fg.state.dense.parts) == members_before
        before = Dict(p => [ids(open_follower(g, "p=$p"), q) for q in Q] for p in ("a", "b", "c"))
        close_project!(g)

        g = open_project(workdir, "gr")
        fg = open_follower(g, "p")
        @test Dict(p.name => p.graph for p in partitions(fg)) == Dict("a" => false, "b" => true, "c" => true)
        vc = open_follower(g, "p=c")
        @test length(search(vc, Q[1], 10; minrecall=0.9)) == 10      # the calibrated levels survive
        # A reopened graph is not equal query by query (§0, item 13); most results are.
        after = Dict(p => [ids(open_follower(g, "p=$p"), q) for q in Q] for p in ("a", "b", "c"))
        @test after["a"] == before["a"]
        @test count(i -> after["c"][i] == before["c"][i], eachindex(Q)) >= 0.8 * length(Q)
        close_project!(g)
    end
end

@testset "migrate_partitions! moves the partitions inside a project to a follower" begin
    mktempworkdir() do workdir
        rows = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:120]]
        h = create_project(workdir, "txt", TextDocuments(FitFromCorpus()); partitions=PartitionBy("repo"))
        append_items!(h, pf_text_items(rows, 1:120))
        index!(h)
        old = Dict(p => [r._id for r in ftsearch(h, rows[q]["text"], 10; partition=p)] for p in ("a", "b", "c"), q in (3, 40))
        # Lists in the field: a disjoint follower is refused, and the project keeps its partitions.
        err = try migrate_partitions!(h, "repo"; disjoint=true); nothing catch e; e end
        @test err isa PayloadMismatch && occursin("disjoint=false", err.msg)
        @test partition_field(h) == "repo" && isempty(followers(h))

        f = migrate_partitions!(h, "repo"; disjoint=false)
        @test partition_field(h) === nothing
        new = Dict(p => [r._id for r in ftsearch(open_follower(h, "repo=$p"), rows[q]["text"], 10)] for p in ("a", "b", "c"), q in (3, 40))
        @test new == old
        err = try ftsearch(h, rows[3]["text"], 5; partition="a"); nothing catch e; e end
        @test err isa InvalidOption && occursin("\"repo\"", err.msg) && occursin("open_follower", err.msg)
        @test_throws InvalidOption migrate_partitions!(h, "repo"; disjoint=false)   # nothing left inside
        # New items reach the follower, not the project.
        append_items!(h, pf_text_items(rows, 1:7))
        index!(h)
        @test f.position == 127
        table = h.project.db.column_families[PF.PARTITION_TABLE_CF]
        @test isempty(collect(RocksDB.DBIterator(h.project.db; cf=table)))
        close_project!(h)

        # After a reopen, the follower equals the global search restricted to each partition.
        # Items 121 to 127 repeat rows 1 to 7, with their partitions.
        in_id(id, p) = pf_in(id <= 120 ? id : id - 120, p)
        h = open_project(workdir, "txt")
        @test partition_field(h) === nothing
        for p in ("a", "b", "c"), q in (3, 40)
            want = [r._id for r in ftsearch(h, rows[q]["text"], 127) if in_id(r._id, p)]
            @test [r._id for r in ftsearch(open_follower(h, "repo=$p"), rows[q]["text"], 10)] == want[1:min(10, end)]
        end
        close_project!(h)

        # A dense project keeps its `graph_min`.
        c = pf_dense_corpus(Xoshiro(51))
        g = create_project(workdir, "gr", DenseVectors(); partitions=PartitionBy("p"; graph_min=250))
        append_items!(g, c.items)
        index!(g)
        fg = migrate_partitions!(g, "p"; disjoint=false)
        @test fg.spec.graph_min == 250
        @test Dict(p.name => p.graph for p in partitions(fg)) == Dict("a" => false, "b" => true, "c" => true)
        @test partition_graph_min(g) === nothing
        close_project!(g)
    end
end
