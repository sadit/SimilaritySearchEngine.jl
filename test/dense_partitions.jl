# Partitions of a dense project (PLAN.md §5.10). Three partitions of 40, 300 and 1,200
# items, and item 1 in both "a" and "b". With `partition_graph_min=250`, "a" is scanned and
# "b" and "c" have graphs.
dense_partition_corpus(rng) = begin
    labels = vcat(fill("a", 40), fill("b", 300), fill("c", 1200))
    labels = labels[sortperm(rand(rng, length(labels)))]
    X = [randn(rng, Float32, 16) for _ in labels]
    part(i) = i == 1 ? ["a", "b"] : [labels[i]]
    items = [DenseItem(X[i]; doc_id="d$i", meta=Dict("p" => (i == 1 ? part(i) : labels[i]))) for i in eachindex(X)]
    members(p) = [i for i in eachindex(X) if p in part(i)]
    (; X, items, members)
end
exact_ids(X, members, q, k) = members[sortperm([sum(abs2, X[i] .- q) for i in members])[1:min(k, end)]]

@testset "dense partitions on an exact project equal the filtered global search" begin
    mktempworkdir() do workdir
        c = dense_partition_corpus(Xoshiro(21))
        Q = [randn(Xoshiro(22), Float32, 16) for _ in 1:20]
        h = create_project(workdir, "dpe"; engine=DenseEngine, backend=ExhaustiveSearch, partition_by="p")
        append_items!(h, c.items)
        inpart(p) = (record, meta) -> meta !== nothing && (meta["p"] isa AbstractString ? meta["p"] == p : p in meta["p"])
        same(h) = all(Iterators.product(Q, ("a", "b", "c"))) do (q, p)
            got = search(h, q, 10; partition=p)
            want = search(h, q, 10; filter=inpart(p), candidates=10^6)
            [r._id for r in got] == [r._id for r in want] && [r.distance for r in got] == [r.distance for r in want]
        end
        @test same(h)
        @test all(p -> !p.graph, partitions(h))
        @test 1 in [r._id for r in search(h, c.X[1], 3; partition="a")]
        @test 1 in [r._id for r in search(h, c.X[1], 3; partition="b")]
        close_project!(h)
        h = open_project(workdir, "dpe")
        @test same(h)
        close_project!(h)
    end
end

@testset "dense partitions on a graph project: scan, graph, threshold, reopen" begin
    mktempworkdir() do workdir
        c = dense_partition_corpus(Xoshiro(31))
        Q = [randn(Xoshiro(32), Float32, 16) for _ in 1:30]
        ids(h, q, p) = [r._id for r in search(h, q, 10; partition=p)]
        recall(h, p) = sum(length(intersect(ids(h, q, p), exact_ids(c.X, c.members(p), q, 10))) for q in Q) / (10 * length(Q))
        graphs(h) = Dict(p.name => p.graph for p in partitions(h))

        h = create_project(workdir, "dpg"; engine=DenseEngine, backend=SearchGraph, partition_by="p",
                           partition_graph_min=250)
        append_items!(h, c.items)
        # Staged items are not searchable yet, in a partition as in the global graph.
        @test isempty(search(h, Q[1], 10; partition="c"))
        index!(h)
        @test graphs(h) == Dict("a" => false, "b" => true, "c" => true)
        # A scanned partition returns the exact answer; a graph returns members only, with
        # the recall its construction tuned for.
        @test all(q -> ids(h, q, "a") == exact_ids(c.X, c.members("a"), q, 10), Q)
        for p in ("b", "c")
            @test all(q -> all(in(Set(c.members(p))), ids(h, q, p)), Q)
            @test recall(h, p) >= 0.8
        end
        @test 1 in ids(h, c.X[1], "a") && 1 in ids(h, c.X[1], "b")

        # A search does not calibrate a partition, and a scan has nothing to calibrate.
        @test_throws NotTrained search(h, Q[1], 10; partition="c", minrecall=0.9)
        @test_throws InvalidOption calibrate!(h; partition="a")
        @test_throws UnknownPartition search(h, Q[1], 10; partition="zz")
        report = calibrate!(h; partition="c", levels=0.9, numqueries=16, nboot=10, rng=Xoshiro(1))
        @test only(report).minrecall ≈ 0.9
        @test length(search(h, Q[1], 10; partition="c", minrecall=0.9)) == 10

        delete_item!(h, c.members("a")[1])
        q1 = c.X[c.members("a")[1]]
        @test search(h, q1, 1; partition="a")[1].deleted

        before = Dict(p => [ids(h, q, p) for q in Q] for p in ("a", "b", "c"))
        close_project!(h)

        h = open_project(workdir, "dpg")
        @test graphs(h) == Dict("a" => false, "b" => true, "c" => true)
        # The scan is exact; a partition graph is rebuilt as the global graph is, so a query
        # can visit its vertices in another order (see the reopen test of the global graph).
        @test [ids(h, q, "a") for q in Q] == before["a"]
        for p in ("b", "c")
            @test count([ids(h, q, p) for q in Q] .== before[p]) >= 0.8 * length(Q)
        end
        @test length(search(h, Q[1], 10; partition="c", minrecall=0.9)) == 10

        # "a" reaches the threshold in the next index!, and gets a graph.
        extra = [DenseItem(randn(Xoshiro(33), Float32, 16); meta=Dict("p" => "a")) for _ in 1:220]
        append_items!(h, extra)
        index!(h)
        @test graphs(h)["a"]
        @test all(in(Set(vcat(c.members("a"), length(c.items)+1:length(c.items)+220))), ids(h, Q[1], "a"))
        close_project!(h)

        h = open_project(workdir, "dpg"; read_only=true)
        @test graphs(h)["a"]
        close_project!(h)
    end
end

@testset "dense partitions: declaring later, an interrupted index!, and the options" begin
    mktempworkdir() do workdir
        c = dense_partition_corpus(Xoshiro(41))
        Q = [randn(Xoshiro(42), Float32, 16) for _ in 1:10]

        h = create_project(workdir, "dpd"; engine=DenseEngine, backend=SearchGraph)
        append_items!(h, c.items)
        index!(h)
        parts = declare_partition!(h, "p"; partition_graph_min=250)
        @test [(p.name, p.items, p.graph) for p in parts] ==
              [(n, length(c.members(n)), g) for (n, g) in (("a", false), ("b", true), ("c", true))]
        @test all(q -> [r._id for r in search(h, q, 10; partition="a")] == exact_ids(c.X, c.members("a"), q, 10), Q)
        @test_throws InvalidOption declare_partition!(h, "p")
        close_project!(h)

        # The global graph indexes, and the process stops before the partition graphs do:
        # the next open has the members and no graphs, and the next index! builds them.
        h = create_project(workdir, "dpi"; engine=DenseEngine, backend=SearchGraph, partition_by="p",
                           partition_graph_min=250)
        append_items!(h, c.items)
        IndexEngine.index!(h.engine)
        close_project!(h)
        h = open_project(workdir, "dpi")
        @test all(p -> !p.graph, partitions(h))
        @test all(q -> [r._id for r in search(h, q, 10; partition="c")] == exact_ids(c.X, c.members("c"), q, 10), Q[1:3])
        index!(h)
        @test Dict(p.name => p.graph for p in partitions(h)) == Dict("a" => false, "b" => true, "c" => true)
        close_project!(h)

        @test_throws InvalidOption create_project(workdir, "o1"; engine=DenseEngine, partition_graph_min=10)
        @test_throws InvalidOption create_project(workdir, "o2"; engine=FullTextEngine, backend=BM25InvertedFile,
                                                  textmodel=FitFromCorpus(), partition_by="p", partition_graph_min=10)
        @test_throws InvalidOption create_project(workdir, "o3"; engine=SparseEngine, dimension=8, partition_by="p")
        @test_throws InvalidOption create_project(workdir, "o4"; engine=DenseEngine, partition_by="p", partition_graph_min=0)
        h = create_project(workdir, "o5"; engine=DenseEngine)
        append_items!(h, c.items[1:10]); index!(h)
        @test_throws InvalidOption search(h, Q[1], 3; partition="a")
        close_project!(h)
    end
end
