using Test
using HTTP
using JSON3
import SimilaritySearchClient as C

# The Julia client (server/julia/) against the suite's server. The routes themselves are tested
# by the files before this one; this file checks that the client sends what they accept and
# reads back what they answer, in the engine's types.
@testset "17. Julia client" begin
    srv = ensure_test_server()
    conn = C.Connection(srv.root_url)
    data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")
    rows = [JSON3.read(l, Dict{String,Any}) for l in readlines(data_path)[1:240]]
    repo_of(i) = i % 4 == 0 ? "cuatro" : "otro"
    vec(i) = Float32.(rows[i]["vector"])

    @testset "a dense dataset" begin
        h = C.create_project(conn, "jc_dense"; index_type="exhaustive_search", meta_schema=["n" => :int64],
                             partition_by="repo")
        @test h isa C.RemoteProject && h.id == "jc_dense"
        items = [C.DenseItem(vec(i); doc_id=rows[i]["doc_id"], keywords=["k$(i % 3)"],
                             meta=Dict("n" => i, "repo" => repo_of(i))) for i in eachindex(rows)]
        @test C.append_items!(h, items) == length(rows)
        @test C.index!(h) === nothing

        @test "jc_dense" in C.list_projects(conn)
        info = C.project_info(h)
        @test info["doc_count"] == length(rows) && info["loaded"] == true
        @test C.meta_schema(h) == ["n" => :int64]
        @test C.partition_field(h) == "repo"
        @test C.partition_graph_min(h) == info["partition_graph_min"]

        # An exact index returns the true nearest neighbors, with the engine's result type.
        dist(i, q) = sum(abs2, vec(i) .- q)
        q = vec(9)
        res = C.search(h, q, 5)
        @test res isa Vector{C.SearchResult} && length(res) == 5
        @test [r.doc_id for r in res] == [rows[i]["doc_id"] for i in sortperm(dist.(eachindex(rows), Ref(q)))[1:5]]
        @test res[1]._id == 9 && abs(res[1].distance) < 1e-4 && !res[1].deleted
        @test issorted([r.distance for r in res])

        # A batch is one request per query, with the results in the order of the queries.
        batch = C.search(h, [vec(3), vec(17), vec(40)], 3; ntasks=2)
        @test [b[1]._id for b in batch] == [3, 17, 40]
        @test batch[2] == C.search(h, vec(17), 3)

        # A filter object, a partition, and both.
        got = C.search(h, q, 5; filter=Dict("n" => Dict("gte" => 200)))
        @test !isempty(got) && all(r -> r._id >= 200, got)
        got = C.search(h, q, 5; partition="cuatro")
        @test length(got) == 5 && all(r -> r._id % 4 == 0, got)
        @test Dict(p.name => p.items for p in C.partitions(h)) == Dict("cuatro" => 60, "otro" => 180)
        @test all(p -> !p.graph, C.partitions(h))

        # Errors arrive with the engine's type names and categories.
        e = try C.search(h, q, 5; partition="cuatr"); nothing catch err; err end
        @test e isa C.UnknownPartition && e isa C.NotFound && e.status == 404
        @test occursin("cuatro", e.msg)
        @test_throws C.ResourceNotFound C.open_project(conn, "jc_missing")
        @test_throws C.ResourceNotFound C.search(C.RemoteProject(conn, "jc_missing"), q)
        @test_throws C.InvalidRequest C.search(h, q, 5; beamsearch_overrides=(bsize=8,))

        # fetch_items and exists accept an `_id` or a `doc_id`, as in the engine.
        fetched = C.fetch_items(h, [5, rows[6]["doc_id"], "no-such-id"])
        @test [f._id for f in fetched] == [5, 6]
        @test fetched[1].doc_id == rows[5]["doc_id"] && fetched[1].keywords == ["k2"]
        @test fetched[1].meta["n"] == 5 && fetched[1].meta["repo"] == "otro"
        @test fetched[1].payload === nothing
        ex = C.exists(h, [5, "no-such-id"])
        @test ex == [C.ExistsResult("5", true, false), C.ExistsResult("no-such-id", false, false)]

        @test C.delete_item!(h, 9) === nothing
        @test C.exists(h, [9])[1].deleted
        @test all(r -> r._id != 9, C.search(h, q, 5))
        @test C.close_project!(h) === nothing
    end

    @testset "a graph and its calibration" begin
        h = C.create_project(conn, "jc_graph"; index_type="searchgraph", distance="L2")
        C.append_items!(h, [C.DenseItem(vec(i); doc_id=rows[i]["doc_id"]) for i in eachindex(rows)])
        cal = C.calibrate!(h; levels=0.9, numqueries=16)
        @test length(cal) == 1 && cal[1] isa C.CalibrationLevel
        @test cal[1].minrecall ≈ 0.9 && 0 <= cal[1].recall <= 1 && cal[1].numqueries > 0
        @test C.project_info(h)["beamsearch_baseline"]["bsize"] == cal[1].beamsearch.bsize
        @test C.search(h, vec(12), 3; beamsearch_overrides=(bsize=cal[1].beamsearch.bsize,))[1]._id == 12
        # A dataset without partitions refuses to calibrate one.
        @test_throws C.InvalidOption C.calibrate!(h; partition="otro")
    end

    @testset "a text dataset and a hybrid search" begin
        t = C.create_project(conn, "jc_text"; index_type="bm25_invfile")
        @test C.append_items!(t, [C.TextItem(rows[i]["text"]; doc_id=rows[i]["doc_id"]) for i in eachindex(rows)]) ==
              length(rows)
        res = C.ftsearch(t, rows[30]["text"], 5)
        @test length(res) == 5 && res[1].doc_id == rows[30]["doc_id"]
        @test_throws C.InvalidOption C.ftsearch(t, "monster"; partition="otro")
        report = C.vocabulary_report(t)
        @test report["id"] == "jc_text" && haskey(report, "queries")

        d = C.open_project(conn, "jc_graph")
        fused = C.hybrid_search(d, t; vector=vec(30), text=rows[30]["text"], k=4)
        @test length(fused) == 4 && fused[1]._id == 30 && fused[1].doc_id == rows[30]["doc_id"]
        @test issorted([r.score for r in fused]; rev=true)
    end

    @testset "followers, with the engine's names" begin
        text_item(i) = C.TextItem(rows[i]["text"]; doc_id=rows[i]["doc_id"], keywords=rows[i]["keywords"],
                                  meta=Dict("repo" => repo_of(i)))
        t = C.create_project(conn, "jc_fl"; index_type="bm25_invfile")
        C.append_items!(t, [text_item(i) for i in 1:120])
        f = C.create_follower(t, "repo"; kind="partition", source="meta:repo")
        kw = C.create_follower(t, "kw"; kind=:keyword, source="keywords", policy=:manual)
        @test f isa C.RemoteFollower && kw isa C.RemoteFollower
        st = C.follower_status(f)
        @test st isa C.FollowerStatus && st.name == "jc_fl/repo" && st.synchronized && st.error === nothing
        @test st.kind == "PartitionFollower" && st.stats.partitions == 2
        @test sort([s.name for s in C.followers(t)]) == ["jc_fl/kw", "jc_fl/repo"]
        @test Dict(p.name => p.items for p in C.partitions(f)) == Dict("cuatro" => 30, "otro" => 90)

        # A view searches one partition: the global search restricted to it.
        v = C.open_follower(t, "jc_fl/repo=cuatro")
        @test v isa C.RemoteFollowerView && C.follower_status(v).name == "jc_fl/repo"
        q = rows[8]["text"]
        want = [r._id for r in C.ftsearch(t, q, 120) if r._id % 4 == 0]
        @test [r._id for r in C.ftsearch(v, q, 5)] == want[1:min(5, end)]
        @test_throws C.UnknownPartition C.ftsearch(C.open_follower(t, "repo=cinco"), q, 3)
        @test_throws C.UnknownFollower C.open_follower(t, "rep")
        @test_throws ArgumentError C.open_follower(t, "otro/repo")

        # A keyword follower: a search by keys and its posting lists.
        key = "Elizabeth"
        with_key = [i for i in 1:120 if key in rows[i]["keywords"]]
        @test C.posting_list(kw, key) == with_key
        @test C.posting_list(kw, key; limit=2) == with_key[1:2]
        @test sort([r._id for r in C.search(kw, [key], 50)]) == with_key
        @test_throws C.UnsupportedOperation C.search(f, [key], 5)
        tokens = C.posting_list(t, "the"; limit=3)
        @test tokens isa Vector{Int32} && length(tokens) <= 3 && issorted(tokens)

        # `append_items!` processes the on_index follower; the manual one lags until `sync!`.
        C.append_items!(t, [text_item(i) for i in 121:140])
        @test C.follower_status(f).position == 140
        @test C.follower_status(kw).lag == 20
        @test C.sync!(kw) == 20 && C.follower_status(kw).synchronized
        @test C.rebuild_follower!(t, "kw") isa C.RemoteFollower
        @test C.follower_status(kw).position == 140

        # A dataset with followers is not deleted.
        @test_throws C.HasFollowers C.delete_project!(t)
        @test C.delete_follower!(t, "kw") === nothing
        @test C.delete_follower!(t, "repo") === nothing
        @test isempty(C.followers(t))
        @test C.delete_project!(t) === nothing

        # The partitions of a dense dataset, moved to a follower; a partition with a graph is
        # searched and calibrated through its view.
        d = C.create_project(conn, "jc_fl_dense"; index_type="searchgraph", partition_by="repo", partition_graph_min=100)
        C.append_items!(d, [C.DenseItem(vec(i); doc_id=rows[i]["doc_id"], meta=Dict("repo" => repo_of(i)))
                            for i in eachindex(rows)])
        g = C.migrate_partitions!(d, "repo"; disjoint=true)
        @test Dict(p.name => p.graph for p in C.partitions(g)) == Dict("cuatro" => false, "otro" => true)
        @test C.partition_field(d) === nothing
        otro = C.open_follower(d, "repo=otro")
        res = C.search(otro, vec(13), 5)
        @test length(res) == 5 && res[1]._id == 13 && all(r -> r._id % 4 != 0, res)
        cal = C.calibrate!(otro; levels=0.9, numqueries=16)
        @test length(cal) == 1 && cal[1] isa C.CalibrationLevel && cal[1].minrecall ≈ 0.9
        @test C.delete_follower!(d, "repo") === nothing
        @test C.delete_project!(d) === nothing
    end

    @testset "pages of a search, through a cursor" begin
        h = C.open_project(conn, "jc_dense")
        q = vec(21)
        pages = C.search_pages(h, q, 50; page_size=20)
        # Item 9 is deleted above, and the server drops deleted results before it pages them.
        want = C.search(h, q, 50)
        @test pages.total == length(want) == 49
        got = collect(pages)
        @test length.(got) == [20, 20, 9]
        @test reduce(vcat, got) == want
        # The cursor is closed after its last page, so a second pass fails at its second page.
        @test_throws C.StateConflict collect(pages)
    end

    @testset "jobs, with the engine's result types" begin
        h = C.open_project(conn, "jc_dense")
        n = length(rows)
        # Each job is a Julia process on the server; running them in tasks overlaps them.
        catching(f) = @async try f() catch e e end
        t_knn = @async C.allknn(h; k=3, poll_interval=0.5)
        t_fft = @async C.fft(h, 5; poll_interval=0.5)
        t_dup = @async C.neardup(h, 0.5; poll_interval=0.5)
        t_text = catching(() -> C.fft(C.RemoteProject(conn, "jc_text"), 3; poll_interval=0.5))
        t_missing = catching(() -> C.allknn(C.RemoteProject(conn, "jc_missing"); poll_interval=0.5))

        knn = fetch(t_knn)
        @test knn isa Vector{C.KnnRow} && length(knn) == n
        @test all(r -> length(r.neighbors) == 3 && issorted(r.dists), knn)
        r = fetch(t_fft)
        @test r isa C.CenterSelectionResult && length(r.centers) == 5 && length(r.assign) == n
        @test all(a -> 1 <= a <= 5, r.assign) && r.covering ≈ maximum(r.assigndist) && r.costdists > 0
        d = fetch(t_dup)
        @test d isa C.NearDupResult && length(d.assign) == n && d.epsilon ≈ 0.5
        @test length(d.centers) <= n

        # A failed job carries the category of the engine error, read from the exit code.
        e = fetch(t_text)
        @test e isa C.JobFailed && e.kind == "fft" && e.exit_code == 2 && e.category === C.InvalidRequest
        e = fetch(t_missing)
        @test e isa C.JobFailed && e.exit_code == 1 && e.category === nothing

        # The generic functions, on a job of its own.
        id = C.submit_job(conn, "allknn"; dataset="jc_dense", k=2)
        @test C.wait_job(conn, id; poll_interval=0.5)["kind"] == "allknn"
        @test C.job_info(conn, id).status === :completed
        @test any(j -> j.id == id && j.status === :completed, C.list_jobs(conn; kind="allknn"))
        @test length(split(strip(C.job_result(conn, id)), '\n')) == n
        @test_throws C.StateConflict C.cancel_job!(conn, id)
        @test_throws C.StateConflict C.kill_job!(conn, id)
        @test_throws C.StateConflict C.resume_job!(conn, id)
        @test_throws C.ResourceNotFound C.job_info(conn, "no-such-job")

        # dump writes a bundle on the server; load creates a dataset from it and loads it.
        bundle = C.dump_project(C.open_project(conn, "jc_graph"); poll_interval=0.5)
        @test isdir(bundle.bundle_dir) && !isempty(bundle.manifest)
        restored = C.load_project(conn, bundle.bundle_dir, "jc_graph_copy"; poll_interval=0.5)
        @test C.project_info(restored)["doc_count"] == n
        @test C.search(restored, vec(12), 1)[1].doc_id == rows[12]["doc_id"]
        C.delete_project!(restored)
    end

    @testset "a token is sent with every request" begin
        new_token(user, perm, headers) = JSON3.read(String(HTTP.post("$(srv.base_url)/admin/tokens", headers,
            JSON3.write(Dict("user" => user, "permissions" => [perm]))).body)).token
        admin = new_token("jc", "admin:*", [])
        # The shared server runs with authentication disabled; test_14 explains the switch.
        srv.app.auth_enabled[] = true
        try
            @test_throws C.Unauthenticated C.list_projects(conn)
            @test "jc_text" in C.list_projects(C.Connection(srv.root_url; token=admin))
            reader = new_token("jcr", "read:*", ["Authorization" => "Bearer $admin"])
            @test_throws C.Forbidden C.delete_project!(C.RemoteProject(C.Connection(srv.root_url; token=reader), "jc_text"))
        finally
            srv.app.auth_enabled[] = false
        end
        @test "jc_text" in C.list_projects(conn)

        C.delete_project!(C.RemoteProject(conn, "jc_text"))
        @test !("jc_text" in C.list_projects(conn))
    end
end
