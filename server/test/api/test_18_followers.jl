using Test
using HTTP
using JSON3

# Followers over HTTP (PLAN.md §5.12, issue #5, phases 5a and 5b). The engine's own tests check
# what a follower computes; this file checks that the server creates, opens, processes, searches
# and reports followers, that it maps their errors, and that similarity-search-ctl calls it.
@testset "18. Followers" begin
    with_test_server() do base_url, workdir
        data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")
        repo_of(i) = i % 5 == 0 ? ["norte", "sur"] : i % 2 == 0 ? "norte" : "sur"
        docs = map(enumerate(readlines(data_path)[1:120])) do (i, line)
            d = JSON3.read(line, Dict{String,Any})
            Dict{String,Any}("doc_id" => d["doc_id"], "text" => d["text"], "keywords" => d["keywords"],
                             "repo" => repo_of(i), "n" => i)
        end
        in_part(i, p) = (v = repo_of(i); v isa AbstractString ? v == p : p in v)
        post(path, body) = JSON3.read(String(HTTP.post("$base_url/$path", [], JSON3.write(body)).body))
        get_(path) = JSON3.read(String(HTTP.get("$base_url/$path").body))
        failed(f) = try f(); nothing catch e; e.response end

        @test HTTP.post("$base_url/datasets", [], JSON3.write(Dict("id" => "fl_txt", "index_type" => "bm25_invfile"))).status == 201
        @test HTTP.post("$base_url/datasets/fl_txt/append", [], JSON3.write(Dict("items" => docs[1:80]))).status == 200

        # A non-disjoint partition follower and a keyword follower.
        created = post("datasets/fl_txt/followers",
                       Dict("name" => "repo", "kind" => "partition", "source" => "meta:repo", "disjoint" => false))
        @test created.status == "created" && created.name == "fl_txt/repo" && created.position == 80
        @test created.synchronized && created.stats.partitions == 2
        kw = post("datasets/fl_txt/followers", Dict("name" => "kw", "kind" => "keyword", "source" => "keywords"))
        @test kw.kind == "KeywordFollower" && kw.position == 80
        manual = post("datasets/fl_txt/followers",
                      Dict("name" => "m", "kind" => "keyword", "source" => "meta:repo", "policy" => "manual"))
        periodic = post("datasets/fl_txt/followers",
                        Dict("name" => "p", "kind" => "keyword", "source" => "meta:repo",
                             "policy" => Dict("periodic" => Dict("seconds" => 1))))
        @test failed(() -> HTTP.post("$base_url/datasets/fl_txt/followers", [],
                                     JSON3.write(Dict("name" => "x", "kind" => "tree", "source" => "keywords")))).status == 400
        resp = failed(() -> HTTP.post("$base_url/datasets/fl_txt/followers", [],
                                      JSON3.write(Dict("name" => "repo", "kind" => "partition", "source" => "meta:repo"))))
        @test resp.status == 400 && JSON3.read(String(resp.body)).kind == "InvalidOption"

        # `append` processes an on_index follower in the same request; a manual one lags.
        @test HTTP.post("$base_url/datasets/fl_txt/append", [], JSON3.write(Dict("items" => docs[81:120]))).status == 200
        @test get_("datasets/fl_txt/followers/repo").position == 120
        @test get_("datasets/fl_txt/followers/m").lag == 40
        synced = post("datasets/fl_txt/followers/m/sync", Dict())
        @test synced.processed == 40 && synced.synchronized
        # The periodic task processes the periodic follower within a few of its checks.
        deadline = time() + 15
        while get_("datasets/fl_txt/followers/p").lag > 0 && time() < deadline
            sleep(0.25)
        end
        @test get_("datasets/fl_txt/followers/p").lag == 0

        # A partition search equals the global search restricted to the partition.
        doc_index = Dict(d["doc_id"] => i for (i, d) in enumerate(docs))
        query = docs[12]["text"]
        glob = post("datasets/fl_txt/ftsearch", Dict("text" => query, "k" => 120)).results
        for p in ("norte", "sur")
            want = [r.doc_id for r in glob if in_part(doc_index[r.doc_id], p)]
            got = post("datasets/fl_txt/followers/repo=$p/ftsearch", Dict("text" => query, "k" => 5)).results
            @test [r.doc_id for r in got] == want[1:min(5, end)]
        end
        parts = get_("datasets/fl_txt/followers/repo/partitions").partitions
        @test Dict(p.name => p.items for p in parts) == Dict(p => count(i -> in_part(i, p), 1:120) for p in ("norte", "sur"))
        resp = failed(() -> HTTP.post("$base_url/datasets/fl_txt/followers/repo=nortee/ftsearch", [],
                                      JSON3.write(Dict("text" => query))))
        @test resp.status == 404 && JSON3.read(String(resp.body)).kind == "UnknownPartition"

        # Keyword search and posting lists.
        name = first(k for d in docs for k in d["keywords"] if count(x -> k in x["keywords"], docs) >= 3)
        want = [i for (i, d) in enumerate(docs) if name in d["keywords"]]
        @test get_("datasets/fl_txt/followers/kw/postings?key=$(HTTP.URIs.escapeuri(name))").ids == want
        @test get_("datasets/fl_txt/followers/kw/postings?key=$(HTTP.URIs.escapeuri(name))&limit=2").ids == want[1:2]
        hits = post("datasets/fl_txt/followers/kw/search", Dict("keys" => [name], "k" => 50)).results
        @test sort([r._id for r in hits]) == want
        @test failed(() -> HTTP.post("$base_url/datasets/fl_txt/followers/kw/search", [], JSON3.write(Dict("k" => 3)))).status == 400
        token_ids = get_("datasets/fl_txt/postings?key=the&limit=3").ids
        @test length(token_ids) <= 3 && issorted(token_ids)

        # Status views: the dataset, one follower, the group; errors by kind.
        desc = get_("datasets/fl_txt")
        @test sort([f.name for f in desc.followers]) == ["fl_txt/kw", "fl_txt/m", "fl_txt/p", "fl_txt/repo"]
        group = get_("datasets/fl_txt/group")
        @test length(group.followers) == 4 && group.leader.id == "fl_txt"
        resp = failed(() -> HTTP.get("$base_url/datasets/fl_txt/followers/kv"))
        @test resp.status == 404 && JSON3.read(String(resp.body)).kind == "UnknownFollower"

        # A leader with followers is not deleted.
        resp = failed(() -> HTTP.delete("$base_url/datasets/fl_txt"))
        @test resp.status == 409
        body = JSON3.read(String(resp.body))
        @test body.kind == "HasFollowers" && occursin("followers/{name}", body.error)

        # The followers are opened again after unload and reload.
        @test HTTP.post("$base_url/admin/datasets/fl_txt/unload", [], "").status == 200
        @test HTTP.post("$base_url/admin/datasets/fl_txt/reload", [], "").status == 200
        st = get_("datasets/fl_txt/followers/repo")
        @test st.open && st.position == 120
        @test [r.doc_id for r in post("datasets/fl_txt/followers/repo=sur/ftsearch", Dict("text" => query, "k" => 3)).results] ==
              [r.doc_id for r in glob if in_part(doc_index[r.doc_id], "sur")][1:3]

        # Rebuild and delete a follower; then the leader can be deleted.
        @test post("datasets/fl_txt/followers/kw/rebuild", Dict()).position == 120
        for f in ("repo", "kw", "m", "p")
            @test HTTP.delete("$base_url/datasets/fl_txt/followers/$f").status == 200
        end
        @test isempty(get_("datasets/fl_txt/followers").followers)
        @test HTTP.delete("$base_url/datasets/fl_txt").status == 200

        # A dense dataset: migrate its partitions to a follower, and search one partition.
        vecs = [Float32[sin(i * j) for j in 1:8] for i in 1:60]
        @test HTTP.post("$base_url/datasets", [], JSON3.write(Dict("id" => "fl_dense", "index_type" => "exhaustive_search",
                                                                   "partition_by" => "group"))).status == 201
        items = [Dict{String,Any}("doc_id" => "v$i", "vector" => vecs[i], "group" => isodd(i) ? "odd" : "even") for i in 1:60]
        @test HTTP.post("$base_url/datasets/fl_dense/append", [], JSON3.write(Dict("items" => items))).status == 200
        @test failed(() -> HTTP.post("$base_url/datasets/fl_dense/migrate-partitions", [],
                                     JSON3.write(Dict("name" => "group")))).status == 400
        migrated = post("datasets/fl_dense/migrate-partitions", Dict("name" => "group", "disjoint" => true))
        @test migrated.status == "migrated" && migrated.stats.partitions == 2
        got = post("datasets/fl_dense/followers/group=odd/search", Dict("vector" => vecs[3], "k" => 5)).results
        @test all(r -> isodd(parse(Int, r.doc_id[2:end])), got) && got[1].doc_id == "v3"
        resp = failed(() -> HTTP.post("$base_url/datasets/fl_dense/search", [],
                                      JSON3.write(Dict("vector" => vecs[3], "partition" => "odd"))))
        @test resp.status == 400 && occursin("group", JSON3.read(String(resp.body)).error)
        @test failed(() -> HTTP.get("$base_url/datasets/fl_dense/postings?key=1")).status == 400
        @test HTTP.delete("$base_url/datasets/fl_dense/followers/group").status == 200
        @test HTTP.delete("$base_url/datasets/fl_dense").status == 200

        # The follower commands of similarity-search-ctl (phase 5b) call the routes above.
        port = string(HTTP.URI(base_url).port)
        ctl(args...) = _run_ctl(String["--port", port, (string(a) for a in args)...])
        @test HTTP.post("$base_url/datasets", [], JSON3.write(Dict("id" => "fl_ctl", "index_type" => "bm25_invfile"))).status == 201
        @test HTTP.post("$base_url/datasets/fl_ctl/append", [], JSON3.write(Dict("items" => docs[1:60]))).status == 200
        code, out = ctl("follower-create", "--dataset", "fl_ctl", "--name", "kw", "--kind", "keyword", "--source", "keywords",
                        "--policy", "manual")
        @test code == 0 && JSON3.read(out).position == 60
        code, out = ctl("follower-create", "--dataset", "fl_ctl", "--name", "x", "--kind", "partition", "--source", "keywords",
                        "--distance", "dice")
        @test code == 2 && occursin("--distance applies to a keyword follower", out)
        code, out = ctl("follower-create", "--dataset", "fl_ctl", "--name", "kw", "--kind", "keyword", "--source", "keywords")
        @test code == 1 && occursin("400", out)
        @test HTTP.post("$base_url/datasets/fl_ctl/append", [], JSON3.write(Dict("items" => docs[61:80]))).status == 200
        code, out = ctl("followers", "--dataset", "fl_ctl")
        @test code == 0 && only(JSON3.read(out).followers).lag == 20
        code, out = ctl("follower-sync", "--dataset", "fl_ctl", "--name", "kw")
        @test code == 0 && JSON3.read(out).processed == 20
        code, out = ctl("follower-rebuild", "--dataset", "fl_ctl", "--name", "kw")
        @test code == 0 && JSON3.read(out).position == 80
        @test first(ctl("follower-delete", "--dataset", "fl_ctl", "--name", "kw")) == 0
        code, out = ctl("follower-delete", "--dataset", "fl_ctl", "--name", "kw")
        @test code == 1 && occursin("404", out)
        @test HTTP.delete("$base_url/datasets/fl_ctl").status == 200
        @test HTTP.post("$base_url/datasets", [], JSON3.write(Dict("id" => "fl_ctl_dense", "index_type" => "exhaustive_search",
                                                                   "partition_by" => "group"))).status == 201
        @test HTTP.post("$base_url/datasets/fl_ctl_dense/append", [], JSON3.write(Dict("items" => items))).status == 200
        code, out = ctl("migrate-partitions", "--dataset", "fl_ctl_dense", "--name", "group", "--disjoint", "false")
        @test code == 0 && JSON3.read(out).status == "migrated"
        @test first(ctl("follower-delete", "--dataset", "fl_ctl_dense", "--name", "group")) == 0
        @test HTTP.delete("$base_url/datasets/fl_ctl_dense").status == 200

        # A follower that cannot process an item stops at it, and its status has the error. The
        # leader keeps the item.
        @test HTTP.post("$base_url/datasets", [], JSON3.write(Dict("id" => "fl_bad", "index_type" => "bm25_invfile"))).status == 201
        @test HTTP.post("$base_url/datasets/fl_bad/append", [], JSON3.write(Dict("items" => docs[1:10]))).status == 200
        post("datasets/fl_bad/followers", Dict("name" => "repo", "kind" => "partition", "source" => "meta:repo", "disjoint" => false))
        bad = merge(docs[11], Dict{String,Any}("repo" => 5))
        resp = failed(() -> HTTP.post("$base_url/datasets/fl_bad/append", [], JSON3.write(Dict("items" => [bad]))))
        @test resp.status == 400 && JSON3.read(String(resp.body)).kind == "PayloadMismatch"
        st = get_("datasets/fl_bad/followers/repo")
        @test st.position == 10 && st.lag == 1 && occursin("item 11", st.error)
        @test get_("datasets/fl_bad").doc_count == 11
        @test HTTP.delete("$base_url/datasets/fl_bad/followers/repo").status == 200
        @test HTTP.delete("$base_url/datasets/fl_bad").status == 200
    end
end
