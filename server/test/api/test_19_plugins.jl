using Test
using HTTP
using JSON3

# Plugins in the server (PLAN.md §5.13, issue #5, phases 6a and 6b). `ensure_test_server` loads the test
# plugin of `server/test/plugins/`, as `serve` loads `[plugins]` before it opens any dataset.
@testset "19. Plugins" begin
    with_test_server() do base_url, workdir
        post(path, body) = JSON3.read(String(HTTP.post("$base_url/$path", [], JSON3.write(body)).body))
        get_(path) = JSON3.read(String(HTTP.get("$base_url/$path").body))
        failed(f) = try f(); nothing catch e; e.response end
        # `String(body)` empties the vector, and a response is read twice here.
        kind_of(resp) = JSON3.read(String(copy(resp.body))).kind
        message(resp) = JSON3.read(String(copy(resp.body))).error

        listed = only(filter(p -> p.name == "SSETestPlugin", get_("plugins").plugins))
        @test listed.key_sources == ["SSETestPlugin.decade"] && listed.follower_kinds == ["SSETestPlugin.key_counts"]
        @test endswith(listed.path, "sse_test_plugin.jl")

        # A text dataset whose items have a year: 1961 to 2020, six decades.
        data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")
        year(i) = 1960 + i
        docs = map(enumerate(readlines(data_path)[1:70])) do (i, line)
            d = JSON3.read(line, Dict{String,Any})
            Dict{String,Any}("doc_id" => d["doc_id"], "text" => d["text"], "keywords" => d["keywords"], "year" => year(i))
        end
        @test HTTP.post("$base_url/datasets", [], JSON3.write(Dict("id" => "pl_txt", "index_type" => "bm25_invfile"))).status == 201
        @test HTTP.post("$base_url/datasets/pl_txt/append", [], JSON3.write(Dict("items" => docs[1:60]))).status == 200

        # A partition follower over the key source of the plugin.
        f = post("datasets/pl_txt/followers", Dict("name" => "decade", "kind" => "partition",
                 "source" => Dict("name" => "SSETestPlugin.decade", "field" => "year")))
        @test f.plugin == "SSETestPlugin" && f.source == "Decade(\"year\")" && f.position == 60
        decades = Dict(p.name => p.items for p in get_("datasets/pl_txt/followers/decade/partitions").partitions)
        @test decades == Dict("1960s" => 9, "1970s" => 10, "1980s" => 10, "1990s" => 10, "2000s" => 10, "2010s" => 10, "2020s" => 1)
        query = docs[35]["text"]
        doc_index = Dict(d["doc_id"] => i for (i, d) in enumerate(docs))
        want = [r.doc_id for r in post("datasets/pl_txt/ftsearch", Dict("text" => query, "k" => 60)).results
                if 1990 <= year(doc_index[r.doc_id]) <= 1999]
        got = post("datasets/pl_txt/followers/decade=1990s/ftsearch", Dict("text" => query, "k" => 5)).results
        @test [r.doc_id for r in got] == want[1:min(5, end)]

        # A follower of the kind of the plugin: created, processed in `append`, and reported.
        c = post("datasets/pl_txt/followers", Dict("name" => "counts", "source" => "keywords",
                 "kind" => Dict("name" => "SSETestPlugin.key_counts", "min_count" => 2)))
        @test c.kind == "KeyCounts" && c.plugin == "SSETestPlugin" && c.stats.items == 60
        @test HTTP.post("$base_url/datasets/pl_txt/append", [], JSON3.write(Dict("items" => docs[61:70]))).status == 200
        st = get_("datasets/pl_txt/followers/counts")
        @test st.position == 70 && st.synchronized && st.stats.items == 70
        @test get_("datasets/pl_txt/followers/decade").position == 70
        # The search routes call the functions of the engine, which do not know the kind.
        resp = failed(() -> HTTP.post("$base_url/datasets/pl_txt/followers/counts/search", [],
                                      JSON3.write(Dict("keys" => ["Elizabeth"]))))
        @test resp.status == 400 && kind_of(resp) == "UnsupportedOperation"

        # The errors of a wire name.
        create(body) = failed(() -> HTTP.post("$base_url/datasets/pl_txt/followers", [], JSON3.write(body)))
        resp = create(Dict("name" => "x", "kind" => "partition", "source" => "Nope.decade"))
        @test resp.status == 404 && kind_of(resp) == "PluginNotLoaded" && occursin("[plugins.Nope]", message(resp))
        resp = create(Dict("name" => "x", "kind" => "partition", "source" => "SSETestPlugin.century"))
        @test resp.status == 400 && occursin("SSETestPlugin.decade", message(resp))
        resp = create(Dict("name" => "x", "kind" => "partition", "source" => "SSETestPlugin.decade"))
        @test resp.status == 400 && occursin("refused", message(resp))
        resp = create(Dict("name" => "x", "kind" => "SSETestPlugin.key_counts", "source" => "keywords", "distance" => "dice"))
        @test resp.status == 400

        # The followers of a plugin are opened again at reload.
        @test HTTP.post("$base_url/admin/datasets/pl_txt/unload", [], "").status == 200
        @test HTTP.post("$base_url/admin/datasets/pl_txt/reload", [], "").status == 200
        @test get_("datasets/pl_txt/followers/counts").open && get_("datasets/pl_txt/followers/decade").open

        # A process that has not registered the plugin does not open its followers, reports why,
        # and keeps the leader available.
        mod = SimilaritySearchServer.Server.PLUGINS["SSETestPlugin"].mod
        delete!(SimilaritySearchEngine._PLUGINS, "SSETestPlugin")
        try
            @test HTTP.post("$base_url/admin/datasets/pl_txt/unload", [], "").status == 200
            @test HTTP.post("$base_url/admin/datasets/pl_txt/reload", [], "").status == 200
            st = get_("datasets/pl_txt/followers/decade")
            @test !st.open && occursin("SSETestPlugin", st.error)
            @test !isempty(post("datasets/pl_txt/ftsearch", Dict("text" => query, "k" => 3)).results)
            resp = failed(() -> HTTP.post("$base_url/datasets/pl_txt/followers/decade=1990s/ftsearch", [],
                                          JSON3.write(Dict("text" => query))))
            @test resp.status == 404 && kind_of(resp) == "PluginNotLoaded"
        finally
            SimilaritySearchEngine.register_plugin!(mod)
        end
        @test HTTP.post("$base_url/admin/datasets/pl_txt/unload", [], "").status == 200
        @test HTTP.post("$base_url/admin/datasets/pl_txt/reload", [], "").status == 200
        @test get_("datasets/pl_txt/followers/decade").open

        # similarity-search-ctl lists the plugins, and creates a follower of a kind of a plugin.
        port = string(HTTP.URI(base_url).port)
        ctl(args...) = _run_ctl(String["--port", port, (string(a) for a in args)...])
        code, out = ctl("plugins")
        @test code == 0 && "SSETestPlugin.key_counts" in only(filter(p -> p.name == "SSETestPlugin", JSON3.read(out).plugins)).follower_kinds
        code, out = ctl("follower-create", "--dataset", "pl_txt", "--name", "counts2", "--kind", "SSETestPlugin.key_counts",
                        "--kind-options", "{\"min_count\": 3}", "--source", "keywords")
        @test code == 0 && JSON3.read(out).kind == "KeyCounts"
        code, out = ctl("follower-create", "--dataset", "pl_txt", "--name", "x", "--kind", "partition", "--source", "Nope.decade")
        @test code == 1 && occursin("404", out) && occursin("PluginNotLoaded", out)

        for name in ("decade", "counts", "counts2")
            @test HTTP.delete("$base_url/datasets/pl_txt/followers/$name").status == 200
        end
        @test HTTP.delete("$base_url/datasets/pl_txt").status == 200
    end
end
