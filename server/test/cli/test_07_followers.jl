using Test
using JSON

# The follower commands of `similarity-search` (CHANGELOG.md §5.12, issue #5, phase 5b), run in this
# process: `follower-create`, `follower-list`, `follower-sync`, `follower-rebuild`,
# `follower-delete`, `migrate-partitions`, `postings` and `searchbatch --follower`, and
# `describe`, `rebuild`, `dump` and `load` of a dataset with followers.
@testset "CLI followers" begin
    E = SimilaritySearchEngine
    workdir = mktempdir()
    w = ("--workdir", workdir)
    data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")
    rows = [JSON.parse(l; dicttype=Dict{String,Any}) for l in readlines(data_path)[1:150]]
    repo_of(i) = i % 5 == 0 ? ["norte", "sur"] : i % 2 == 0 ? "norte" : "sur"
    in_part(i, p) = (v = repo_of(i); v isa AbstractString ? v == p : p in v)
    item(i) = Dict{String,Any}("doc_id" => rows[i]["doc_id"], "text" => rows[i]["text"],
                               "keywords" => rows[i]["keywords"], "repo" => repo_of(i))
    input = tempname() * ".jsonl"
    write(input, join([JSON.json(item(i)) for i in 1:120], "\n"))
    queries = tempname() * ".jsonl"
    write(queries, join([JSON.json(Dict("text" => rows[i]["text"])) for i in (3, 12, 40)], "\n"))

    cli(args...) = _run_cli(String[string(a) for a in args])
    json_line(out) = JSON.parse(last(filter(!isempty, split(out, '\n'))))
    function search_ids(dataset, k, extra...)
        out = tempname()
        code, printed = cli("searchbatch", "--dataset", dataset, "--queries", queries, "--output", out, "--k", k, w..., extra...)
        code == 0 || return (code, printed)
        return [collect(JSON.parse(l).ids) for l in readlines(out)]
    end
    # The global search restricted to a partition, with `old` mapping a current id to the
    # number of its item in `rows`.
    restricted(glob, p, k; old=identity) = [filter(id -> in_part(old(id), p), g)[1:min(k, end)] for g in glob]

    @test first(cli("build", "--dataset", "fl", "--input", input, "--index-kind", "bm25_invfile", w...)) == 0

    # --- follower-create -----------------------------------------------------------------
    code, out = cli("follower-create", "--dataset", "fl", "--name", "repo", "--kind", "partition",
                    "--source", "meta:repo", "--non-disjoint", w...)
    @test code == 0
    st = json_line(out)
    @test st.name == "fl/repo" && st.position == 120 && st.synchronized && st.stats.partitions == 2
    code, out = cli("follower-create", "--dataset", "fl", "--name", "kw", "--kind", "keyword", "--source", "keywords", w...)
    @test code == 0 && json_line(out).kind == "KeywordFollower"
    @test first(cli("follower-create", "--dataset", "fl", "--name", "m", "--kind", "keyword", "--source", "meta:repo",
                    "--policy", "manual", w...)) == 0
    # A combination of arguments that does not apply is refused before the dataset is opened.
    code, out = cli("follower-create", "--dataset", "fl", "--name", "x", "--kind", "partition", "--source", "keywords",
                    "--distance", "dice", w...)
    @test code == 2 && occursin("--distance applies to a keyword follower", out)
    code, out = cli("follower-create", "--dataset", "fl", "--name", "x", "--kind", "keyword", "--source", "keywords",
                    "--policy", "periodic", w...)
    @test code == 2 && occursin("--periodic-seconds", out)
    code, out = cli("follower-create", "--dataset", "fl", "--name", "x", "--kind", "keyword", "--source", "colour", w...)
    @test code == 2 && occursin("unknown 'source'", out)
    code, out = cli("follower-create", "--dataset", "fl", "--name", "repo", "--kind", "partition", "--source", "meta:repo", w...)
    @test code == 2 && occursin("already has a follower named 'repo'", out)

    # --- searchbatch --follower ----------------------------------------------------------
    glob = search_ids("fl", 120)
    for p in ("norte", "sur")
        @test search_ids("fl", 5, "--follower", "repo=$p") == restricted(glob, p, 5)
    end
    key = "Elizabeth"
    want = [i for i in 1:120 if key in rows[i]["keywords"]]
    keyq = tempname() * ".jsonl"
    write(keyq, JSON.json(Dict("keys" => [key])))
    out = tempname()
    @test first(cli("searchbatch", "--dataset", "fl", "--queries", keyq, "--output", out, "--k", 50,
                    "--follower", "kw", w...)) == 0
    @test sort(collect(JSON.parse(only(readlines(out))).ids)) == want
    code, out = search_ids("fl", 5, "--follower", "repo")
    @test code == 2 && occursin("--follower repo=<value>", out)
    @test first(search_ids("fl", 5, "--follower", "repo=oeste")) == 4
    @test first(search_ids("fl", 5, "--follower", "repo=sur", "--partition", "sur")) == 2

    # --- postings, describe, follower-list -----------------------------------------------
    code, out = cli("postings", "--dataset", "fl", "--follower", "kw", "--key", key, w...)
    @test code == 0 && collect(json_line(out).ids) == want
    code, out = cli("postings", "--dataset", "fl", "--follower", "kw", "--key", key, "--limit", 2, w...)
    @test collect(json_line(out).ids) == want[1:2]
    code, out = cli("postings", "--dataset", "fl", "--key", "the", "--limit", 3, w...)
    @test code == 0 && length(json_line(out).ids) <= 3 && issorted(json_line(out).ids)
    code, out = cli("describe", "--dataset", "fl", w...)
    @test code == 0 && sort([f.name for f in json_line(out).followers]) == ["fl/kw", "fl/m", "fl/repo"]

    # --- follower-sync -------------------------------------------------------------------
    # Items appended through a handle that does not open the followers: every follower lags.
    h = E.open_project(workdir, "fl")
    E.append_items!(h, [SimilaritySearchServer.Server._typed_item(:text, item(i)) for i in 121:150])
    E.index!(h)
    # While another handle holds the dataset, a command that writes it is refused.
    code, out = cli("follower-sync", "--dataset", "fl", "--name", "m", w...)
    @test code == 5 && occursin("open in another process", out)
    E.close_project!(h)
    code, out = cli("follower-list", "--dataset", "fl", w...)
    @test Dict(f.name => f.lag for f in json_line(out).followers) == Dict("fl/kw" => 30, "fl/m" => 30, "fl/repo" => 30)
    code, out = cli("follower-sync", "--dataset", "fl", "--name", "m", w...)
    @test code == 0 && occursin("30 item(s) processed", out) && json_line(out).position == 150
    @test first(cli("follower-sync", "--dataset", "fl", "--name", "repo=norte", w...)) == 2

    # --- rebuild -------------------------------------------------------------------------
    # Item 7 is deleted, so the rebuild numbers the items after it again: `old` maps a new id
    # to the number of its item.
    h = E.open_project(workdir, "fl")
    E.delete_item!(h, 7)
    E.close_project!(h)
    old(j) = j < 7 ? j : j + 1
    want150 = [i - (i > 7) for i in 1:150 if i != 7 && key in rows[i]["keywords"]]
    code, out = cli("rebuild", "--dataset", "fl", w...)
    @test code == 0
    @test occursin("follower fl/kw: stale; build it again with similarity-search follower-rebuild --dataset fl --name kw", out)
    @test first(search_ids("fl", 5, "--follower", "repo=sur")) == 5
    code, out = cli("follower-rebuild", "--dataset", "fl", "--name", "repo", w...)
    @test code == 0 && json_line(out).synchronized && json_line(out).position == 149
    glob = search_ids("fl", 149)
    @test search_ids("fl", 5, "--follower", "repo=norte") == restricted(glob, "norte", 5; old)
    code, out = cli("rebuild", "--dataset", "fl", "--rebuild-followers", w...)
    @test code == 0 && count("synchronized", out) == 3
    code, out = cli("postings", "--dataset", "fl", "--follower", "kw", "--key", key, w...)
    @test collect(json_line(out).ids) == want150

    # --- dump and load -------------------------------------------------------------------
    bundle = joinpath(workdir, "bundle")
    @test first(cli("dump", "--dataset", "fl", "--output", bundle, w...)) == 0
    @test isdir(joinpath(bundle, "followers", "kw"))
    @test sort(collect(JSON.parse(read(joinpath(bundle, "manifest.json"), String)).followers)) == ["kw", "m", "repo"]
    code, out = cli("load", "--bundle", bundle, "--dataset", "fl2", w...)
    @test code == 0 && count("synchronized", out) == 3
    @test isdir(joinpath(workdir, "datasets", "fl2.followers", "kw"))
    code, out = cli("follower-list", "--dataset", "fl2", w...)
    @test sort([f.name for f in json_line(out).followers]) == ["fl2/kw", "fl2/m", "fl2/repo"]
    @test all(f -> !f.stale && f.synchronized, json_line(out).followers)
    code, out = cli("postings", "--dataset", "fl2", "--follower", "kw", "--key", key, w...)
    @test collect(json_line(out).ids) == want150

    # --- follower-delete -----------------------------------------------------------------
    for f in ("kw", "m", "repo")
        @test first(cli("follower-delete", "--dataset", "fl2", "--name", f, w...)) == 0
    end
    @test !isdir(joinpath(workdir, "datasets", "fl2.followers", "kw"))
    @test first(cli("follower-delete", "--dataset", "fl2", "--name", "kw", w...)) == 4

    # --- migrate-partitions --------------------------------------------------------------
    dense_input = tempname() * ".jsonl"
    write(dense_input, join([JSON.json(Dict("doc_id" => rows[i]["doc_id"], "vector" => rows[i]["vector"],
                                              "grp" => isodd(i) ? "odd" : "even")) for i in 1:100], "\n"))
    @test first(cli("build", "--dataset", "dn", "--input", dense_input, "--index-kind", "exhaustive_search",
                    "--partition-by", "grp", w...)) == 0
    code, out = cli("migrate-partitions", "--dataset", "dn", "--name", "grp", "--disjoint", "true", w...)
    @test code == 0 && json_line(out).stats.partitions == 2
    vq = tempname() * ".jsonl"
    write(vq, join([JSON.json(Dict("vector" => rows[i]["vector"])) for i in (3, 8)], "\n"))
    out = tempname()
    @test first(cli("searchbatch", "--dataset", "dn", "--queries", vq, "--output", out, "--k", 5,
                    "--follower", "grp=odd", w...)) == 0
    res = [collect(JSON.parse(l).ids) for l in readlines(out)]
    @test all(r -> length(r) == 5 && all(isodd, r), res) && res[1][1] == 3
    # The partitions are no longer in the dataset.
    @test first(cli("searchbatch", "--dataset", "dn", "--queries", vq, "--output", tempname(), "--partition", "odd", w...)) != 0

    # --- plugins (CHANGELOG.md §5.13) ---------------------------------------------------------
    # `--config` names the file whose `[plugins]` the commands that open a follower load.
    cfg = joinpath(workdir, "config.toml")
    plugin_file = joinpath(@__DIR__, "..", "plugins", "sse_test_plugin.jl")
    write(cfg, "[plugins.SSETestPlugin]\npath = \"$(escape_string(plugin_file))\"\n" *
               "[plugins.SSETestPlugin.settings]\nlabel = \"test\"\n")
    pc(args...) = cli("--config", cfg, args...)
    year(i) = 1960 + i
    plugin_input = tempname() * ".jsonl"
    write(plugin_input, join([JSON.json(Dict("doc_id" => rows[i]["doc_id"], "text" => rows[i]["text"],
                                                "keywords" => rows[i]["keywords"], "year" => year(i))) for i in 1:60], "\n"))
    @test first(cli("build", "--dataset", "pl", "--input", plugin_input, "--index-kind", "bm25_invfile", w...)) == 0
    code, out = pc("follower-create", "--dataset", "pl", "--name", "decade", "--kind", "partition",
                   "--source", "SSETestPlugin.decade", "--source-options", "{\"field\": \"year\"}", w...)
    @test code == 0
    st = json_line(out)
    @test st.plugin == "SSETestPlugin" && st.source == "Decade(\"year\")" && st.stats.partitions == 7
    code, out = pc("follower-create", "--dataset", "pl", "--name", "counts", "--kind", "SSETestPlugin.key_counts",
                   "--kind-options", "{\"min_count\": 2}", "--source", "keywords", w...)
    @test code == 0 && json_line(out).kind == "KeyCounts" && json_line(out).stats.items == 60
    code, out = pc("follower-create", "--dataset", "pl", "--name", "x", "--kind", "partition", "--source", "keywords",
                   "--source-options", "{}", w...)
    @test code == 2 && occursin("--source-options applies to a key source of a plugin", out)
    code, out = pc("follower-create", "--dataset", "pl", "--name", "x", "--kind", "partition",
                   "--source", "SSETestPlugin.decade", "--source-options", "field=year", w...)
    @test code == 2 && occursin("must be a JSON object", out)
    code, out = pc("follower-create", "--dataset", "pl", "--name", "x", "--kind", "partition",
                   "--source", "SSETestPlugin.century", w...)
    @test code == 2 && occursin("it has SSETestPlugin.decade", out)

    # A partition of the key source of the plugin, searched from the command line.
    pq = tempname() * ".jsonl"
    write(pq, join([JSON.json(Dict("text" => rows[i]["text"])) for i in (33, 35)], "\n"))
    out = tempname()
    @test first(pc("searchbatch", "--dataset", "pl", "--queries", pq, "--output", out, "--k", 5,
                   "--follower", "decade=1990s", w...)) == 0
    @test all(l -> all(id -> 1990 <= year(id) <= 1999, JSON.parse(l).ids), readlines(out))

    # A process that has not registered the plugin cannot open its followers, and the message
    # names the file of --config.
    plugin_module = SimilaritySearchEngine._PLUGINS["SSETestPlugin"]
    delete!(SimilaritySearchEngine._PLUGINS, "SSETestPlugin")
    try
        code, out = cli("--config", joinpath(workdir, "missing.toml"), "follower-sync", "--dataset", "pl", "--name", "decade", w...)
        @test code == 4 && occursin("plugin SSETestPlugin", out) && occursin("missing.toml, which does not exist", out)
    finally
        SimilaritySearchEngine.register_plugin!(plugin_module)
    end

    # `rebuild --rebuild-followers` builds the followers of the plugin again.
    code, out = pc("rebuild", "--dataset", "pl", "--rebuild-followers", w...)
    @test code == 0 && count("synchronized", out) == 2
end
