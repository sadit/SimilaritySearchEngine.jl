using Test
using ArgParse
using HTTP
using JSON3

@testset "CLI Interactive Mode Tests" begin

    # --- TTY gate: the one thing that genuinely needs a real subprocess -----------
    # (the rest of this file exercises interactive.jl's pure helpers in-process --
    # there is no way to drive REPL.TerminalMenus' actual keypress loop from a
    # non-interactive test harness, and `run_interactive`'s own TTY gate is precisely
    # what makes that a non-goal rather than a gap: see PLAN.md §1's "defense in depth
    # against deadlocking a Job-spawned subprocess" note.)

    cli_script = joinpath(@__DIR__, "..", "..", "src", "apps", "similarity-search.jl")
    proj = joinpath(@__DIR__, "..", "..")

    out_file = tempname()
    err_file = tempname()
    proc = run(pipeline(`julia --project=$proj $cli_script interactive`; stdin=devnull, stdout=out_file, stderr=err_file); wait=false)
    wait(proc)
    @test !success(proc)
    @test occursin("requires a real terminal", read(err_file, String))

    # --- Pure helpers, in-process (no subprocess, no TTY needed) -------------------

    settings = SimilaritySearchServer.build_settings()

    names = SimilaritySearchServer._command_menu_names(settings)
    @test names == ["build", "searchbatch", "allknn", "fft", "neardup", "hsp", "closestpair", "describe", "rebuild", "dump", "load"]
    @test "interactive" ∉ names
    @test "serve" ∉ names

    build_fields = SimilaritySearchServer._command_fields(settings, "build")
    @test Set(f.dest_name for f in build_fields) == Set(["dataset", "input", "index-kind", "distance", "workdir"])

    @test SimilaritySearchServer._is_new_id_field("build", "dataset") == true
    @test SimilaritySearchServer._is_new_id_field("load", "dataset") == true
    @test SimilaritySearchServer._is_new_id_field("describe", "dataset") == false
    @test SimilaritySearchServer._is_new_id_field("rebuild", "dataset") == false

    min_k_field = only(f for f in SimilaritySearchServer._command_fields(settings, "closestpair") if f.dest_name == "min-k")
    @test SimilaritySearchServer._convert_value(min_k_field, "12") === 12
    @test SimilaritySearchServer._convert_value(min_k_field, "12") isa Int64

    epsilon_field = only(f for f in SimilaritySearchServer._command_fields(settings, "neardup") if f.dest_name == "epsilon")
    @test SimilaritySearchServer._convert_value(epsilon_field, "0.5") === 0.5

    dataset_field = only(f for f in build_fields if f.dest_name == "dataset")
    @test SimilaritySearchServer._convert_value(dataset_field, "myid") == "myid" # untyped (Any) field passes through as-is

    # --- _discover_project_ids: both on-disk layouts, and non-datasets excluded ---

    tmp = mktempdir()
    mkpath(joinpath(tmp, "datasets", "http_ds"))
    mkpath(joinpath(tmp, "cli_ds"))
    touch(joinpath(tmp, "cli_ds", "CURRENT"))
    mkpath(joinpath(tmp, "not_a_dataset")) # no snapshot -> must be excluded
    mkpath(joinpath(tmp, "datasets")) # already covered above; sanity no crash on re-scan

    discovered = SimilaritySearchServer._discover_project_ids(tmp)
    @test Set(discovered) == Set(["http_ds", "cli_ds"])

    @test SimilaritySearchServer._discover_project_ids(mktempdir()) == String[] # empty workdir -> no candidates

    rm(tmp, force=true, recursive=true)

    # --- _format_invocation: omits default-valued flags, keeps non-default ones ---

    cmd_args_all_defaults = Dict{String, Any}("dataset" => "x", "input" => "y.jsonl", "index-kind" => "searchgraph", "distance" => "L2", "workdir" => "data")
    @test SimilaritySearchServer._format_invocation("build", cmd_args_all_defaults, build_fields) == "similarity-search build --dataset x --input y.jsonl"

    cmd_args_overridden = Dict{String, Any}("dataset" => "x", "input" => "y.jsonl", "index-kind" => "bm25_invfile", "distance" => "L2", "workdir" => "/tmp/custom")
    formatted = SimilaritySearchServer._format_invocation("build", cmd_args_overridden, build_fields)
    @test occursin("--index-kind bm25_invfile", formatted)
    @test occursin("--workdir /tmp/custom", formatted)
    @test !occursin("--distance", formatted) # L2 is still the default, stays omitted

    # --- CLI_CHOICES / range_tester: non-interactive parsing also validates now ---

    @test SimilaritySearchServer.CLI_CHOICES["index-kind"] == ["searchgraph", "exhaustive_search", "parallel_exhaustive_search", "invfile", "bm25_invfile"]
    @test SimilaritySearchServer.CLI_CHOICES["distance"] == ["L2", "Cosine", "Angle", "NormalizedCosine"]

    good = SimilaritySearchServer.parse_commandline(["build", "--dataset", "x", "--input", "y", "--index-kind", "searchgraph"])
    @test good["build"]["index-kind"] == "searchgraph"

    # An invalid choice must go through a subprocess, not an in-process call: ArgParse's
    # default error handler calls `exit()` directly when `isinteractive()` is false (true
    # for this test process), rather than throwing a catchable exception -- calling
    # `parse_commandline` in-process with a bad value would kill this whole test run.
    out_invalid = tempname()
    err_invalid = tempname()
    proc_invalid = run(pipeline(`julia --project=$proj $cli_script build --dataset x --input y --index-kind not-a-real-kind`; stdout=out_invalid, stderr=err_invalid); wait=false)
    wait(proc_invalid)
    @test !success(proc_invalid)
    @test occursin("out of range", read(err_invalid, String))

    # --- similarity-search-ctl's own interactive mode (chunk 16) -------------------
    # Same philosophy as above: everything up to the TTY-gated request()/RadioMenu loop
    # is exercised directly and in-process; the loop itself needs a real subprocess only
    # for the TTY gate. `-ctl`'s version differs structurally in two ways worth covering:
    # `jobs` is a nested subcommand (a second menu level), and `--dataset` fields never
    # get the local-workdir discovery menu (see `_prompt_field`'s `allow_dataset_discovery`
    # kwarg) since `-ctl` is a remote HTTP client with no workdir of its own.

    ctl_script = joinpath(@__DIR__, "..", "..", "src", "apps", "similarity-search-ctl.jl")

    out_ctl = tempname()
    err_ctl = tempname()
    proc_ctl = run(pipeline(`julia --project=$proj $ctl_script interactive`; stdin=devnull, stdout=out_ctl, stderr=err_ctl); wait=false)
    wait(proc_ctl)
    @test !success(proc_ctl)
    @test occursin("requires a real terminal", read(err_ctl, String))

    ctl_settings = SimilaritySearchServer.build_ctl_settings()
    ctl_names = SimilaritySearchServer._command_menu_names(ctl_settings)
    @test Set(ctl_names) == Set(["list", "stats", "log", "jobs", "add-token", "del-token", "prune-tokens", "log-tokens", "dump", "load"])
    @test "interactive" ∉ ctl_names

    jobs_sub = ctl_settings["jobs"]
    @test SimilaritySearchServer._command_menu_names(jobs_sub) == ["queue", "result", "block", "resume", "kill", "gc"]

    queue_fields = [f for f in jobs_sub["queue"].args_table.fields if f.dest_name != "help"]
    @test Set(f.dest_name for f in queue_fields) == Set(["status", "kind"])

    @test SimilaritySearchServer._format_ctl_invocation(["list"], Dict{String, Any}(), ArgParse.ArgParseField[]) == "similarity-search-ctl list"
    fmt = SimilaritySearchServer._format_ctl_invocation(["jobs", "queue"], Dict{String, Any}("status" => "blocked", "kind" => nothing), queue_fields)
    @test fmt == "similarity-search-ctl jobs queue --status blocked" # "kind" left at its (nothing) default -> omitted

    # `dispatch_ctl_command` accepts the exact reconstructed shape `run_interactive_ctl`
    # builds -- verified against the shared live test server with read-only commands only
    # (list/stats/jobs queue), never a state-mutating job kind, to avoid racing the live
    # dispatcher the way a resume/kill assertion immediately after submission would.
    with_test_server() do base_url, workdir
        resp = HTTP.post("$base_url/datasets", [], JSON3.write(Dict("id" => "ictl_ds", "index_type" => "searchgraph")))
        @test resp.status == 201

        @test SimilaritySearchServer.dispatch_ctl_command(Dict("%COMMAND%" => "list", "list" => Dict{String, Any}()), base_url) == 0
        @test SimilaritySearchServer.dispatch_ctl_command(Dict("%COMMAND%" => "stats", "stats" => Dict{String, Any}("dataset" => "ictl_ds")), base_url) == 0

        srv = ensure_test_server()
        blocked_id = Jobs.create_job!(srv.app.job_mgr, "allknn", String[])
        Jobs.update_job_state!(srv.app.job_mgr, blocked_id, Jobs.Queued, Jobs.Blocked)

        parsed_jobs_queue = Dict("%COMMAND%" => "jobs", "jobs" => Dict("%COMMAND%" => "queue", "queue" => Dict{String, Any}("status" => "blocked", "kind" => nothing)))
        @test SimilaritySearchServer.dispatch_ctl_command(parsed_jobs_queue, base_url) == 0
    end

    # --- similarity-search-server's own interactive mode + settings split (chunks 18/21) ---
    # `serve` has real --host/--port/--workdir ArgParseFields, but as of chunk 21 they live
    # on a genuinely separate ArgParseSettings (`build_serve_settings`) -- `serve` is no
    # longer reachable through the data-ops `settings` object at all (confirmed below via
    # `settings["serve"]` throwing `KeyError`, and via a real subprocess rejecting it as an
    # unknown command), and `similarity-search-server` in turn rejects every data-ops
    # command (`build` etc). `run_interactive_serve` still has no command menu -- a
    # single-purpose binary with exactly one thing to configure is dispatched straight to
    # it, never shown as one menu item among many. The actual real-subprocess-boots-with-
    # CLI-flags / config-file-fallback-workdir-bug-fix behavior was verified by hand (not
    # as an automated test here, to avoid a slow, real-port-binding integration test) --
    # see PLAN.md's chunk 18/21 notes.

    @test_throws KeyError settings["serve"]

    serve_settings = SimilaritySearchServer.build_serve_settings()
    serve_names = SimilaritySearchServer._command_menu_names(serve_settings)
    @test Set(serve_names) == Set(["serve"]) # "interactive" excluded, same as the other two binaries

    @test Set(f.dest_name for f in serve_settings["serve"].args_table.fields) == Set(["host", "port", "workdir"])

    parsed_serve = SimilaritySearchServer.parse_serve_commandline(["serve", "--host", "0.0.0.0", "--port", "9999", "--workdir", "/tmp/x"])
    @test parsed_serve["serve"]["host"] == "0.0.0.0"
    @test parsed_serve["serve"]["port"] == 9999
    @test parsed_serve["serve"]["workdir"] == "/tmp/x"

    parsed_serve_bare = SimilaritySearchServer.parse_serve_commandline(["serve"])
    @test parsed_serve_bare["serve"]["host"] === nothing
    @test parsed_serve_bare["serve"]["port"] === nothing
    @test parsed_serve_bare["serve"]["workdir"] === nothing

    serve_script = joinpath(@__DIR__, "..", "..", "src", "apps", "similarity-search-server.jl")
    out_serve = tempname()
    err_serve = tempname()
    proc_serve = run(pipeline(`julia --project=$proj $serve_script interactive`; stdin=devnull, stdout=out_serve, stderr=err_serve); wait=false)
    wait(proc_serve)
    @test !success(proc_serve)
    @test occursin("requires a real terminal", read(err_serve, String))

    # The actual "thin binary" restriction, both directions, via real subprocesses --
    # ArgParse's own error handler calls `exit()` directly when `isinteractive()` is false,
    # so this must go through a subprocess, not an in-process `parse_commandline` call
    # (same gotcha this file's `build --index-kind not-a-real-kind` check above avoids).
    out_bad1 = tempname(); err_bad1 = tempname()
    proc_bad1 = run(pipeline(`julia --project=$proj $cli_script serve --port 9999`; stdout=out_bad1, stderr=err_bad1); wait=false)
    wait(proc_bad1)
    @test !success(proc_bad1)
    @test occursin("unknown command: serve", read(err_bad1, String))

    out_bad2 = tempname(); err_bad2 = tempname()
    proc_bad2 = run(pipeline(`julia --project=$proj $serve_script build --dataset x --input y`; stdout=out_bad2, stderr=err_bad2); wait=false)
    wait(proc_bad2)
    @test !success(proc_bad2)
    @test occursin("unknown command: build", read(err_bad2, String))
end
