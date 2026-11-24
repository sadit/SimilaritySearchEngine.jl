using Test
using HTTP
using JSON

# `_run_ctl` is in `test_server_helpers.jl`.

@testset "13. similarity-search-ctl CLI (admin control-plane client)" begin
    with_test_server() do base_url, workdir
        port = string(HTTP.URI(base_url).port)
        srv = ensure_test_server()

        resp = HTTP.post("$base_url/datasets", [], JSON.json(Dict("id" => "ctl_ds", "index_type" => "searchgraph", "distance" => "L2")))
        @test resp.status == 201

        data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")
        docs = [JSON.parse(line) for line in readlines(data_path)[1:5]]
        resp = HTTP.post("$base_url/datasets/ctl_ds/append", [], JSON.json(Dict("items" => docs)))
        @test resp.status == 200

        # --- list / stats -------------------------------------------------------------

        code, out = _run_ctl(["--port", port, "list"])
        @test code == 0
        listed = JSON.parse(out).datasets
        entry = only(filter(d -> d.id == "ctl_ds", listed))
        @test entry.doc_count == 5

        code, out = _run_ctl(["--port", port, "stats", "--dataset", "ctl_ds"])
        @test code == 0
        @test JSON.parse(out).doc_count == 5

        code, out = _run_ctl(["--port", port, "stats", "--dataset", "does_not_exist"])
        @test code == 1
        @test occursin("404", out)

        # --- log (chunk 17: Telemetry.log_operation now wired into append/search/etc, so
        # ctl_ds's earlier append above already left a real op_log entry) ----------------

        code, out = _run_ctl(["--port", port, "log", "--dataset", "ctl_ds"])
        @test code == 0
        log_body = JSON.parse(out)
        @test log_body.total == 1
        @test log_body.entries[1].operation == "append"
        @test log_body.entries[1].details.items_inserted == 5

        code, out = _run_ctl(["--port", port, "log", "--dataset", "does_not_exist"])
        @test code == 1
        @test occursin("404", out)

        # --- add-token / del-token -----------------------------------------------------

        code, out = _run_ctl(["--port", port, "add-token", "--user", "alice", "--permissions", "read,write"])
        @test code == 0
        token = JSON.parse(out).token
        @test !isempty(token)

        code, out = _run_ctl(["--port", port, "del-token", "--token", token])
        @test code == 0
        @test JSON.parse(out).status == "revoked"

        # --- log-tokens / prune-tokens --------------------------------------------------

        code, out = _run_ctl(["--port", port, "add-token", "--user", "carol", "--permissions", ""])
        @test code == 0
        alive_token = JSON.parse(out).token

        code, out = _run_ctl(["--port", port, "add-token", "--user", "dave", "--permissions", "", "--expires-at", "2000-01-01T00:00:00"])
        @test code == 0
        expired_token = JSON.parse(out).token

        code, out = _run_ctl(["--port", port, "log-tokens"])
        @test code == 0
        logged = JSON.parse(out).tokens
        @test any(t -> t.token == alive_token, logged)
        @test any(t -> t.token == expired_token, logged)

        code, out = _run_ctl(["--port", port, "prune-tokens"])
        @test code == 0
        @test JSON.parse(out).pruned == 1

        code, out = _run_ctl(["--port", port, "log-tokens"])
        @test code == 0
        logged_after = JSON.parse(out).tokens
        @test any(t -> t.token == alive_token, logged_after)
        @test !any(t -> t.token == expired_token, logged_after)

        # --- dump / load (submits a real job and blocks until it finishes) -------------

        code, out = _run_ctl(["--port", port, "dump", "--dataset", "ctl_ds"])
        @test code == 0
        dump_out = JSON.parse(out)
        @test isdir(dump_out.bundle_dir)
        @test dump_out.manifest.index_kind == "searchgraph"

        code, out = _run_ctl(["--port", port, "load", "--bundle", dump_out.bundle_dir, "--dataset", "ctl_ds_loaded"])
        @test code == 0
        load_out = JSON.parse(out)
        @test load_out.id == "ctl_ds_loaded"

        resp = HTTP.post("$base_url/admin/datasets/ctl_ds_loaded/reload", [], "")
        @test resp.status == 200
        resp = HTTP.get("$base_url/datasets/ctl_ds_loaded")
        @test JSON.parse(String(resp.body)).doc_count == 5

        code, out = _run_ctl(["--port", port, "dump", "--dataset", "does_not_exist"])
        @test code == 1

        # --- jobs: queue/result/resume/kill/gc, driven against the real job spool ------
        # (the shared test server's dispatcher is live, so these place jobs directly into
        # the state each assertion needs rather than racing the dispatcher for it -- e.g.
        # a job placed straight into `blocked/` is never a dispatch target at all, per
        # `Jobs`'s own design, so `resume` below is fully deterministic.)

        blocked_id = Jobs.create_job!(srv.app.job_mgr, "allknn", String[])
        Jobs.update_job_state!(srv.app.job_mgr, blocked_id, Jobs.Queued, Jobs.Blocked)

        code, out = _run_ctl(["--port", port, "jobs", "queue", "--status", "blocked"])
        @test code == 0
        @test any(j -> j.id == blocked_id, JSON.parse(out).jobs)

        code, out = _run_ctl(["--port", port, "jobs", "resume", "--job-id", blocked_id])
        @test code == 0
        @test JSON.parse(out).status == "queued"
        # The job is back in `queued/`, which the live dispatcher reads: by the time this line
        # runs it may already be running or finished. On 2026-11-19 a full run found it
        # `Running`. What `resume` guarantees is that the job left `blocked/`.
        @test first(Jobs.get_job(srv.app.job_mgr, blocked_id)) in (Jobs.Queued, Jobs.Running, Jobs.Completed, Jobs.Failed)

        code, out = _run_ctl(["--port", port, "jobs", "resume", "--job-id", blocked_id]) # already queued, not blocked
        @test code == 1
        @test occursin("409", out)

        running_id = Jobs.create_job!(srv.app.job_mgr, "describe", ["describe", "--dataset", "nope", "--workdir", "/tmp"])
        handle = Executors.submit(srv.app.executor, ["describe", "--dataset", "nope", "--workdir", "/tmp"])
        Jobs.update_job_state!(srv.app.job_mgr, running_id, Jobs.Queued, Jobs.Running)
        Jobs.update_job_content!(srv.app.job_mgr, Jobs.Running, running_id, Dict("executor_handle" => handle))

        code, out = _run_ctl(["--port", port, "jobs", "kill", "--job-id", running_id])
        @test code == 0
        @test JSON.parse(out).status == "killed"
        @test first(Jobs.get_job(srv.app.job_mgr, running_id)) == Jobs.Failed

        code, out = _run_ctl(["--port", port, "jobs", "result", "--job-id", "does-not-exist"])
        @test code == 1
        @test occursin("404", out)

        result_job_id = Jobs.create_job!(srv.app.job_mgr, "allknn", String[]; extra=Dict("result_ref" => joinpath(workdir, "ctl_result_test.txt")))
        write(joinpath(workdir, "ctl_result_test.txt"), "hello from a ctl-fetched job result")
        Jobs.update_job_state!(srv.app.job_mgr, result_job_id, Jobs.Queued, Jobs.Running)
        Jobs.finish_job!(srv.app.job_mgr, result_job_id, true)

        code, out = _run_ctl(["--port", port, "jobs", "result", "--job-id", result_job_id])
        @test code == 0
        @test out == "hello from a ctl-fetched job result\n" # printed inline (no --output given)

        output_path = tempname()
        code, out = _run_ctl(["--port", port, "jobs", "result", "--job-id", result_job_id, "--output", output_path])
        @test code == 0
        @test read(output_path, String) == "hello from a ctl-fetched job result"

        code, out = _run_ctl(["--port", port, "jobs", "gc", "--retention-seconds", "0"])
        @test code == 0
        gc_body = JSON.parse(out)
        @test result_job_id in gc_body.jobs_purged
        @test first(Jobs.get_job(srv.app.job_mgr, result_job_id)) === nothing

        # --- server not reachable -------------------------------------------------------

        code, out = _run_ctl(["--port", "1", "list"]) # nothing listens on port 1
        @test code == 1
        @test occursin("not reachable", out)
    end
end
