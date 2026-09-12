using SimilaritySearchEngine
using Test
using HTTP
using JSON3
using Dates

@testset "11. Admin job lifecycle (block/resume/kill/gc) + single-dataset unload/reload" begin

    # A dummy request: block/resume/kill/unload never read `req.body`, only `app`/the path
    # id -- and the live dispatcher on the shared test server would race any job placed in
    # its own `job_mgr`, so this section drives a throwaway `AppState` (own workdir, own
    # JobManager/CursorManager/Executor, no dispatcher started at all) and calls the
    # handlers directly, exactly like the HTTP layer would, but deterministically.
    dummy_req = HTTP.Request("POST", "/dummy")

    admin_workdir = mktempdir()
    job_mgr = Jobs.init_job_manager(admin_workdir)
    cursor_mgr = Cursors.init_cursor_manager(admin_workdir)
    executor = LocalCLIExecutor()
    token_mgr = Tokens.open_token_manager(admin_workdir)
    app2 = AppState(
        admin_workdir, job_mgr, cursor_mgr, executor, token_mgr,
        Dict{String, SimilaritySearchEngine.EmbeddedEngine}(), ReentrantLock(),
    )

    # --- block / resume ---------------------------------------------------------

    job_id = Jobs.create_job!(job_mgr, "allknn", ["allknn", "--dataset", "x", "--k", "5", "--output", "o", "--workdir", "."])
    @test first(Jobs.get_job(job_mgr, job_id)) == Jobs.Queued

    resp = Server.handle_block_job(dummy_req, app2, job_id)
    @test resp.status == 200
    @test first(Jobs.get_job(job_mgr, job_id)) == Jobs.Blocked

    resp = Server.handle_block_job(dummy_req, app2, job_id) # already blocked
    @test resp.status == 409

    resp = Server.handle_resume_job(dummy_req, app2, job_id)
    @test resp.status == 200
    @test first(Jobs.get_job(job_mgr, job_id)) == Jobs.Queued

    resp = Server.handle_resume_job(dummy_req, app2, job_id) # not blocked anymore
    @test resp.status == 409

    resp = Server.handle_block_job(dummy_req, app2, "does-not-exist")
    @test resp.status == 404
    resp = Server.handle_resume_job(dummy_req, app2, "does-not-exist")
    @test resp.status == 404

    # --- kill --------------------------------------------------------------------

    kill_cmd = ["describe", "--dataset", "nonexistent_kill_target", "--workdir", mktempdir()]
    job_id2 = Jobs.create_job!(job_mgr, "describe", kill_cmd)
    handle = Executors.submit(executor, kill_cmd)
    Jobs.update_job_state!(job_mgr, job_id2, Jobs.Queued, Jobs.Running)
    Jobs.update_job_content!(job_mgr, Jobs.Running, job_id2, Dict("executor_handle" => handle, "started_at" => string(now(UTC))))

    resp = Server.handle_kill_job(dummy_req, app2, "does-not-exist")
    @test resp.status == 404

    resp = Server.handle_kill_job(dummy_req, app2, job_id) # queued, not running
    @test resp.status == 409

    resp = Server.handle_kill_job(dummy_req, app2, job_id2)
    @test resp.status == 200
    state, record = Jobs.get_job(job_mgr, job_id2)
    @test state == Jobs.Failed
    @test record["error"] == "killed by operator"

    resp = Server.handle_kill_job(dummy_req, app2, job_id2) # already failed
    @test resp.status == 409

    # --- jobs gc: retention-based purge of completed/failed jobs, and cursor GC --

    ok_job = Jobs.create_job!(job_mgr, "allknn", String[])
    Jobs.update_job_state!(job_mgr, ok_job, Jobs.Queued, Jobs.Running)
    Jobs.finish_job!(job_mgr, ok_job, true)
    @test first(Jobs.get_job(job_mgr, ok_job)) == Jobs.Completed

    failed_job = Jobs.create_job!(job_mgr, "allknn", String[])
    Jobs.update_job_state!(job_mgr, failed_job, Jobs.Queued, Jobs.Running)
    Jobs.finish_job!(job_mgr, failed_job, false; error="boom")
    @test first(Jobs.get_job(job_mgr, failed_job)) == Jobs.Failed

    long_ttl_cursor = Cursors.create_cursor!(cursor_mgr, "some_ds", [1, 2, 3]; ttl_seconds=3600)
    short_ttl_cursor = Cursors.create_cursor!(cursor_mgr, "some_ds", [1, 2, 3]; ttl_seconds=0)

    # A large retention should leave both finished jobs (too recent to purge) untouched,
    # but the already-lapsed cursor still gets swept into expired/ and purged either way
    # (cursor TTL and job retention are independent knobs).
    resp = Server.handle_jobs_gc(HTTP.Request("POST", "/dummy", [], JSON3.write(Dict("retention_seconds" => 999_999))), app2)
    @test resp.status == 200
    body = JSON3.read(String(resp.body))
    @test isempty(body.jobs_purged)
    @test short_ttl_cursor in body.cursors_expired
    @test short_ttl_cursor in body.cursors_purged
    @test first(Jobs.get_job(job_mgr, ok_job)) == Jobs.Completed
    @test first(Jobs.get_job(job_mgr, failed_job)) == Jobs.Failed
    @test first(Cursors.get_cursor(cursor_mgr, short_ttl_cursor)) === nothing
    @test first(Cursors.get_cursor(cursor_mgr, long_ttl_cursor)) == Cursors.Open

    # retention_seconds=0 makes every already-finished job immediately eligible.
    resp = Server.handle_jobs_gc(HTTP.Request("POST", "/dummy", [], JSON3.write(Dict("retention_seconds" => 0))), app2)
    @test resp.status == 200
    body = JSON3.read(String(resp.body))
    @test ok_job in body.jobs_purged
    @test failed_job in body.jobs_purged
    @test first(Jobs.get_job(job_mgr, ok_job)) === nothing
    @test first(Jobs.get_job(job_mgr, failed_job)) === nothing
    @test first(Cursors.get_cursor(cursor_mgr, long_ttl_cursor)) == Cursors.Open # untouched

    # --- single-dataset unload / reload (PLAN.md §5.7's flagged open issue) ------

    with_test_server() do base_url, workdir
        resp = HTTP.post("$base_url/datasets", [], JSON3.write(Dict("id" => "admin_reload_ds", "index_type" => "searchgraph", "distance" => "L2")))
        @test resp.status == 201

        data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")
        docs = [JSON3.read(line) for line in readlines(data_path)[1:10]]
        resp = HTTP.post("$base_url/simsearch/admin_reload_ds/append", [], JSON3.write(Dict("items" => docs)))
        @test resp.status == 200

        resp = HTTP.post("$base_url/simsearch/admin_reload_ds/delete", [], JSON3.write(Dict("doc_id" => 4)))
        @test resp.status == 200

        # Unload releases the RocksDB write lock (and drops the live engine)...
        resp = HTTP.post("$base_url/admin/datasets/admin_reload_ds/unload", [], "")
        @test resp.status == 200
        resp = HTTP.get("$base_url/datasets/admin_reload_ds")
        @test JSON3.read(String(resp.body)).loaded == false

        resp = try
            HTTP.post("$base_url/admin/datasets/admin_reload_ds/unload", [], "") # already unloaded
        catch e
            e.response
        end
        @test resp.status == 404

        # ...which is exactly what lets a CLI `rebuild` run against it while this server
        # is still up, purging the one tombstone -- the concrete motivating case for this
        # whole feature.
        proj = joinpath(@__DIR__, "..", "..")
        cli_script = joinpath(@__DIR__, "..", "..", "src", "apps", "similarity-search.jl")
        cmd_rebuild = `julia --project=$proj $cli_script rebuild --dataset admin_reload_ds --workdir $workdir`
        out = read(cmd_rebuild, String); println("REBUILD OUT: ", out)

        # ...and reload picks the rebuilt state back up into this same live server.
        resp = HTTP.post("$base_url/admin/datasets/admin_reload_ds/reload", [], "")
        @test resp.status == 200
        resp = HTTP.get("$base_url/datasets/admin_reload_ds")
        detail = JSON3.read(String(resp.body))
        @test detail.loaded == true
        @test detail.doc_count == 9
        @test detail.tombstone_count == 0

        resp = try
            HTTP.post("$base_url/admin/datasets/does_not_exist/reload", [], "")
        catch e
            e.response
        end
        @test resp.status == 404
    end
end
