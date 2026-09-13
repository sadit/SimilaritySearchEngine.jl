using Test
using HTTP
using JSON3

@testset "4. Jobs and Data Access Tests" begin
    with_test_server() do base_url, workdir

        resp = HTTP.post("$base_url/datasets", [], JSON3.write(Dict("id" => "ops_dataset", "index_type" => "searchgraph")))
        @test resp.status == 201

        # 1. Fetch Metadata (Data Access by ID) -- ids that were never appended simply
        # come back as an empty result list, not an error.
        fetch_req = JSON3.write(Dict(
            "ids" => ["frankenstein_1", "frankenstein_2"]
        ))
        resp = HTTP.post("$base_url/simsearch/ops_dataset/fetch", [], fetch_req)
        @test resp.status == 200
        @test isempty(JSON3.read(String(resp.body)).results)

        # allknn/fft/neardup/hsp need a real dense corpus to operate on -- a job submitted
        # against an empty index is a real, expected failure (see execute_allknn's
        # `_require_dense` guard), not something these tests want to exercise.
        data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")
        docs = [JSON3.read(line) for line in readlines(data_path)[1:100]]
        resp = HTTP.post("$base_url/simsearch/ops_dataset/append", [], JSON3.write(Dict("items" => docs)))
        @test resp.status == 200

        function wait_terminal(job_id)
            terminal = Set(["completed", "failed"])
            status = nothing
            for _ in 1:500
                resp = HTTP.get("$base_url/jobs/$job_id")
                status = string(JSON3.read(String(resp.body)).status)
                status in terminal && break
                sleep(0.3)
            end
            return status
        end

        # 2. allknn Job Execution -- a real `SimilaritySearch.allknn` call via the
        # `allknn` CLI subcommand (PLAN.md §1/§5.5), dispatched as a subprocess and
        # polled to completion; its result file is fetched back via /result.
        resp = HTTP.post("$base_url/jobs/allknn", [], JSON3.write(Dict("dataset" => "ops_dataset", "k" => 5)))
        @test resp.status == 202
        allknn_job_id = JSON3.read(String(resp.body)).job_id
        @test wait_terminal(allknn_job_id) == "completed"

        resp = HTTP.get("$base_url/jobs/$allknn_job_id/result")
        @test resp.status == 200
        allknn_lines = split(strip(String(resp.body)), "\n")
        @test length(allknn_lines) == 100
        first_row = JSON3.read(allknn_lines[1])
        @test first_row.id == 1
        @test length(first_row.neighbors) == 5
        @test first_row.neighbors[1] == 1  # each object is its own nearest neighbor
        @test first_row.dists[1] ≈ 0.0 atol = 1e-6

        # 3. fft Job Execution (index-free, operates on the raw database directly).
        resp = HTTP.post("$base_url/jobs/fft", [], JSON3.write(Dict("dataset" => "ops_dataset", "k" => 8)))
        @test resp.status == 202
        fft_job_id = JSON3.read(String(resp.body)).job_id
        @test wait_terminal(fft_job_id) == "completed"

        resp = HTTP.get("$base_url/jobs/$fft_job_id/result")
        @test resp.status == 200
        fft_result = JSON3.read(String(resp.body))
        @test length(fft_result.centers) == 8
        @test length(unique(fft_result.centers)) == 8

        # 4. neardup Job Execution.
        resp = HTTP.post("$base_url/jobs/neardup", [], JSON3.write(Dict("dataset" => "ops_dataset", "epsilon" => 0.001)))
        @test resp.status == 202
        neardup_job_id = JSON3.read(String(resp.body)).job_id
        @test wait_terminal(neardup_job_id) == "completed"

        resp = HTTP.get("$base_url/jobs/$neardup_job_id/result")
        @test resp.status == 200
        neardup_result = JSON3.read(String(resp.body))
        # `assign`/`centers`/`dists` are the engine's own `NearDupResult` field names (the
        # CLI used to call the assignment array `nn`, from an older SimilaritySearch API).
        @test length(neardup_result.assign) == 100
        @test neardup_result.duplicate_count == length(neardup_result.assign) - length(neardup_result.centers)

        # 5. hsp Job Execution (needs a queries file -- CLI/offline-style, like searchbatch).
        resp = HTTP.post("$base_url/jobs/hsp", [], JSON3.write(Dict("dataset" => "ops_dataset", "queries" => data_path, "k" => 5)))
        @test resp.status == 202
        hsp_job_id = JSON3.read(String(resp.body)).job_id
        @test wait_terminal(hsp_job_id) == "completed"

        resp = HTTP.get("$base_url/jobs/$hsp_job_id/result")
        @test resp.status == 200
        hsp_lines = split(strip(String(resp.body)), "\n")
        @test length(hsp_lines) > 0

        # 6. Missing required params for a heavy job kind -> 400, not a submitted-then-failed job.
        resp = try
            HTTP.post("$base_url/jobs/fft", [], JSON3.write(Dict("dataset" => "ops_dataset")))
        catch e
            e.response
        end
        @test resp.status == 400

        # 7. An unsupported job kind (no explicit "command", not a known heavy kind) -> 400.
        resp = try
            HTTP.post("$base_url/jobs/not_a_real_kind", [], JSON3.write(Dict("dataset" => "ops_dataset")))
        catch e
            e.response
        end
        @test resp.status == 400

        # 8. Fetching a result before the job reaches a terminal state -> 409.
        resp = HTTP.post("$base_url/jobs/allknn", [], JSON3.write(Dict("dataset" => "ops_dataset", "k" => 5)))
        pending_job_id = JSON3.read(String(resp.body)).job_id
        resp = try
            HTTP.get("$base_url/jobs/$pending_job_id/result")
        catch e
            e.response
        end
        @test resp.status == 409
        wait_terminal(pending_job_id)  # drain it so it doesn't linger past the test

        # 9. dump/load as a Job/HTTP surface (PLAN.md §5.5's job-kind enum) -- the same
        # execute_dump/execute_load CLI functions from chunk 10, now reachable without a
        # CLI subprocess of their own. dump's result is a bundle *directory*, unlike every
        # other HEAVY_JOB_KINDS result (a single file) -- GET .../result reports it as a
        # JSON pointer (bundle_dir + parsed manifest.json) instead of streaming bytes.
        resp = HTTP.post("$base_url/jobs/dump", [], JSON3.write(Dict("dataset" => "ops_dataset")))
        @test resp.status == 202
        dump_job_id = JSON3.read(String(resp.body)).job_id
        @test wait_terminal(dump_job_id) == "completed"

        resp = HTTP.get("$base_url/jobs/$dump_job_id/result")
        @test resp.status == 200
        dump_result = JSON3.read(String(resp.body))
        @test isdir(dump_result.bundle_dir)
        @test dump_result.manifest.index_kind == "searchgraph"

        # 10. load job, targeting a brand-new dataset id from the bundle above --
        # composes with chunk 9's admin reload to bring the freshly loaded dataset live,
        # exactly like test_12_dump_load_via_admin.jl's CLI-driven version of this flow.
        resp = HTTP.post("$base_url/jobs/load", [], JSON3.write(Dict("bundle" => dump_result.bundle_dir, "dataset" => "ops_dataset_loaded")))
        @test resp.status == 202
        load_job_id = JSON3.read(String(resp.body)).job_id
        @test wait_terminal(load_job_id) == "completed"

        resp = HTTP.get("$base_url/jobs/$load_job_id/result")
        @test resp.status == 200
        load_result = JSON3.read(String(resp.body))
        @test load_result.id == "ops_dataset_loaded"

        resp = HTTP.post("$base_url/admin/datasets/ops_dataset_loaded/reload", [], "")
        @test resp.status == 200
        resp = HTTP.get("$base_url/datasets/ops_dataset_loaded")
        @test resp.status == 200
        @test JSON3.read(String(resp.body)).doc_count == 100

        resp = HTTP.post("$base_url/simsearch/ops_dataset_loaded/search", [], JSON3.write(Dict("vector" => docs[1].vector, "k" => 3)))
        @test resp.status == 200
        @test !isempty(JSON3.read(String(resp.body)).results)

        # 11. dump/load missing required params -> 400, not a submitted-then-failed job.
        resp = try
            HTTP.post("$base_url/jobs/dump", [], JSON3.write(Dict()))
        catch e
            e.response
        end
        @test resp.status == 400

        resp = try
            HTTP.post("$base_url/jobs/load", [], JSON3.write(Dict("dataset" => "ops_dataset_loaded2")))
        catch e
            e.response
        end
        @test resp.status == 400

        # 12. DELETE /api/v1/jobs/{job_id} (PLAN.md §5.5): best-effort cancel of a job
        # that hasn't started running yet -- distinct from kill (only for a running job).
        # Driven directly against the job spool (same throwaway-manager-free style as this
        # file's other job-state assertions) rather than racing the shared dispatcher.
        srv = ensure_test_server()

        queued_id = Jobs.create_job!(srv.app.job_mgr, "allknn", String[])
        resp = HTTP.delete("$base_url/jobs/$queued_id")
        @test resp.status == 200
        @test JSON3.read(String(resp.body)).status == "cancelled"
        state, record = Jobs.get_job(srv.app.job_mgr, queued_id)
        @test state == Jobs.Failed
        @test record["error"] == "cancelled by operator"

        blocked_id = Jobs.create_job!(srv.app.job_mgr, "allknn", String[])
        Jobs.update_job_state!(srv.app.job_mgr, blocked_id, Jobs.Queued, Jobs.Blocked)
        resp = HTTP.delete("$base_url/jobs/$blocked_id")
        @test resp.status == 200
        @test first(Jobs.get_job(srv.app.job_mgr, blocked_id)) == Jobs.Failed

        running_id = Jobs.create_job!(srv.app.job_mgr, "describe", ["describe", "--dataset", "nope", "--workdir", "/tmp"])
        Jobs.update_job_state!(srv.app.job_mgr, running_id, Jobs.Queued, Jobs.Running)
        resp = try
            HTTP.delete("$base_url/jobs/$running_id")
        catch e
            e.response
        end
        @test resp.status == 409 # running -> use kill instead
        @test first(Jobs.get_job(srv.app.job_mgr, running_id)) == Jobs.Running

        Jobs.finish_job!(srv.app.job_mgr, running_id, true)
        resp = try
            HTTP.delete("$base_url/jobs/$running_id")
        catch e
            e.response
        end
        @test resp.status == 409 # already completed -> nothing to cancel

        resp = try
            HTTP.delete("$base_url/jobs/does-not-exist")
        catch e
            e.response
        end
        @test resp.status == 404
    end
end
