using Test
using HTTP
using JSON3

@testset "5. Dataset Administration and Health Tests" begin
    with_test_server() do base_url, workdir
        root_url = replace(base_url, "/api/v1" => "")

        # 1. Health/readiness/metrics don't live under /api/v1 (PLAN.md §5.8).
        @test HTTP.get("$root_url/healthz").status == 200
        @test HTTP.get("$root_url/readyz").status == 200

        metrics_resp = HTTP.get("$root_url/metrics")
        @test metrics_resp.status == 200
        metrics_body = String(metrics_resp.body)
        @test occursin("simsearch_active_jobs", metrics_body)
        @test occursin("simsearch_threads", metrics_body)

        # 2. Create + populate a dataset to exercise the admin endpoints against.
        resp = HTTP.post("$base_url/datasets", [], JSON3.write(Dict("id" => "admin_test_ds", "index_type" => "searchgraph", "distance" => "L2")))
        @test resp.status == 201

        data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")
        docs = [JSON3.read(line) for line in readlines(data_path)[1:5]]
        resp = HTTP.post("$base_url/datasets/admin_test_ds/append", [], JSON3.write(Dict("items" => docs)))
        @test resp.status == 200

        resp = HTTP.post("$base_url/datasets/admin_test_ds/delete", [], JSON3.write(Dict("_id" => 1)))
        @test resp.status == 200

        # 2b. /metrics now carries the counters the operation log feeds (PLAN.md §5.8): the
        # append and the delete above are in it, with their dataset and their operation.
        metrics_body = String(HTTP.get("$root_url/metrics").body)
        @test occursin("simsearch_requests_total{dataset=\"admin_test_ds\",operation=\"append\"} 1", metrics_body)
        @test occursin("simsearch_requests_total{dataset=\"admin_test_ds\",operation=\"delete\"} 1", metrics_body)
        @test occursin("simsearch_items_inserted_total{dataset=\"admin_test_ds\"} 5", metrics_body)
        @test occursin("simsearch_request_duration_seconds_count{dataset=\"admin_test_ds\",operation=\"append\"} 1", metrics_body)

        # 3. GET /api/v1/datasets lists it with live stats merged in.
        resp = HTTP.get("$base_url/datasets")
        @test resp.status == 200
        listed = JSON3.read(String(resp.body)).datasets
        entry = only(filter(d -> d.id == "admin_test_ds", listed))
        @test entry.index_kind == "searchgraph"
        @test entry.distance == "L2"
        @test entry.loaded == true
        @test entry.doc_count == 5
        @test entry.tombstone_count == 1
        @test entry.tombstone_ratio ≈ 0.2

        # 4. GET /api/v1/datasets/{id} single-dataset detail matches the list entry.
        resp = HTTP.get("$base_url/datasets/admin_test_ds")
        @test resp.status == 200
        detail = JSON3.read(String(resp.body))
        @test detail.doc_count == 5
        @test detail.tombstone_ratio ≈ 0.2
        @test detail.join_group === nothing

        # 5. Unknown dataset -> 404; path-unsafe id -> 400 (PLAN.md §4.6's path-safety warning).
        resp = try
            HTTP.get("$base_url/datasets/does_not_exist")
        catch e
            e.response
        end
        @test resp.status == 404

        resp = try
            HTTP.post("$base_url/datasets", [], JSON3.write(Dict("id" => "../escape")))
        catch e
            e.response
        end
        @test resp.status == 400

        # 6. /exists reports existence + tombstone status without a full metadata fetch.
        resp = HTTP.get("$base_url/datasets/admin_test_ds/exists?ids=1,2,999,frankenstein_2")
        @test resp.status == 200
        results = JSON3.read(String(resp.body)).results
        by_id = Dict(r.id => r for r in results)
        @test by_id["1"].exists == true && by_id["1"].deleted == true
        @test by_id["2"].exists == true && by_id["2"].deleted == false
        @test by_id["999"].exists == false
        @test by_id["frankenstein_2"].exists == true && by_id["frankenstein_2"].deleted == false

        # 7. GET .../log (PLAN.md §1's admin `log` command, backs `similarity-search-ctl
        # log`): Telemetry.log_operation is now wired into search/append/ftsearch/delete,
        # so this dataset's append+delete above (plus a search here) already left real
        # op_log entries -- newest first, real distance-evaluation/timing/token fields.
        resp = HTTP.post("$base_url/datasets/admin_test_ds/search", ["Authorization" => "Bearer admin-log-test-token"], JSON3.write(Dict("vector" => docs[1].vector, "k" => 3)))
        @test resp.status == 200

        resp = HTTP.get("$base_url/datasets/admin_test_ds/log")
        @test resp.status == 200
        log_body = JSON3.read(String(resp.body))
        @test log_body.total == 3
        @test [e.operation for e in log_body.entries] == ["search", "delete", "append"] # newest first

        search_entry = log_body.entries[1]
        @test search_entry.details.index_uuid == "admin_test_ds"
        # The log identifies the caller without carrying what they presented: the token of
        # that request appears as its fingerprint, and never as itself.
        @test search_entry.details.token_fingerprint == "484fe328e19dd3b7"
        @test !occursin("admin-log-test-token", String(resp.body))
        # Authentication is disabled in this suite, so the token resolves to no user
        @test search_entry.details.user === nothing
        @test search_entry.details.distance_name !== nothing
        @test search_entry.details.dimension == length(docs[1].vector)
        @test search_entry.details.distance_evaluations > 0
        @test search_entry.details.elapsed_seconds >= 0

        append_entry = log_body.entries[3]
        @test append_entry.details.items_inserted == 5
        @test append_entry.details.token_fingerprint === nothing # no Authorization header on that request
        @test append_entry.details.user === nothing

        # The same search is in /metrics, with the distance computations it performed: the
        # figure §5.8 asked for, which the engine reports through SearchStats.
        metrics_body = String(HTTP.get("$root_url/metrics").body)
        @test occursin("simsearch_requests_total{dataset=\"admin_test_ds\",operation=\"search\"} 1", metrics_body)
        evals = match(r"simsearch_distance_evaluations_total\{dataset=\"admin_test_ds\"\} (\d+)", metrics_body)
        @test evals !== nothing && parse(Int, evals.captures[1]) > 0
        @test occursin("simsearch_request_duration_seconds_count{dataset=\"admin_test_ds\",operation=\"search\"} 1", metrics_body)

        resp = HTTP.get("$base_url/datasets/admin_test_ds/log?offset=1&limit=1")
        @test resp.status == 200
        page = JSON3.read(String(resp.body))
        @test length(page.entries) == 1
        @test page.entries[1].operation == "delete"

        resp = try
            HTTP.get("$base_url/datasets/does_not_exist/log")
        catch e
            e.response
        end
        @test resp.status == 404
    end
end
