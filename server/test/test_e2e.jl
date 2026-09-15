using Test
using HTTP
using JSON3
using SimilaritySearchServer
using SimilaritySearchServer.Server
using SimilaritySearchEngine.Project
using SimilaritySearchServer.Jobs
using SimilaritySearchServer.Executors

@testset "End-to-End API and Job Execution" begin
    srv = ensure_test_server()
    app = srv.app
    root = srv.root_url

    # Test Ready
    resp = HTTP.get("$root/readyz")
    @test resp.status == 200
    @test JSON3.read(String(resp.body))["status"] == "ok"

    # Test Create Dataset
    req_body = JSON3.write(Dict("id" => "test_dataset_1"))
    resp = HTTP.post("$root/api/v1/datasets", [], req_body)
    @test resp.status == 201
    parsed_resp = JSON3.read(String(resp.body))
    @test parsed_resp["status"] == "created"
    @test parsed_resp["id"] == "test_dataset_1"

    # Check dataset was actually opened in server state
    lock(app.lock) do
        @test haskey(app.handles, "test_dataset_1")
    end

    # Test Submit Job -- the HTTP handler only ever enqueues (queued/); the background
    # dispatcher (Executors.run_dispatcher!, started alongside the server) is what
    # actually moves it through running/ to a terminal state.
    req_body = JSON3.write(Dict("command" => ["build", "--dataset", "test_dataset_1"]))
    resp = HTTP.post("$root/api/v1/jobs/build", [], req_body)
    @test resp.status == 202
    parsed_resp = JSON3.read(String(resp.body))
    @test parsed_resp["status"] == "accepted"
    job_id = parsed_resp["job_id"]
    @test !isempty(job_id)

    # Get Job Status
    resp = HTTP.get("$root/api/v1/jobs/$job_id")
    @test resp.status == 200

    # Poll until the dispatcher moves the job to a terminal state (completed or failed --
    # this "build" subcommand is missing required args, so it's expected to fail, but it
    # must reach a *terminal* state rather than being stuck in queued/running forever).
    # Each poll spawns/loads a fresh Julia subprocess (SimilaritySearch/TextSearch/RocksDB
    # take real time to load even precompiled), so this can legitimately take ~30s.
    terminal = Set(["completed", "failed"])
    status = nothing
    for _ in 1:180
        resp = HTTP.get("$root/api/v1/jobs/$job_id")
        status = JSON3.read(String(resp.body))["status"]
        status in terminal && break
        sleep(0.3)
    end
    @test status in terminal
end
