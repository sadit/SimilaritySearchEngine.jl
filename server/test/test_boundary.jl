# The seam between this package and the engine, tested without starting anything: no HTTP
# server, no subprocess, no dataset on disk. These are the functions that translate between
# the wire's vocabulary and the engine's, and they are what an engine-side change breaks
# first -- which is why they run on every change while the end-to-end suites do not.

using SimilaritySearchEngine
import SimilaritySearchEngine as SSE
using SimilaritySearchServer.Server: parse_index_kind, parse_distance, default_textmodel,
                                     engine_error_response, _typed_item, AppState, _guard,
                                     json_response
using SimilaritySearchServer: cli_exit_code, wire_kind
import SimilaritySearchServer.Tokens as Tokens
import SimilaritySearchServer.Executors as Executors
using SimilaritySearchServer: resolve_batch_threads_pct
using HTTP
using JSON3

@testset "the wire's index kinds map onto engine/backend pairs" begin
    @test parse_index_kind("searchgraph") == (DenseEngine, SearchGraph)
    @test parse_index_kind("exhaustive_search") == (DenseEngine, ExhaustiveSearch)
    @test parse_index_kind("bm25_invfile") == (FullTextEngine, BM25InvertedFile)
    @test parse_index_kind("invfile") == (FullTextEngine, TextInvertedFile)
    @test parse_index_kind("sparse_invfile") == (SparseEngine, InvertedFile)
    # An unknown name is an invalid request, not an opaque failure: the handler turns this
    # very type into a 400.
    @test_throws SSE.UnknownBackend parse_index_kind("nope")

    # Text projects must state a text model; nothing else may carry one.
    @test default_textmodel(FullTextEngine) isa FitFromCorpus
    @test default_textmodel(DenseEngine) === nothing
    @test default_textmodel(SparseEngine) === nothing
end

@testset "engine errors become status codes by category, not by message" begin
    cases = [
        (SSE.ProfileNotInstalled("es", "no profile"), 404),
        (SSE.PendingBacklog(:allknn, 10, 4, "staged backlog"), 409),
        (SSE.NothingStaged("nothing staged"), 409),
        (SSE.PayloadMismatch("wrong item"), 400),
        (SSE.InvalidOption(:dimension, "needs a dimension"), 400),
        (SSE.WrongDimension(8, 4, "wrong dimension"), 400),
        (SSE.CorruptedStorage("unreadable"), 500),
    ]
    for (err, status) in cases
        resp = engine_error_response(err)
        @test resp.status == status
        body = JSON3.read(String(resp.body), Dict{String, Any})
        # The concrete type travels too, for a client that wants to branch without parsing prose
        @test body["kind"] == string(nameof(typeof(err)))
        @test !isempty(body["error"])
    end
end

@testset "engine errors become exit codes by the same categories" begin
    @test cli_exit_code(SSE.ProfileNotInstalled("es", "x")) == 4
    @test cli_exit_code(SSE.PendingBacklog(:fft, 3, 1, "x")) == 5
    @test cli_exit_code(SSE.NoIndex("x")) == 5
    @test cli_exit_code(SSE.PayloadMismatch("x")) == 2
    @test cli_exit_code(SSE.UnsupportedOperation(:calibrate!, "x")) == 2
    @test cli_exit_code(SSE.CorruptedStorage("x")) == 70
end

@testset "a wire item becomes the engine's own item type" begin
    dense = _typed_item(:dense, Dict{String, Any}("vector" => [1.0, 2.0], "doc_id" => "d1",
                                                  "keywords" => ["a"], "year" => 2020))
    @test dense isa DenseItem
    @test dense.doc_id == "d1"
    @test dense.keywords == ["a"]
    # Anything that is not a reserved field is free-form metadata
    @test dense.meta["year"] == 2020

    text = _typed_item(:text, Dict{String, Any}("text" => "hola", "meta" => Dict("lang" => "es")))
    @test text isa TextItem
    @test text.meta["lang"] == "es"
    @test text.doc_id === nothing

    sparse = _typed_item(:sparse, Dict{String, Any}("indices" => [1, 5], "values" => [0.5, 0.25],
                                                    "dimension" => 8))
    @test sparse isa SparseItem

    # An item carrying no payload for this kind of project is skipped, not an error
    @test _typed_item(:dense, Dict{String, Any}("text" => "no vector here")) === nothing
    @test _typed_item(:text, Dict{String, Any}("vector" => [1.0])) === nothing
end


# ---------------------------------------------------------------------------------------
# Authorization (PLAN.md §4.1). No server and no socket: the decision is a pure function of
# the token record and the permission a route asks for, and `_guard` is where it is taken.
# ---------------------------------------------------------------------------------------

@testset "a permission string is an operation and a dataset" begin
    @test Tokens.parse_permission("write:corpus_es") == (:write, "corpus_es")
    @test Tokens.parse_permission("admin:*") == (:admin, "*")
    # A permission that names no dataset applies to every dataset
    @test Tokens.parse_permission("read") == (:read, "*")
    @test Tokens.parse_permission("read:") == (:read, "*")
    # Anything else grants nothing rather than granting something unintended
    @test Tokens.parse_permission("superuser") === nothing
    @test Tokens.parse_permission("") === nothing
    @test Tokens.normalize_permissions(["read", "admin:ds", "nonsense"]) == ["read:*", "admin:ds"]
end

@testset "an operation includes the ones before it, within its own dataset" begin
    rec(perms) = Tokens.TokenRecord("tok", "user", perms, "2026-01-01T00:00:00", nothing)

    reader = rec(["read:*"])
    @test Tokens.has_permission(reader, :read, "anything")
    @test !Tokens.has_permission(reader, :write, "anything")
    @test !Tokens.has_permission(reader, :admin, "anything")

    writer = rec(["write:corpus_es"])
    @test Tokens.has_permission(writer, :write, "corpus_es")
    @test Tokens.has_permission(writer, :read, "corpus_es")   # write includes read
    @test !Tokens.has_permission(writer, :read, "corpus_en")  # ... in its own dataset only
    @test !Tokens.has_permission(writer, :admin, "corpus_es")

    admin = rec(["admin:*"])
    @test all(op -> Tokens.has_permission(admin, op, "whatever"), (:read, :write, :admin))

    # A route that is not about one dataset asks for "*", and only a "*" permission matches
    @test Tokens.has_permission(reader, :read, "*")
    @test !Tokens.has_permission(writer, :read, "*")

    @test !Tokens.has_permission(rec(String[]), :read, "*")
    @test !Tokens.has_permission(rec(["bogus:*"]), :read, "*")
end

@testset "a token expires, and an unreadable expiry counts as expired" begin
    rec(exp) = Tokens.TokenRecord("tok", "user", ["read:*"], "2026-01-01T00:00:00", exp)
    @test !Tokens.is_expired(rec(nothing))
    @test Tokens.is_expired(rec("2000-01-01T00:00:00"))
    @test !Tokens.is_expired(rec("2999-01-01T00:00:00"))
    @test Tokens.is_expired(rec("last tuesday"))
end

@testset "_guard answers 401, 403, or the handler" begin
    workdir = mktempdir()
    mgr = Tokens.open_token_manager(workdir)
    ok = () -> json_response(200, Dict("status" => "ok"))
    app = AppState(workdir, nothing, nothing, nothing, mgr,
                   Dict{String, SSE.EmbeddedEngine}(), ReentrantLock(), true)
    request(token) = token === nothing ? HTTP.Request("GET", "/api/v1/datasets") :
                                         HTTP.Request("GET", "/api/v1/datasets", ["Authorization" => "Bearer $token"])

    reader = Tokens.create_token!(mgr, "reader", ["read:*"])
    writer = Tokens.create_token!(mgr, "writer", ["write:corpus_es"])
    expired = Tokens.create_token!(mgr, "past", ["admin:*"]; expires_at="2000-01-01T00:00:00")

    # No token, an unknown token, and an expired token are all 401: a caller without a valid
    # token learns nothing from the difference between them.
    @test _guard(ok, request(nothing), app, :read, "*").status == 401
    @test _guard(ok, request("not-a-token"), app, :read, "*").status == 401
    @test _guard(ok, request(expired), app, :read, "*").status == 401

    # A valid token without the permission is 403, and the response names what was required,
    # because its holder is a legitimate caller.
    resp = _guard(ok, request(reader), app, :write, "corpus_es")
    @test resp.status == 403
    @test JSON3.read(String(resp.body)).required == "write:corpus_es"

    @test _guard(ok, request(reader), app, :read, "corpus_es").status == 200
    @test _guard(ok, request(writer), app, :write, "corpus_es").status == 200
    @test _guard(ok, request(writer), app, :read, "*").status == 403

    # A token presented without the "Bearer " prefix is accepted, as the header parser allows
    @test _guard(ok, HTTP.Request("GET", "/x", ["Authorization" => reader]), app, :read, "*").status == 200

    # With authentication disabled the handler runs and no token is consulted
    app.auth_enabled[] = false
    @test _guard(ok, request(nothing), app, :read, "*").status == 200

    Tokens.close_token_manager(mgr)
    rm(workdir; recursive=true, force=true)
end

@testset "every /api/v1 route is registered through _guard" begin
    # The authorization table is the route block of `run_server`, so the invariant that matters
    # is structural: a route added without a guard is a route that answers without a token.
    src = read(joinpath(@__DIR__, "..", "src", "server.jl"), String)
    routes = [m.match for m in eachmatch(r"@(get|post|delete|put) \"[^\"]+\"[^\n]*", src)]
    api = filter(r -> occursin("\"/api/v1", r), routes)
    open_routes = filter(r -> !occursin("\"/api/v1", r), routes)

    @test length(api) == 31
    @test all(r -> occursin("_guard(", r), api)
    # Health and metrics stay open on purpose: a supervisor and a metrics collector have no
    # token, and refusing them would report a healthy server as down.
    @test sort([match(r"\"([^\"]+)\"", r).captures[1] for r in open_routes]) ==
          ["/healthz", "/metrics", "/readyz"]
end


# ---------------------------------------------------------------------------------------
# The thread budget of job execution (PLAN.md §2). Arithmetic and a command, no processes.
# ---------------------------------------------------------------------------------------

@testset "the reserved share becomes processes and threads" begin
    # The product never exceeds the share, and a few jobs with several threads each is
    # preferred over many jobs with one: the work inside a job is parallel already.
    @test Executors.job_thread_budget(64, 20) == (4, 3)    # 12 threads of the 12.8 reserved
    @test Executors.job_thread_budget(32, 50) == (4, 4)
    @test Executors.job_thread_budget(16, 25) == (4, 1)
    @test Executors.job_thread_budget(8, 20) == (1, 1)

    # Degenerate shares still execute jobs, one at a time on one thread
    @test Executors.job_thread_budget(1, 20) == (1, 1)
    @test Executors.job_thread_budget(64, 0) == (1, 1)
    @test Executors.job_thread_budget(0, 20) == (1, 1)

    # The whole machine, for a server that is only a job runner
    @test Executors.job_thread_budget(64, 100) == (4, 16)

    for (n, pct) in ((64, 20), (32, 50), (12, 33), (7, 20), (1, 100))
        c, t = Executors.job_thread_budget(n, pct)
        @test c * t <= max(1, floor(Int, n * pct / 100)) || (c, t) == (1, 1)
    end
end

@testset "a job carries its thread budget in the environment" begin
    # `Base.julia_cmd()` does not propagate `--threads`, so a job would otherwise run on one
    # thread whatever the server was given.
    cmd = Executors._job_command(["describe", "--dataset", "x"], 3)
    @test any(e -> e == "JULIA_NUM_THREADS=3", cmd.env)
    @test occursin("similarity-search.jl", join(cmd.exec, " "))
    @test Executors._job_command(String[], 0).env |> env -> any(e -> e == "JULIA_NUM_THREADS=1", env)
end

@testset "the reserved share is read from the configuration" begin
    @test resolve_batch_threads_pct(Dict("batch_threads_pct" => 20)) == 20
    @test resolve_batch_threads_pct(Dict("query_threads_pct" => 80)) == 20
    @test resolve_batch_threads_pct(Dict("query_threads_pct" => 90)) == 10
    # Stated twice and inconsistently: the batch share applies
    @test resolve_batch_threads_pct(Dict("query_threads_pct" => 50, "batch_threads_pct" => 20)) == 20
    # Absent, out of range, or not a number: the default of the generated file
    @test resolve_batch_threads_pct(Dict{String, Any}()) == 20
    @test resolve_batch_threads_pct(Dict("batch_threads_pct" => 150)) == 20
    @test resolve_batch_threads_pct(Dict("batch_threads_pct" => "half")) == 20
end
