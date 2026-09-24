# The seam between this package and the engine, tested without starting anything: no HTTP
# server, no subprocess, no dataset on disk. These are the functions that translate between
# the wire's vocabulary and the engine's, and they are what an engine-side change breaks
# first -- which is why they run on every change while the end-to-end suites do not.

using SimilaritySearchEngine
import SimilaritySearchEngine as SSE
using SimilaritySearchServer.Server: parse_index_kind, parse_distance, default_textmodel,
                                     engine_error_response, _typed_item, AppState, _guard,
                                     json_response, parse_meta_schema, serialize_meta_schema,
                                     _filter_predicate, query_slot_count, _with_query_slot,
                                     token_fingerprint, inprocess_batch_cap
using SimilaritySearchServer: cli_exit_code, wire_kind
import SimilaritySearchServer.Tokens as Tokens
import SimilaritySearchServer.Executors as Executors
import SimilaritySearchServer.Telemetry as Telemetry
import SimilaritySearchServer.Server as Server
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

@testset "a request body is parsed as JSON, never opened as a path" begin
    # Given a `String` shorter than 255 bytes, `JSON3.read` asks `isfile` of it first and parses
    # the file when one exists. A body that names a JSON file in the working directory would then
    # be answered with that file's contents: here, a permission decided by a file instead of by
    # the request, and a dataset created from a file.
    workdir = mktempdir()
    app = AppState(workdir, nothing, nothing, nothing, Tokens.open_token_manager(workdir),
                   Dict{String, SSE.EmbeddedEngine}(), ReentrantLock(), false)
    cd(mktempdir()) do
        write("req.json", JSON3.write(Dict("dataset" => "corpus_es", "id" => "from_file",
                                           "index_type" => "searchgraph")))
        req = HTTP.Request("POST", "/api/v1/datasets", [], "req.json")
        # Not JSON, so no dataset is named, and the permission required is the one over all.
        @test Server._body_dataset(req, "dataset") == "*"
        @test_throws ArgumentError Server.handle_create_dataset(req, app)
        @test !isdir(joinpath(workdir, "datasets", "from_file"))
    end
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

    @test length(api) == 32
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


# ---------------------------------------------------------------------------------------
# The counters of /metrics (PLAN.md §5.8), without a server: they are an aggregate of the
# same values that go into the operation log.
# ---------------------------------------------------------------------------------------

@testset "request counters accumulate what the operation log records" begin
    reg = Telemetry.MetricsRegistry()
    Telemetry.record_metric!(reg, "ds", "search", 0.004; distance_evaluations=120)
    Telemetry.record_metric!(reg, "ds", "search", 0.9; distance_evaluations=300)
    Telemetry.record_metric!(reg, "ds", "append", 0.2; items_inserted=5)
    Telemetry.record_metric!(reg, "other", "search", 0.002)

    @test reg.requests[("ds", "search")] == 2
    @test reg.requests[("ds", "append")] == 1
    @test reg.evaluations["ds"] == 420
    @test reg.items_inserted["ds"] == 5
    @test reg.duration_sum[("ds", "search")] ≈ 0.904
    # A dataset that only served searches reports no inserted items rather than zero
    @test !haskey(reg.items_inserted, "other")
    # A search with no distance computations (an empty index) adds no counter entry
    @test !haskey(reg.evaluations, "other")

    # Histogram buckets are cumulative: each one counts every request at most that slow
    buckets = reg.duration_buckets[("ds", "search")]
    @test buckets == [0, 1, 1, 1, 1, 1, 2, 2]
    @test issorted(buckets)

    # No registry, no record: the command lines log operations without reporting metrics
    @test Telemetry.record_metric!(nothing, "ds", "search", 1.0) === nothing
end

@testset "the counters render as a Prometheus exposition" begin
    reg = Telemetry.MetricsRegistry()
    Telemetry.record_metric!(reg, "ds", "search", 0.004; distance_evaluations=120)
    text = join(Telemetry.prometheus_lines(reg), "\n")

    @test occursin("# TYPE simsearch_requests_total counter", text)
    @test occursin("simsearch_requests_total{dataset=\"ds\",operation=\"search\"} 1", text)
    @test occursin("simsearch_distance_evaluations_total{dataset=\"ds\"} 120", text)
    @test occursin("# TYPE simsearch_request_duration_seconds histogram", text)
    @test occursin("simsearch_request_duration_seconds_count{dataset=\"ds\",operation=\"search\"} 1", text)
    @test occursin("le=\"+Inf\"", text)
    # A restart is visible to a collector, which is what a counter that starts at zero needs
    @test occursin("simsearch_process_start_time_seconds", text)

    # Every sample line carries a metric name and a value, and no line is left half-formatted
    for line in Telemetry.prometheus_lines(reg)
        startswith(line, "#") && continue
        @test occursin(r"^simsearch_[a-z_]+(\{[^}]*\})? -?[0-9.e+]+$", line)
    end

    # A dataset id with a quote in it cannot break the exposition
    reg2 = Telemetry.MetricsRegistry()
    Telemetry.record_metric!(reg2, "we\"ird", "search", 0.1)
    @test occursin("dataset=\"we\\\"ird\"", join(Telemetry.prometheus_lines(reg2), "\n"))

    # Nothing recorded: the declarations stand and there are no samples
    empty_text = join(Telemetry.prometheus_lines(Telemetry.MetricsRegistry()), "\n")
    @test occursin("# TYPE simsearch_requests_total counter", empty_text)
    @test !occursin("simsearch_requests_total{", empty_text)
end


# ---------------------------------------------------------------------------------------
# The declared meta schema (PLAN.md §4.5) crossing the wire: the engine owns the type, this
# package only translates the request into it and reports it back.
# ---------------------------------------------------------------------------------------

@testset "a meta schema crosses the wire as the engine's own declaration" begin
    declared = parse_meta_schema(Dict{String,Any}("meta_schema" => [
        Dict("name" => "year", "type" => "int64"),
        Dict("name" => "lang", "type" => "string"),
        Dict("name" => "when", "type" => "timestamp"),
    ]))
    @test declared isa SSE.MetaSchema
    @test [f.name for f in declared.fields] == ["year", "lang", "when"]
    @test [f.type for f in declared.fields] == [:int64, :string, :timestamp]

    # A dataset that declares nothing gets the empty declaration, which is what every one
    # of them had before this existed
    @test isempty(parse_meta_schema(Dict{String,Any}()))

    # Malformed declarations are invalid requests, which the handler answers as 400
    @test_throws SSE.InvalidOption parse_meta_schema(Dict{String,Any}("meta_schema" => "year:int64"))
    @test_throws SSE.InvalidOption parse_meta_schema(Dict{String,Any}("meta_schema" => [Dict("name" => "year")]))
    @test_throws SSE.InvalidOption parse_meta_schema(Dict{String,Any}("meta_schema" => [Dict("name" => "year", "type" => "int32")]))
    @test engine_error_response(SSE.InvalidOption(:meta_schema, "x")).status == 400

    # And it reports back in the shape it arrived in
    @test serialize_meta_schema(declared) == [
        Dict("name" => "year", "type" => "int64"),
        Dict("name" => "lang", "type" => "string"),
        Dict("name" => "when", "type" => "timestamp"),
    ]
    @test serialize_meta_schema(SSE.MetaSchema()) == []
end

@testset "a filter compares in the declared type" begin
    record = SSE.Schema.MetadataRecord(Int32(1), 1, "d1", String[], String[])
    meta = Dict("when" => "2026-01-02T00:00:00")
    declared = parse_meta_schema(Dict{String,Any}("meta_schema" => [Dict("name" => "when", "type" => "timestamp")]))

    # One instant, two spellings: equal as instants, different as text
    typed = _filter_predicate(Dict("when" => "2026-01-02T00:00"), declared)
    untyped = _filter_predicate(Dict("when" => "2026-01-02T00:00"), SSE.MetaSchema())
    @test typed(record, meta)
    @test !untyped(record, meta)

    ranged = _filter_predicate(Dict("when" => Dict("gte" => "2026-01-01T00:00:00")), declared)
    @test ranged(record, meta)
end


# ---------------------------------------------------------------------------------------
# The bound on concurrent searches (PLAN.md §2). No server: the bound is a semaphore and a
# derivation, and both can be exercised directly.
# ---------------------------------------------------------------------------------------

@testset "how many searches run at once" begin
    # Derived from the split: the threads that job execution did not reserve
    @test query_slot_count(64, 20, 0) == 52
    @test query_slot_count(8, 25, 0) == 6
    @test query_slot_count(1, 20, 0) == 1
    # A share that would leave nothing still answers, one search at a time
    @test query_slot_count(2, 100, 0) == 1
    @test query_slot_count(0, 20, 0) == 1
    # A configured value is used as given, whatever the split says
    @test query_slot_count(64, 20, 4) == 4
    @test query_slot_count(2, 20, 100) == 100
end

@testset "in-process batch work is capped at the reserved share" begin
    # `@BATCHES` dispatches one task per batch, so a cap below the thread count leaves the
    # remaining threads for queries. In a job this is not needed: that subprocess starts with
    # JULIA_NUM_THREADS already set to its share.
    @test inprocess_batch_cap(64, 20) == 12
    @test inprocess_batch_cap(8, 25) == 2
    @test inprocess_batch_cap(64, 100) == 64
    # A share that rounds to nothing still indexes, one batch at a time
    @test inprocess_batch_cap(2, 20) == 1
    @test inprocess_batch_cap(64, 0) == 1
    @test inprocess_batch_cap(0, 20) == 1
end

@testset "a search waits for a slot instead of joining an unbounded crowd" begin
    workdir = mktempdir()
    mgr = Tokens.open_token_manager(workdir)
    app = AppState(workdir, nothing, nothing, nothing, mgr,
                   Dict{String, SSE.EmbeddedEngine}(), ReentrantLock(), false, 3)

    running = Threads.Atomic{Int}(0)
    peak = Threads.Atomic{Int}(0)
    done = Threads.Atomic{Int}(0)
    @sync for _ in 1:24
        Threads.@spawn _with_query_slot(app) do
            n = Threads.atomic_add!(running, 1) + 1
            Threads.atomic_max!(peak, n)
            sleep(0.01)                      # long enough for the others to pile up
            Threads.atomic_sub!(running, 1)
            Threads.atomic_add!(done, 1)
        end
    end

    @test done[] == 24                        # every request ran: waiting, not refused
    @test peak[] <= 3                         # ... and never more than the bound at once
    @test app.queries_running[] == 0          # the gauges return to zero
    @test app.queries_waiting[] == 0
    # Every request asked for a slot, and the ones that waited are visible as time
    @test app.metrics.waits[] == 24
    @test app.metrics.wait_seconds[] > 0

    # A slot is released even when the body fails, or one failure would shrink the server
    @test_throws ErrorException _with_query_slot(() -> error("boom"), app)
    @test app.queries_running[] == 0
    @test _with_query_slot(() -> :ok, app) === :ok

    Tokens.close_token_manager(mgr)
    rm(workdir; recursive=true, force=true)
end
