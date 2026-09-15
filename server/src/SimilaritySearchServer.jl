module SimilaritySearchServer

using ArgParse
using TOML
using HTTP
using JSON3
using SimilaritySearch
using TextSearch
using RocksDB
using SimilaritySearchEngine
# The engine's public surface, qualified: `search`/`index!`/`append_items!` are exported by
# SimilaritySearch too, so a bare import would make each of them ambiguous here, and the
# qualification keeps the boundary legible at every call site.
import SimilaritySearchEngine as SSE

# Schema/Persistence/Dataset/IndexEngine used to be defined locally here (include'd from
# schema.jl/persistence.jl/dataset.jl/index_engine.jl) -- extracted verbatim into the new
# SimilaritySearchEngine.jl package (PLAN.md §8.5, chunk 22), since none of the four ever
# depended on anything else in this package (Jobs/Cursors/Tokens/Telemetry/Executors/Server
# are HTTP/CLI-specific glue that stayed behind). The bare `using SimilaritySearchEngine.X`
# form below reproduces the exact same bare-name availability the old `include`d-then-
# nested-`module` definitions gave -- both the plain module-name binding every nested
# `Server`/`Jobs`/`Cursors`/`Telemetry` module's own `using ..Project`-style relative
# import needs, and the unqualified `ProjectManager`/`AbstractSearchEngine` availability
# this file's own `run_serve` uses. Placed *before* every `include` below (not just before
# the ones that need it) since a nested module's own top-level `using ..Project` resolves
# at `include`-time, not at first-use time -- confirmed by a clean `using
# SimilaritySearchServer` compile check right after this swap.
using SimilaritySearchEngine.Schema
using SimilaritySearchEngine.Persistence
using SimilaritySearchEngine.Project
using SimilaritySearchEngine.IndexEngine

include("config.jl")
include("cli.jl")
include("cli_handlers.jl")
include("interactive.jl")
include("ctl.jl")
include("ctl_handlers.jl")
include("telemetry.jl")
include("tokens.jl")
include("jobs.jl")
include("cursors.jl")
include("executors.jl")
include("server.jl")

using .Telemetry
using .Tokens
using .Server
using .Jobs
using .Cursors
using .Executors

"""
    resolve_batch_threads_pct(resources_cfg::AbstractDict) -> Float64

The share of threads reserved for job execution, from the `[resources]` section of the
configuration file (PLAN.md §2). `batch_threads_pct` states it directly;
`query_threads_pct` states its complement, and is used when the first is absent. When both
are present and they do not add up to 100, `batch_threads_pct` is the one that applies, and
the discrepancy is reported rather than resolved silently.

Out-of-range or non-numeric values fall back to 20, which is the default of the generated
configuration file.
"""
function resolve_batch_threads_pct(resources_cfg::AbstractDict)
    as_pct(v) = v isa Real && 0 <= v <= 100 ? Float64(v) : nothing
    batch = as_pct(get(resources_cfg, "batch_threads_pct", nothing))
    query = as_pct(get(resources_cfg, "query_threads_pct", nothing))

    if batch !== nothing && query !== nothing && batch + query != 100
        println(stderr, "Warning: [resources] query_threads_pct ($query) and batch_threads_pct ($batch) do not add up to 100; using batch_threads_pct.")
    end
    batch !== nothing && return batch
    query !== nothing && return 100 - query
    return 20.0
end

"""
    run_serve(host::String, port::Int, workdir::String) -> Cint

The actual "start serving" body -- extracted out of `main`'s `serve` branch so both the
plain non-interactive CLI path and `run_interactive_serve`'s confirm-then-run step
(`interactive.jl`, PLAN.md §1) call the exact same function, the same "one command-name ->
handler mapping regardless of how the arguments were collected" principle
`dispatch_command`/`dispatch_ctl_command` already apply to their own binaries.
"""
function run_serve(host::String, port::Int, workdir::String;
                   auth_enabled::Bool=false, batch_threads_pct::Real=20,
                   max_concurrent_queries::Int=0)::Cint
    job_mgr = Jobs.init_job_manager(workdir)
    cursor_mgr = Cursors.init_cursor_manager(workdir)
    executor = Executors.LocalCLIExecutor()
    token_mgr = Tokens.open_token_manager(workdir)

    # Refused here rather than at the first request: with authentication enabled and no token
    # defined, every endpoint would answer 401 and the server would serve nothing.
    if auth_enabled && !Tokens.any_token_exists(token_mgr)
        println(stderr, "Error: [auth] enabled is true and this workdir holds no token, so no request could be answered.")
        println(stderr, "Create the first one against the workdir, then start the server again:")
        println(stderr, "    similarity-search add-token --user admin --permissions \"admin:*\" --workdir ", workdir)
        println(stderr, "(`similarity-search-ctl add-token` talks to a running server, so it cannot create this one.)")
        Tokens.close_token_manager(token_mgr)
        return 1
    end
    auth_enabled || println("Authentication is disabled: every /api/v1 endpoint answers without a token ([auth] enabled in config.toml).")

    app = Server.AppState(
        workdir,
        job_mgr,
        cursor_mgr,
        executor,
        token_mgr,
        Dict{String, SimilaritySearchEngine.EmbeddedEngine}(),
        ReentrantLock(),
        auth_enabled,
        Server.query_slot_count(Threads.nthreads(), batch_threads_pct, max_concurrent_queries)
    )

    # Reopen every dataset already on disk (from a previous run of this same server)
    # before accepting any requests -- otherwise a restarted server could list/describe
    # existing datasets but not search/append/ftsearch/calibrate them until recreated.
    reloaded = Server.reload_datasets!(app)
    isempty(reloaded) || println("Reloaded $(length(reloaded)) existing dataset(s) from $workdir: ", reloaded)

    # Crash recovery: anything left in running/ from a previous serve process
    # (killed mid-dispatch) gets requeued before the dispatcher starts (PLAN.md §5.5).
    requeued = Jobs.requeue_stale_running!(job_mgr)
    isempty(requeued) || println("Requeued $(length(requeued)) job(s) orphaned by a previous run: ", requeued)

    # What job execution may use, from the share reserved for it (PLAN.md §2). A job runs as
    # its own process, so the share is spent as a number of processes and a number of threads
    # each; `Base.julia_cmd()` does not carry this process's `--threads`, so without this every
    # job would run on one thread whatever the machine has.
    slots = Server.query_slot_count(Threads.nthreads(), batch_threads_pct, max_concurrent_queries)
    println("Queries: at most $slots at once; a request that finds every slot taken waits for one.")
    max_concurrent, threads_per_job = Executors.job_thread_budget(Threads.nthreads(), batch_threads_pct)
    println("Job execution: at most $max_concurrent concurrent job(s), $threads_per_job thread(s) each ",
            "($(batch_threads_pct)% of this server's $(Threads.nthreads()) thread(s)). Queries use the rest.")
    @async Executors.run_dispatcher!(job_mgr, executor; max_concurrent, threads_per_job)

    Server.run_server(host, port, app)
    return 0
end

"""
    main(args::Vector{String}) -> Cint

Entry point for `similarity-search` (PLAN.md §1's data-operations CLI): the eleven
`build`-through-`load` subcommands plus its own `interactive` guided form. Only ever
reached for this one binary -- `similarity-search-ctl.jl`/`similarity-search-server.jl`
call `main_ctl`/`main_serve` directly instead of this function (each wrapper statically
knows which binary it is, so there's no env-var sniffing needed to pick an entry point,
unlike the `SIMSEARCH_CLI_BIN`-based dispatch a previous chunk used here). `serve` used to
still be reachable through this same parser/function too, sharing it with the data-ops
binary; it now has a genuinely separate `ArgParseSettings` (`build_serve_settings`,
`cli.jl`) and its own `main_serve` below, matching PLAN.md's "thin, single-purpose serve
binary" framing in full, not just for its own fields (chunk 21).
"""
function main(args::Vector{String})::Cint
    parsed_args = parse_commandline(args)

    if parsed_args["%COMMAND%"] == "interactive"
        return run_interactive()
    elseif parsed_args["%COMMAND%"] !== nothing
        return dispatch_command(parsed_args["%COMMAND%"], parsed_args[parsed_args["%COMMAND%"]])
    else
        println("No command specified.")
    end

    return 0
end

"""
    main_ctl(args::Vector{String}) -> Cint

Entry point for `similarity-search-ctl` (PLAN.md §1's admin/control CLI): parses against
`build_ctl_settings` (a completely separate `ArgParseSettings` from the data-operations
one -- see `ctl.jl`'s header note) and dispatches through `dispatch_ctl_command`, except
for `interactive` (chunk 16), which routes to `run_interactive_ctl` instead. No
`--config`/workdir loading at all -- every `-ctl` command is a plain HTTP call to
`--host`/`--port`, nothing else.
"""
function main_ctl(args::Vector{String})::Cint
    parsed = parse_ctl_commandline(args)
    base_url = "http://$(parsed["host"]):$(parsed["port"])/api/v1"
    parsed["%COMMAND%"] == "interactive" && return run_interactive_ctl(base_url)
    return dispatch_ctl_command(parsed, base_url)
end

"""
    main_serve(args::Vector{String}) -> Cint

Entry point for `similarity-search-server` (PLAN.md §1): parses against its own
`build_serve_settings` (chunk 21 -- genuinely separate from the data-operations
`build_settings`, not shared) and handles exactly its two commands, `serve` and
`interactive` -- nothing else is reachable through this binary at all.

`serve`'s `--host`/`--port`/`--workdir` (all optional, `cli.jl`) layer over `--config`'s
TOML file over a hardcoded literal, in that priority order -- fixes a real, previously
unnoticed bug found in chunk 18: the original code read `get(config, "workdir", "data")`, a
top-level key `generate_default_config` never actually writes (it's nested under
`[paths]`), so a `--config` file's documented `workdir` setting was silently never applied;
`serve` always fell back to the hardcoded `"data"` regardless of what the TOML said.
"""
function main_serve(args::Vector{String})::Cint
    parsed_args = parse_serve_commandline(args)
    config_path = parsed_args["config"]

    if parsed_args["%COMMAND%"] == "interactive"
        return run_interactive_serve(config_path)
    end

    config = load_config(config_path)
    println("Starting SimilaritySearchServer with config from ", config_path)

    server_cfg = get(config, "server", Dict{String, Any}())
    paths_cfg = get(config, "paths", Dict{String, Any}())
    serve_args = parsed_args["serve"]

    auth_cfg = get(config, "auth", Dict{String, Any}())
    resources_cfg = get(config, "resources", Dict{String, Any}())

    host = something(get(serve_args, "host", nothing), get(server_cfg, "host", "127.0.0.1"))
    port = something(get(serve_args, "port", nothing), get(server_cfg, "port", 8080))
    workdir = something(get(serve_args, "workdir", nothing), get(paths_cfg, "workdir", "data"))
    auth_enabled = get(auth_cfg, "enabled", false) === true
    batch_threads_pct = resolve_batch_threads_pct(resources_cfg)
    configured_slots = get(resources_cfg, "max_concurrent_queries", 0)
    max_concurrent_queries = configured_slots isa Integer ? Int(configured_slots) : 0

    return run_serve(host, port, workdir; auth_enabled, batch_threads_pct, max_concurrent_queries)
end

end # module
