# similarity-search-ctl's own CLI argument parsing (PLAN.md §1's admin command surface).
#
# !!! note "Separate ArgParseSettings from build_settings() -- a genuinely different binary"
#     `similarity-search-ctl` talks HTTP to a *running* `similarity-search-serve`, never
#     touches a local workdir directly (PLAN.md §1's "admin is a control-plane client, not
#     a second process touching the same RocksDB files" note) -- its command surface
#     (list/stats/jobs */add-token/del-token) has nothing in common with `similarity-search`'s
#     data-operation commands (build/searchbatch/.../load), so it gets its own settings
#     object rather than being force-fit into `build_settings()`'s subcommand list.
#
# !!! note "Scoped to what already has a real HTTP endpoint behind it"
#     PLAN.md's full admin surface (§1) is `list | log | stats | dump | load | add-token |
#     del-token | prune-tokens | log-tokens | jobs queue|result|block|resume|kill|gc`. This
#     file now covers every one of those, including `log` (chunk 17) -- `Telemetry.log_operation`
#     is wired into `search`/`ftsearch`/`append`/`delete` (`server.jl`'s `_run_search`/
#     `handle_append`/`handle_delete_item`) and `GET /api/v1/datasets/{id}/log` proxies the
#     resulting `op_log` column family, so `log` finally has real content to show instead of
#     an always-empty CF. `dump`/`load` (chunk 14's `POST /api/v1/jobs/dump|load`) and
#     `prune-tokens`/`log-tokens` (`POST /api/v1/admin/tokens/prune` / `GET
#     /api/v1/admin/tokens`, chunk 15) each joined the same day their HTTP prerequisite did.
#
# !!! note "`interactive` (chunk 16): `-ctl` gets a real guided form too"
#     `interactive.jl`'s `run_interactive_ctl` introspects this same settings object --
#     the second (and, per `jobs`'s own nesting, structurally harder) half of PLAN.md's
#     Interactive Mode section, now that this file gives it a real command surface to
#     introspect at all (its own prerequisite, chunks 12/15).

"""
    build_ctl_settings() -> ArgParse.ArgParseSettings

The `similarity-search-ctl` command line: the control-plane subcommands that talk to a
running server over HTTP (`list`, `stats`, `log`, the token commands, and the `jobs`
subtree). The counterpart of [`build_settings`](@ref), which is the data plane against a
working directory, and introspected by `interactive.jl` the same way.
"""
function build_ctl_settings()
    s = ArgParseSettings(description = "SimilaritySearchServer admin/control CLI")

    @add_arg_table! s begin
        "--host"
            help = "Server host"
            default = "127.0.0.1"
        "--port"
            help = "Server port"
            arg_type = Int
            default = 8080
        "list"
            help = "List datasets known to the running server"
            action = :command
        "stats"
            help = "Show a dataset's details/statistics (doc count, tombstone ratio, calibrated beamsearch baseline)"
            action = :command
        "log"
            help = "Show a dataset's op_log (usage telemetry: search/append/ftsearch/delete cost + timing)"
            action = :command
        "jobs"
            help = "Job lifecycle administration"
            action = :command
        "add-token"
            help = "Create a new access token"
            action = :command
        "del-token"
            help = "Revoke an access token"
            action = :command
        "prune-tokens"
            help = "Revoke every token whose expires_at has already passed"
            action = :command
        "log-tokens"
            help = "List every access token (audit listing)"
            action = :command
        "dump"
            help = "Export a dataset as a portable bundle (submits a dump job and waits for it)"
            action = :command
        "load"
            help = "Import a dataset from a bundle produced by dump (submits a load job and waits for it)"
            action = :command
        "interactive"
            help = "Guided interactive mode (menu-driven command builder)"
            action = :command
    end

    @add_arg_table! s["stats"] begin
        "--dataset"
            help = "Dataset ID/name"
            required = true
    end

    @add_arg_table! s["log"] begin
        "--dataset"
            help = "Dataset ID/name"
            required = true
        "--offset"
            help = "Skip this many entries (newest first)"
            arg_type = Int
            default = 0
        "--limit"
            help = "Maximum entries to return"
            default = nothing
    end

    @add_arg_table! s["jobs"] begin
        "queue"
            help = "List queued/running/blocked jobs (and, informationally, currently-open result cursors)"
            action = :command
        "result"
            help = "Fetch a completed job's result payload"
            action = :command
        "block"
            help = "Pause a queued job so the scheduler skips it"
            action = :command
        "resume"
            help = "Resume a blocked job back into the queue"
            action = :command
        "kill"
            help = "Forcibly terminate a running job"
            action = :command
        "gc"
            help = "Garbage-collect finished jobs and expired result cursors past retention"
            action = :command
    end

    @add_arg_table! s["jobs"]["queue"] begin
        "--status"
            help = "Filter by status (queued|running|blocked|completed|failed)"
            default = nothing
        "--kind"
            help = "Filter by job kind"
            default = nothing
    end

    @add_arg_table! s["jobs"]["result"] begin
        "--job-id"
            required = true
        "--output"
            help = "Write the raw result payload here instead of printing it"
            default = nothing
    end

    @add_arg_table! s["jobs"]["block"] begin
        "--job-id"
            required = true
    end

    @add_arg_table! s["jobs"]["resume"] begin
        "--job-id"
            required = true
    end

    @add_arg_table! s["jobs"]["kill"] begin
        "--job-id"
            required = true
    end

    @add_arg_table! s["jobs"]["gc"] begin
        "--retention-seconds"
            help = "Delete completed/failed job records older than this many seconds"
            arg_type = Int
            default = 86400
    end

    @add_arg_table! s["add-token"] begin
        "--user"
            default = "anonymous"
        "--permissions"
            help = "Comma-separated permission list"
            default = ""
        "--expires-at"
            help = "RFC3339 expiry timestamp (omit for a non-expiring token)"
            default = nothing
    end

    @add_arg_table! s["del-token"] begin
        "--token"
            required = true
    end

    @add_arg_table! s["dump"] begin
        "--dataset"
            help = "Source dataset ID/name to export"
            required = true
    end

    @add_arg_table! s["load"] begin
        "--bundle"
            help = "Bundle directory previously produced by dump (path on the server's own filesystem)"
            required = true
        "--dataset"
            help = "Target dataset ID/name (must not already exist on the server)"
            required = true
    end

    return s
end

"""
    parse_ctl_commandline(args::Vector{String}=ARGS) -> Dict

Parses `args` against a freshly built [`build_ctl_settings`](@ref) object -- mirrors
`cli.jl`'s `build_settings`/`parse_commandline` split, kept separate here since `-ctl`'s
settings are a genuinely different object.
"""
function parse_ctl_commandline(args::Vector{String}=ARGS)
    return parse_args(args, build_ctl_settings())
end
