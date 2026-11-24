# similarity-search-ctl command implementations -- thin HTTP calls against a running
# similarity-search-serve instance (CHANGELOG.md §1: "admin is a control-plane client, not a
# second process touching the same RocksDB files"). Every function here takes the base
# API URL plus that subcommand's own parsed args, makes one HTTP call, and prints the
# response -- there is deliberately no business logic duplicated from `server.jl` here.

"""
    _ctl_request(method::Symbol, url::String; body=nothing) -> Union{HTTP.Response, Nothing}

Shared error handling for every `-ctl` command: a non-2xx response prints the server's own
JSON error body and returns `nothing` (so the caller can just check for that and return 1);
a connection failure prints CHANGELOG.md §1's own "server not reachable" wording rather than a
raw stacktrace, also returning `nothing`.
"""
function _ctl_request(method::Symbol, url::String; body=nothing)
    try
        if method === :get
            return HTTP.get(url; retry=false)
        elseif method === :post
            headers = ["Content-Type" => "application/json"]
            return HTTP.post(url, headers, body === nothing ? "" : JSON.json(body); retry=false)
        elseif method === :delete
            return HTTP.delete(url; retry=false)
        else
            error("_ctl_request: unsupported method $method")
        end
    catch e
        if e isa HTTP.Exceptions.StatusError
            println("Error: HTTP $(e.status) -- $(String(e.response.body))")
        elseif e isa HTTP.Exceptions.ConnectError
            println("Error: server not reachable at $url")
        else
            rethrow()
        end
        return nothing
    end
end

"""
    ctl_list(base_url::String) -> Int

`similarity-search-ctl list` -> `GET /api/v1/datasets`.
"""
function ctl_list(base_url::String)
    resp = _ctl_request(:get, "$base_url/datasets")
    resp === nothing && return 1
    println(String(resp.body))
    return 0
end

"""
    ctl_stats(base_url::String, cmd_args::Dict) -> Int

`similarity-search-ctl stats --dataset <id>` -> `GET /api/v1/datasets/{id}`. CHANGELOG.md's
`stats` ("extract and compute statistics usage and costs") maps directly onto this
endpoint's existing doc_count/tombstone_ratio/calibrated-beamsearch-baseline payload --
there is no separate, more detailed stats endpoint to call instead.
"""
function ctl_stats(base_url::String, cmd_args::Dict)
    resp = _ctl_request(:get, "$base_url/datasets/$(cmd_args["dataset"])")
    resp === nothing && return 1
    println(String(resp.body))
    return 0
end

"""
    ctl_log(base_url::String, cmd_args::Dict) -> Int

`similarity-search-ctl log --dataset <id> [--offset] [--limit]` -> `GET
/api/v1/datasets/{id}/log?offset=&limit=` (CHANGELOG.md §1's admin `log` command, chunk 17).
"""
function ctl_log(base_url::String, cmd_args::Dict)
    params = String["offset=$(cmd_args["offset"])"]
    get(cmd_args, "limit", nothing) !== nothing && push!(params, "limit=$(cmd_args["limit"])")
    resp = _ctl_request(:get, "$base_url/datasets/$(cmd_args["dataset"])/log?" * join(params, "&"))
    resp === nothing && return 1
    println(String(resp.body))
    return 0
end

"""
    ctl_jobs_queue(base_url::String, cmd_args::Dict) -> Int

`similarity-search-ctl jobs queue [--status] [--kind]` -> `GET /api/v1/jobs?status=&kind=`.
"""
function ctl_jobs_queue(base_url::String, cmd_args::Dict)
    params = String[]
    get(cmd_args, "status", nothing) !== nothing && push!(params, "status=$(cmd_args["status"])")
    get(cmd_args, "kind", nothing) !== nothing && push!(params, "kind=$(cmd_args["kind"])")
    query = isempty(params) ? "" : "?" * join(params, "&")
    resp = _ctl_request(:get, "$base_url/jobs$query")
    resp === nothing && return 1
    println(String(resp.body))
    return 0
end

"""
    ctl_jobs_result(base_url::String, cmd_args::Dict) -> Int

`similarity-search-ctl jobs result --job-id <id> [--output <path>]` -> `GET
/api/v1/jobs/{id}/result`. The result payload is an arbitrary raw file (e.g. a heavy job's
JSONL output, not necessarily JSON despite the endpoint's `Content-Type` header) --
`--output` writes the raw bytes rather than assuming a text/JSON shape.
"""
function ctl_jobs_result(base_url::String, cmd_args::Dict)
    resp = _ctl_request(:get, "$base_url/jobs/$(cmd_args["job-id"])/result")
    resp === nothing && return 1
    output = get(cmd_args, "output", nothing)
    if output === nothing
        println(String(resp.body))
    else
        write(output, resp.body)
        println("Result saved to $output")
    end
    return 0
end

"""
    ctl_jobs_block(base_url::String, cmd_args::Dict) -> Int

`similarity-search-ctl jobs block --job-id <id>` -> `POST /api/v1/jobs/{id}/block`.
"""
function ctl_jobs_block(base_url::String, cmd_args::Dict)
    resp = _ctl_request(:post, "$base_url/jobs/$(cmd_args["job-id"])/block")
    resp === nothing && return 1
    println(String(resp.body))
    return 0
end

"""
    ctl_jobs_resume(base_url::String, cmd_args::Dict) -> Int

`similarity-search-ctl jobs resume --job-id <id>` -> `POST /api/v1/jobs/{id}/resume`.
"""
function ctl_jobs_resume(base_url::String, cmd_args::Dict)
    resp = _ctl_request(:post, "$base_url/jobs/$(cmd_args["job-id"])/resume")
    resp === nothing && return 1
    println(String(resp.body))
    return 0
end

"""
    ctl_jobs_kill(base_url::String, cmd_args::Dict) -> Int

`similarity-search-ctl jobs kill --job-id <id>` -> `POST /api/v1/jobs/{id}/kill`.
"""
function ctl_jobs_kill(base_url::String, cmd_args::Dict)
    resp = _ctl_request(:post, "$base_url/jobs/$(cmd_args["job-id"])/kill")
    resp === nothing && return 1
    println(String(resp.body))
    return 0
end

"""
    ctl_jobs_gc(base_url::String, cmd_args::Dict) -> Int

`similarity-search-ctl jobs gc [--retention-seconds]` -> `POST /api/v1/admin/jobs/gc`.
"""
function ctl_jobs_gc(base_url::String, cmd_args::Dict)
    resp = _ctl_request(:post, "$base_url/admin/jobs/gc"; body=Dict("retention_seconds" => cmd_args["retention-seconds"]))
    resp === nothing && return 1
    println(String(resp.body))
    return 0
end

"""
    ctl_add_token(base_url::String, cmd_args::Dict) -> Int

`similarity-search-ctl add-token [--user] [--permissions] [--expires-at]` -> `POST
/api/v1/admin/tokens`. `--permissions` is a comma-separated list on the CLI, split into a
real array for the JSON body (`""` becomes an empty permission list, matching the
endpoint's own default).
"""
function ctl_add_token(base_url::String, cmd_args::Dict)
    permissions = filter(!isempty, split(get(cmd_args, "permissions", ""), ","))
    body = Dict{String, Any}("user" => cmd_args["user"], "permissions" => collect(permissions))
    expires_at = get(cmd_args, "expires-at", nothing)
    expires_at !== nothing && (body["expires_at"] = expires_at)

    resp = _ctl_request(:post, "$base_url/admin/tokens"; body=body)
    resp === nothing && return 1
    println(String(resp.body))
    return 0
end

"""
    ctl_del_token(base_url::String, cmd_args::Dict) -> Int

`similarity-search-ctl del-token --token <token>` -> `DELETE /api/v1/admin/tokens/{token}`.
"""
function ctl_del_token(base_url::String, cmd_args::Dict)
    resp = _ctl_request(:delete, "$base_url/admin/tokens/$(cmd_args["token"])")
    resp === nothing && return 1
    println(String(resp.body))
    return 0
end

"""
    ctl_log_tokens(base_url::String) -> Int

`similarity-search-ctl log-tokens` -> `GET /api/v1/admin/tokens` (CHANGELOG.md §1's admin
surface: an audit listing of every token, expired or not).
"""
function ctl_log_tokens(base_url::String)
    resp = _ctl_request(:get, "$base_url/admin/tokens")
    resp === nothing && return 1
    println(String(resp.body))
    return 0
end

"""
    ctl_prune_tokens(base_url::String) -> Int

`similarity-search-ctl prune-tokens` -> `POST /api/v1/admin/tokens/prune`.
"""
function ctl_prune_tokens(base_url::String)
    resp = _ctl_request(:post, "$base_url/admin/tokens/prune")
    resp === nothing && return 1
    println(String(resp.body))
    return 0
end

"""
    _ctl_poll_job(base_url, job_id; interval=0.3, max_attempts=600) -> Union{String, Nothing}

Polls `GET /api/v1/jobs/{job_id}` until it reaches a terminal status (`completed`/
`failed`), returning that status -- or `nothing` on a request failure or if it never
reaches one within `max_attempts`. Backs `ctl_dump`/`ctl_load`: unlike the heavy compute
job kinds (`allknn`/etc, which `-ctl` has no submit command for at all -- only
`queue`/`result`/`block`/`resume`/`kill`/`gc` against a job already submitted some other
way), `dump`/`load` are meant to feel synchronous to the operator running them, so these
two block and wait rather than just printing a `job_id` back.
"""
function _ctl_poll_job(base_url::String, job_id::String; interval::Real=0.3, max_attempts::Int=600)
    for _ in 1:max_attempts
        resp = _ctl_request(:get, "$base_url/jobs/$job_id")
        resp === nothing && return nothing
        status = String(JSON.parse(codeunits(String(resp.body)); dicttype=Dict{String,Any})["status"])
        status in ("completed", "failed") && return status
        sleep(interval)
    end
    return nothing
end

"""
    ctl_dump(base_url::String, cmd_args::Dict) -> Int

`similarity-search-ctl dump --dataset <id>` -> submits a `dump` job (`POST
/api/v1/jobs/dump`), blocks until it finishes, then prints its result -- a JSON pointer
(`bundle_dir` + the parsed `manifest.json`) rather than the bundle's raw file contents,
since the bundle is a directory on the *server's* filesystem, not something meant to be
streamed to the CLI's caller (see `handle_get_job_result`'s `isdir` branch).
"""
function ctl_dump(base_url::String, cmd_args::Dict)
    resp = _ctl_request(:post, "$base_url/jobs/dump"; body=Dict("dataset" => cmd_args["dataset"]))
    resp === nothing && return 1
    job_id = JSON.parse(codeunits(String(resp.body)); dicttype=Dict{String,Any})["job_id"]

    status = _ctl_poll_job(base_url, job_id)
    status === nothing && (println("Error: job did not reach a terminal state"); return 1)

    resp = _ctl_request(:get, "$base_url/jobs/$job_id/result")
    resp === nothing && return 1
    println(String(resp.body))
    return status == "completed" ? 0 : 1
end

"""
    ctl_load(base_url::String, cmd_args::Dict) -> Int

`similarity-search-ctl load --bundle <dir> --dataset <new-id>` -> submits a `load` job
(`POST /api/v1/jobs/load`), blocks until it finishes, then prints the new dataset's
descriptor. Loading only writes the dataset to disk -- it still needs an explicit
`POST /api/v1/admin/datasets/{id}/reload` (not run automatically here) to bring it live on
a running server, exactly like the CLI-driven `dump`/`load` + admin `reload` flow.
"""
function ctl_load(base_url::String, cmd_args::Dict)
    resp = _ctl_request(:post, "$base_url/jobs/load"; body=Dict("bundle" => cmd_args["bundle"], "dataset" => cmd_args["dataset"]))
    resp === nothing && return 1
    job_id = JSON.parse(codeunits(String(resp.body)); dicttype=Dict{String,Any})["job_id"]

    status = _ctl_poll_job(base_url, job_id)
    status === nothing && (println("Error: job did not reach a terminal state"); return 1)

    resp = _ctl_request(:get, "$base_url/jobs/$job_id/result")
    resp === nothing && return 1
    println(String(resp.body))
    return status == "completed" ? 0 : 1
end

# --- Followers (CHANGELOG.md §5.12, issue #5, phase 5b) ----------------------------------------

_ctl_follower_url(base_url::String, cmd_args::AbstractDict) =
    "$base_url/datasets/$(cmd_args["dataset"])/followers/$(HTTP.URIs.escapeuri(cmd_args["name"]))"

function _ctl_print(resp)
    resp === nothing && return 1
    println(String(resp.body))
    return 0
end

"""
    ctl_followers(base_url::String, cmd_args::Dict) -> Int

`similarity-search-ctl followers --dataset <id>` -> `GET /api/v1/datasets/{id}/group`: the
dataset and the status of each of its followers.
"""
ctl_followers(base_url::String, cmd_args::AbstractDict) =
    _ctl_print(_ctl_request(:get, "$base_url/datasets/$(cmd_args["dataset"])/group"))

"""
    ctl_plugins(base_url::String) -> Int

`similarity-search-ctl plugins` -> `GET /api/v1/plugins`: the plugins that the server loaded
(CHANGELOG.md §5.13).
"""
ctl_plugins(base_url::String) = _ctl_print(_ctl_request(:get, "$base_url/plugins"))

"""
    ctl_follower_create(base_url::String, cmd_args::Dict) -> Int

`similarity-search-ctl follower-create --dataset <id> --name <n> --kind <k> --source <s> ...` ->
`POST /api/v1/datasets/{id}/followers`, with the body that [`follower_request`](@ref) builds
from the same arguments as `similarity-search follower-create`.
"""
function ctl_follower_create(base_url::String, cmd_args::AbstractDict)
    body = follower_request(cmd_args)
    body isa String && (println("Error: ", body); return 2)
    return _ctl_print(_ctl_request(:post, "$base_url/datasets/$(cmd_args["dataset"])/followers"; body))
end

"""
    ctl_follower_sync(base_url::String, cmd_args::Dict) -> Int

`similarity-search-ctl follower-sync --dataset <id> --name <n>` -> `POST
/api/v1/datasets/{id}/followers/{n}/sync`.
"""
ctl_follower_sync(base_url::String, cmd_args::AbstractDict) =
    _ctl_print(_ctl_request(:post, _ctl_follower_url(base_url, cmd_args) * "/sync"))

"""
    ctl_follower_rebuild(base_url::String, cmd_args::Dict) -> Int

`similarity-search-ctl follower-rebuild --dataset <id> --name <n>` -> `POST
/api/v1/datasets/{id}/followers/{n}/rebuild`.
"""
ctl_follower_rebuild(base_url::String, cmd_args::AbstractDict) =
    _ctl_print(_ctl_request(:post, _ctl_follower_url(base_url, cmd_args) * "/rebuild"))

"""
    ctl_follower_delete(base_url::String, cmd_args::Dict) -> Int

`similarity-search-ctl follower-delete --dataset <id> --name <n>` -> `DELETE
/api/v1/datasets/{id}/followers/{n}`.
"""
ctl_follower_delete(base_url::String, cmd_args::AbstractDict) =
    _ctl_print(_ctl_request(:delete, _ctl_follower_url(base_url, cmd_args)))

"""
    ctl_migrate_partitions(base_url::String, cmd_args::Dict) -> Int

`similarity-search-ctl migrate-partitions --dataset <id> --name <n> --disjoint true|false` ->
`POST /api/v1/datasets/{id}/migrate-partitions`.
"""
ctl_migrate_partitions(base_url::String, cmd_args::AbstractDict) =
    _ctl_print(_ctl_request(:post, "$base_url/datasets/$(cmd_args["dataset"])/migrate-partitions";
        body=Dict("name" => cmd_args["name"], "disjoint" => cmd_args["disjoint"] == "true")))

"""
    CTL_COMMANDS::Dict{String,Function}

The function that runs each command of `similarity-search-ctl`, by command name, except `jobs`
([`CTL_JOBS_COMMANDS`](@ref)). Each function takes the base URL and the arguments of its
command. [`dispatch_ctl_command`](@ref) finds the function in this table.

The value type is `Function`, an abstract type, so the call to the function is a dynamic call.
Julia then compiles only the function of the command that runs, as with
[`CLI_COMMANDS`](@ref). Measured on 2026-11-24, `similarity-search-ctl list` in a new process
took 15.4 s with a chain of `if` branches, and 11.4 s of it was the compilation of `main_ctl`.
"""
const CTL_COMMANDS = Dict{String,Function}(
    "list" => (base_url, _) -> ctl_list(base_url),
    "stats" => ctl_stats,
    "log" => ctl_log,
    "add-token" => ctl_add_token,
    "del-token" => ctl_del_token,
    "log-tokens" => (base_url, _) -> ctl_log_tokens(base_url),
    "prune-tokens" => (base_url, _) -> ctl_prune_tokens(base_url),
    "dump" => ctl_dump,
    "load" => ctl_load,
    "followers" => ctl_followers,
    "plugins" => (base_url, _) -> ctl_plugins(base_url),
    "follower-create" => ctl_follower_create,
    "follower-sync" => ctl_follower_sync,
    "follower-rebuild" => ctl_follower_rebuild,
    "follower-delete" => ctl_follower_delete,
    "migrate-partitions" => ctl_migrate_partitions,
)

"""
    CTL_JOBS_COMMANDS::Dict{String,Function}

The function that runs each subcommand of `similarity-search-ctl jobs`, by subcommand name.
Each function takes the base URL and the arguments of its subcommand.
"""
const CTL_JOBS_COMMANDS = Dict{String,Function}(
    "queue" => ctl_jobs_queue,
    "result" => ctl_jobs_result,
    "block" => ctl_jobs_block,
    "resume" => ctl_jobs_resume,
    "kill" => ctl_jobs_kill,
    "gc" => ctl_jobs_gc,
)

"""
    dispatch_ctl_command(parsed::AbstractDict, base_url::String) -> Int

Routes a parsed `-ctl` command (see `build_ctl_settings`) to its `ctl_*` implementation --
the `-ctl` counterpart of `dispatch_command` (`cli_handlers.jl`). `jobs` is a nested
subcommand (`jobs queue|result|block|resume|kill|gc`), so its own `%COMMAND%` is resolved
one level deeper than the others. The mappings are [`CTL_COMMANDS`](@ref) and
[`CTL_JOBS_COMMANDS`](@ref).
"""
function dispatch_ctl_command(parsed::AbstractDict, base_url::String)
    cmd = parsed["%COMMAND%"]
    cmd === nothing && (println("No command specified."); return 0)

    if cmd == "jobs"
        jparsed = parsed["jobs"]
        jcmd = jparsed["%COMMAND%"]
        handler = get(CTL_JOBS_COMMANDS, jcmd, nothing)
        handler === nothing && error("dispatch_ctl_command: unknown jobs subcommand '$jcmd'")
        return handler(base_url, jparsed[jcmd])
    end

    handler = get(CTL_COMMANDS, cmd, nothing)
    handler === nothing && error("dispatch_ctl_command: unknown command '$cmd'")
    return handler(base_url, get(parsed, cmd, nothing))
end
