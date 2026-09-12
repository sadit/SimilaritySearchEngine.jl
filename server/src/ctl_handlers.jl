# similarity-search-ctl command implementations -- thin HTTP calls against a running
# similarity-search-serve instance (PLAN.md §1: "admin is a control-plane client, not a
# second process touching the same RocksDB files"). Every function here takes the base
# API URL plus that subcommand's own parsed args, makes one HTTP call, and prints the
# response -- there is deliberately no business logic duplicated from `server.jl` here.

"""
    _ctl_request(method::Symbol, url::String; body=nothing) -> Union{HTTP.Response, Nothing}

Shared error handling for every `-ctl` command: a non-2xx response prints the server's own
JSON error body and returns `nothing` (so the caller can just check for that and return 1);
a connection failure prints PLAN.md §1's own "server not reachable" wording rather than a
raw stacktrace, also returning `nothing`.
"""
function _ctl_request(method::Symbol, url::String; body=nothing)
    try
        if method === :get
            return HTTP.get(url; retry=false)
        elseif method === :post
            headers = ["Content-Type" => "application/json"]
            return HTTP.post(url, headers, body === nothing ? "" : JSON3.write(body); retry=false)
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

`similarity-search-ctl stats --dataset <id>` -> `GET /api/v1/datasets/{id}`. PLAN.md's
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
/api/v1/datasets/{id}/log?offset=&limit=` (PLAN.md §1's admin `log` command, chunk 17).
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

`similarity-search-ctl log-tokens` -> `GET /api/v1/admin/tokens` (PLAN.md §1's admin
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
        status = String(JSON3.read(String(resp.body), Dict{String, Any})["status"])
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
    job_id = JSON3.read(String(resp.body), Dict{String, Any})["job_id"]

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
    job_id = JSON3.read(String(resp.body), Dict{String, Any})["job_id"]

    status = _ctl_poll_job(base_url, job_id)
    status === nothing && (println("Error: job did not reach a terminal state"); return 1)

    resp = _ctl_request(:get, "$base_url/jobs/$job_id/result")
    resp === nothing && return 1
    println(String(resp.body))
    return status == "completed" ? 0 : 1
end

"""
    dispatch_ctl_command(parsed::AbstractDict, base_url::String) -> Int

Routes a parsed `-ctl` command (see `build_ctl_settings`) to its `ctl_*` implementation --
the `-ctl` counterpart of `dispatch_command` (`cli_handlers.jl`). `jobs` is a nested
subcommand (`jobs queue|result|block|resume|kill|gc`), so its own `%COMMAND%` is resolved
one level deeper than the others.
"""
function dispatch_ctl_command(parsed::AbstractDict, base_url::String)
    cmd = parsed["%COMMAND%"]
    cmd === nothing && (println("No command specified."); return 0)

    cmd == "list" && return ctl_list(base_url)
    cmd == "stats" && return ctl_stats(base_url, parsed["stats"])
    cmd == "log" && return ctl_log(base_url, parsed["log"])
    cmd == "add-token" && return ctl_add_token(base_url, parsed["add-token"])
    cmd == "del-token" && return ctl_del_token(base_url, parsed["del-token"])
    cmd == "log-tokens" && return ctl_log_tokens(base_url)
    cmd == "prune-tokens" && return ctl_prune_tokens(base_url)
    cmd == "dump" && return ctl_dump(base_url, parsed["dump"])
    cmd == "load" && return ctl_load(base_url, parsed["load"])

    if cmd == "jobs"
        jparsed = parsed["jobs"]
        jcmd = jparsed["%COMMAND%"]
        jcmd == "queue" && return ctl_jobs_queue(base_url, jparsed["queue"])
        jcmd == "result" && return ctl_jobs_result(base_url, jparsed["result"])
        jcmd == "block" && return ctl_jobs_block(base_url, jparsed["block"])
        jcmd == "resume" && return ctl_jobs_resume(base_url, jparsed["resume"])
        jcmd == "kill" && return ctl_jobs_kill(base_url, jparsed["kill"])
        jcmd == "gc" && return ctl_jobs_gc(base_url, jparsed["gc"])
        error("dispatch_ctl_command: unknown jobs subcommand '$jcmd'")
    end

    error("dispatch_ctl_command: unknown command '$cmd'")
end
