using Test
using HTTP
using JSON
using SimilaritySearchServer
using SimilaritySearchEngine
using SimilaritySearchServer.Server
using SimilaritySearchEngine.Project
using SimilaritySearchServer.Jobs
using SimilaritySearchServer.Executors
using SimilaritySearchEngine.IndexEngine
using SimilaritySearchServer.Tokens

# Oxygen.jl registers routes into a module-global router, so running one Oxygen server
# per test file (as the old HTTP.jl-based `with_test_server` did) would have each new
# `run_server` call silently re-point every route's closure at a different `AppState`,
# corrupting whichever server started first. Instead, the whole test suite shares exactly
# one Oxygen server instance, started lazily on first use and reused by every test file.
const TEST_SERVER = Ref{Union{Nothing, NamedTuple}}(nothing)

function _wait_ready(base_url_root::String; timeout::Real=15.0)
    t0 = time()
    while time() - t0 < timeout
        try
            resp = HTTP.get("$base_url_root/readyz"; readtimeout=1, retry=false)
            resp.status == 200 && return true
        catch
            # not up yet
        end
        sleep(0.05)
    end
    return false
end

function ensure_test_server()
    if TEST_SERVER[] === nothing
        workdir = mktempdir()
        job_mgr = Jobs.init_job_manager(workdir)
        executor = LocalCLIExecutor()
        token_mgr = Tokens.open_token_manager(workdir)
        app = AppState(
            workdir,
            job_mgr,
            executor,
            token_mgr,
            Dict{String, SimilaritySearchEngine.EmbeddedEngine}(),
            ReentrantLock()
        )

        # The dispatcher reads `jobs/` of `workdir` every 0.5 s. At the end of the process, Julia
        # deletes `workdir`, and the next read raises an error. That error is not the failure of a
        # test: the full run of 2026-11-24 printed it after all its tests passed.
        errormonitor(@async try
            Executors.run_dispatcher!(job_mgr, executor)
        catch e
            isdir(workdir) && rethrow()
        end)
        # Periodic followers are checked often, so that their test does not wait for long.
        app.followers.check_seconds = 0.5
        start_follower_task!(app)

        # The test plugin is loaded before the server starts, as `serve` loads `[plugins]`
        # (CHANGELOG.md §5.13). `api/19` uses it.
        load_plugins!(Dict{String,Any}("plugins" => Dict{String,Any}("SSETestPlugin" => Dict{String,Any}(
            "path" => joinpath(@__DIR__, "plugins", "sse_test_plugin.jl"),
            "settings" => Dict{String,Any}("label" => "test")))))

        port = 8765
        run_server("127.0.0.1", port, app; async=true)

        root = "http://127.0.0.1:$port"
        @assert _wait_ready(root) "test server failed to become ready on $root"

        TEST_SERVER[] = (base_url="$root/api/v1", root_url=root, workdir=workdir, app=app)
    end
    return TEST_SERVER[]
end

function with_test_server(f::Function)
    srv = ensure_test_server()
    f(srv.base_url, srv.workdir)
end

# Runs `main(args)` in this process and returns its exit code and what it printed.
function _capture_main(main::Function, args::Vector{String})
    path = tempname()
    code = open(path, "w") do io
        redirect_stdout(io) do
            main(args)
        end
    end
    out = read(path, String)
    rm(path, force=true)
    return code, out
end

"""
Runs `SimilaritySearchServer.main_ctl(args)` in-process against the shared test server and
captures whatever it printed, instead of spawning a real `similarity-search-ctl` subprocess
-- `main_ctl` is a thin, already-unit-testable wrapper (parse -> dispatch_ctl_command ->
one HTTP call), and every one of its `ctl_*` handlers only ever `println`s its result, so
capturing stdout is both faster and gives the assertions below direct string access.
"""
_run_ctl(args::Vector{String}) = _capture_main(SimilaritySearchServer.main_ctl, args)

"""
Runs `SimilaritySearchServer.main(args)`, the `similarity-search` command line, in this process.
A subprocess starts Julia again for each command, which takes longer than the command for the
small datasets of the tests. Measured on 2026-11-24, a `build` of 724 vectors took 18.9 s in a
subprocess, and most of that time was compilation. Three tests run the real binary: a `build`
in `cli/01`, and a `searchbatch` and an exit code in `cli/02`. `cli/05` runs the real binary for
the errors of ArgParse, because ArgParse calls `exit()` for them.
"""
_run_cli(args::Vector{String}) = _capture_main(SimilaritySearchServer.main, args)

"""
Runs a `similarity-search` command in this process with [`_run_cli`](@ref), and returns `true`
when its exit code is 0. Each argument is converted to a `String`.
"""
_cli_ok(args...) = first(_run_cli(String[string(a) for a in args])) == 0

"""
Runs a `similarity-search` command in this process with [`_run_cli`](@ref), and returns `true`
when the command fails. A command fails when its exit code is not 0, or when it raises an
exception. The real binary ends with exit code 1 on an exception that the command does not
convert to an exit code.
"""
function _cli_fails(args...)
    code = try
        first(_run_cli(String[string(a) for a in args]))
    catch e
        e isa InterruptException && rethrow()
        return true
    end
    return code != 0
end
