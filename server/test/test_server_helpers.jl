using Test
using HTTP
using JSON3
using SimilaritySearchServer
using SimilaritySearchEngine
using SimilaritySearchServer.Server
using SimilaritySearchEngine.Project
using SimilaritySearchServer.Jobs
using SimilaritySearchServer.Cursors
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
        cursor_mgr = Cursors.init_cursor_manager(workdir)
        executor = LocalCLIExecutor()
        token_mgr = Tokens.open_token_manager(workdir)
        app = AppState(
            workdir,
            job_mgr,
            cursor_mgr,
            executor,
            token_mgr,
            Dict{String, SimilaritySearchEngine.EmbeddedEngine}(),
            ReentrantLock()
        )

        errormonitor(@async Executors.run_dispatcher!(job_mgr, executor))
        # Periodic followers are checked often, so that their test does not wait for long.
        app.followers.check_seconds = 0.5
        start_follower_task!(app)

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
small datasets of the tests. `cli/01` to `cli/06` run the real binary.
"""
_run_cli(args::Vector{String}) = _capture_main(SimilaritySearchServer.main, args)
