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

        @async Executors.run_dispatcher!(job_mgr, executor)

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
