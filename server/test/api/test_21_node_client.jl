using Test
using HTTP
using JSON

# The Node.js client (node/, PLAN.md §5.17) against the suite's server. The tests are TypeScript, in
# node/test/server.test.ts and node/test/auth.test.ts. This file builds the client, runs the two
# files with `node --test`, and reads the counts that the TAP reporter prints.
#
# The `node` is the one of the environment variable SSE_NODE, or the one in PATH. It must be
# Node.js 22 or later. Without it, the tests are skipped and the reason is printed.

"The `node` for the tests, or the reason why there is none."
function node_for_tests()
    node = get(ENV, "SSE_NODE", nothing)
    node === nothing && (node = Sys.which("node"))
    node === nothing && return (nothing, "no `node` in PATH, and SSE_NODE is not set")
    version = try
        strip(read(`$node --version`, String))
    catch e
        return (nothing, "`$node --version` failed: $(sprint(showerror, e))")
    end
    m = match(r"^v(\d+)\.", version)
    (m === nothing || parse(Int, m[1]) < 22) && return (nothing, "`$node` is $version, and the client needs Node.js 22 or later")
    return (node, version)
end

"The `npm` next to `node`: the one in its directory, or in the directory of the file it links to."
function npm_for(node::AbstractString)
    for dir in (dirname(node), dirname(realpath(node)))
        npm = joinpath(dir, "npm")
        isfile(npm) && return npm
    end
    return nothing
end

"""
Runs `node --test` on `file` with the variables of `env`, and returns the exit code, the counts
of the TAP reporter, and the output.
"""
function run_node_tests(node, dir, file, env)
    out = IOBuffer()
    cmd = setenv(`$node --test --test-reporter=tap $file`, merge(copy(ENV), env); dir)
    proc = run(pipeline(ignorestatus(cmd); stdout=out, stderr=out))
    text = String(take!(out))
    count(name) = (m = match(Regex("^# $name (\\d+)\$", "m"), text); m === nothing ? -1 : parse(Int, m[1]))
    # The diagnostics of the tests (`t.diagnostic`) are the indented comment lines of TAP.
    notes = [strip(l) for l in split(text, '\n') if occursin(r"^\s+# (?!Subtest:)", l)]
    r = (exitcode=proc.exitcode, pass=count("pass"), fail=count("fail"), skipped=count("skipped"), output=text)
    @info "Node.js client, $file: $(r.pass) passed, $(r.fail) failed, $(r.skipped) skipped" notes
    return r
end

@testset "21. Node.js client" begin
    srv = ensure_test_server()
    node, about = node_for_tests()
    if node === nothing
        @warn "The tests of the Node.js client are skipped: $about"
        @test_skip "Node.js client"
    else
        @info "The Node.js client is tested with $node ($about)"
        dir = normpath(joinpath(@__DIR__, "..", "..", "node"))
        npm = npm_for(node)
        @test npm !== nothing
        # `npm install` installs TypeScript and builds the client (its `prepare` script). With the
        # packages installed, `npm run build` only compiles the sources again.
        step = isdir(joinpath(dir, "node_modules")) ? `$npm run build` : `$npm install --no-audit --no-fund`
        path = dirname(node) * ":" * get(ENV, "PATH", "")
        build = IOBuffer()
        built = run(pipeline(ignorestatus(setenv(step, merge(copy(ENV), Dict("PATH" => path)); dir)); stdout=build, stderr=build))
        @test built.exitcode == 0
        built.exitcode == 0 || println(String(take!(build)))

        data = normpath(joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl"))
        env = Dict("SSE_TEST_URL" => srv.root_url, "SSE_TEST_DATA" => data, "PATH" => path)

        r = run_node_tests(node, dir, "dist/test/server.test.js", env)
        @test r.exitcode == 0
        @test r.fail == 0 && r.skipped == 0
        @test r.pass >= 11
        r.exitcode == 0 || println(r.output)

        # The same server with authentication, as test_14 does: two tokens, created before it is
        # enabled, and the switch set back afterwards.
        new_token(user, perm) = JSON.parse(String(HTTP.post("$(srv.base_url)/admin/tokens", [],
            JSON.json(Dict("user" => user, "permissions" => [perm]))).body)).token
        admin, reader = new_token("node_admin", "admin:*"), new_token("node_reader", "read:*")
        srv.app.auth_enabled[] = true
        a = try
            run_node_tests(node, dir, "dist/test/auth.test.js",
                           merge(env, Dict("SSE_TEST_ADMIN_TOKEN" => admin, "SSE_TEST_READER_TOKEN" => reader)))
        finally
            srv.app.auth_enabled[] = false
        end
        @test a.exitcode == 0
        @test a.pass == 1 && a.fail == 0 && a.skipped == 0
        a.exitcode == 0 || println(a.output)
        for t in (admin, reader)
            HTTP.delete("$(srv.base_url)/admin/tokens/$t")
        end
    end
end
