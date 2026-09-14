using Test
using HTTP
using JSON3

# Authentication over HTTP (PLAN.md §4.1). The decision itself is tested without a socket in
# `test_boundary.jl`; what this file adds is that the decision is actually reached on a real
# request, on the routes as they are registered.
#
# The shared test server runs with authentication disabled, like a server whose configuration
# does not enable it. This testset turns it on for its own duration and off again in a
# `finally`, so the files after it see the server they expect.
@testset "14. Authentication and permissions over HTTP" begin
    with_test_server() do base_url, workdir
        srv = ensure_test_server()
        mgr = srv.app.token_mgr

        resp = HTTP.post("$base_url/datasets", [], JSON3.write(Dict("id" => "auth_ds", "index_type" => "searchgraph", "distance" => "L2")))
        @test resp.status == 201
        resp = HTTP.post("$base_url/datasets", [], JSON3.write(Dict("id" => "auth_other", "index_type" => "searchgraph", "distance" => "L2")))
        @test resp.status == 201

        docs = [Dict("doc_id" => "a$i", "vector" => [Float64(i), 0.0, 0.0, 0.0]) for i in 1:5]
        resp = HTTP.post("$base_url/datasets/auth_ds/append", [], JSON3.write(Dict("items" => docs)))
        @test resp.status == 200

        reader = JSON3.read(String(HTTP.post("$base_url/admin/tokens", [],
            JSON3.write(Dict("user" => "reader", "permissions" => ["read:*"]))).body)).token
        writer = JSON3.read(String(HTTP.post("$base_url/admin/tokens", [],
            JSON3.write(Dict("user" => "writer", "permissions" => ["write:auth_ds"]))).body)).token
        admin = JSON3.read(String(HTTP.post("$base_url/admin/tokens", [],
            JSON3.write(Dict("user" => "admin", "permissions" => ["admin:*"]))).body)).token
        expired = JSON3.read(String(HTTP.post("$base_url/admin/tokens", [],
            JSON3.write(Dict("user" => "past", "permissions" => ["admin:*"],
                             "expires_at" => "2000-01-01T00:00:00"))).body)).token

        # A malformed expiry is refused when the token is created, not when it is presented.
        resp = try
            HTTP.post("$base_url/admin/tokens", [], JSON3.write(Dict("user" => "x", "permissions" => ["read"], "expires_at" => "next tuesday")))
        catch e
            e.response
        end
        @test resp.status == 400

        bearer(tok) = ["Authorization" => "Bearer $tok"]
        status(f) = try
            f().status
        catch e
            e.response.status
        end

        srv.app.auth_enabled[] = true
        try
            query = JSON3.write(Dict("vector" => [1.0, 0.0, 0.0, 0.0], "k" => 3))

            # --- no token -----------------------------------------------------------------
            @test status(() -> HTTP.post("$base_url/datasets/auth_ds/search", [], query)) == 401
            @test status(() -> HTTP.get("$base_url/datasets")) == 401
            @test status(() -> HTTP.get("$base_url/jobs")) == 401
            @test status(() -> HTTP.post("$base_url/admin/tokens", [], JSON3.write(Dict("user" => "u", "permissions" => ["read"])))) == 401

            # Health and metrics answer without one, which is what a supervisor and a metrics
            # collector need.
            @test HTTP.get("$(srv.root_url)/healthz").status == 200
            @test HTTP.get("$(srv.root_url)/readyz").status == 200
            @test HTTP.get("$(srv.root_url)/metrics").status == 200

            # --- an expired token is as good as none --------------------------------------
            @test status(() -> HTTP.get("$base_url/datasets", bearer(expired))) == 401

            # --- read -----------------------------------------------------------------------
            @test status(() -> HTTP.post("$base_url/datasets/auth_ds/search", bearer(reader), query)) == 200
            @test status(() -> HTTP.get("$base_url/datasets", bearer(reader))) == 200
            @test status(() -> HTTP.get("$base_url/datasets/auth_ds", bearer(reader))) == 200
            # ... does not include writing, or the administrative routes
            @test status(() -> HTTP.post("$base_url/datasets/auth_ds/append", bearer(reader), JSON3.write(Dict("items" => docs)))) == 403
            @test status(() -> HTTP.delete("$base_url/datasets/auth_other", bearer(reader))) == 403
            @test status(() -> HTTP.get("$base_url/admin/tokens", bearer(reader))) == 403
            # The operation log records the token of each caller, so reading it is administrative
            @test status(() -> HTTP.get("$base_url/datasets/auth_ds/log", bearer(reader))) == 403

            # --- write, scoped to one dataset ----------------------------------------------
            @test status(() -> HTTP.post("$base_url/datasets/auth_ds/append", bearer(writer), JSON3.write(Dict("items" => docs)))) == 200
            @test status(() -> HTTP.post("$base_url/datasets/auth_ds/search", bearer(writer), query)) == 200   # write includes read
            @test status(() -> HTTP.post("$base_url/datasets/auth_other/append", bearer(writer), JSON3.write(Dict("items" => docs)))) == 403
            @test status(() -> HTTP.post("$base_url/datasets/auth_other/search", bearer(writer), query)) == 403
            # A route that is not about one dataset requires a permission over every dataset
            @test status(() -> HTTP.get("$base_url/datasets", bearer(writer))) == 403

            # The 403 body names the permission the caller would need
            resp = try
                HTTP.get("$base_url/admin/tokens", bearer(reader))
            catch e
                e.response
            end
            body = JSON3.read(String(resp.body))
            @test body.required == "admin:*"
            @test body.kind == "Forbidden"

            # A request made with a valid token is logged under the name on it, and the token
            # itself is not in the log at all.
            @test status(() -> HTTP.post("$base_url/datasets/auth_ds/search", bearer(writer), query)) == 200
            log_body = String(HTTP.get("$base_url/datasets/auth_ds/log", bearer(admin)).body)
            @test !occursin(writer, log_body)
            entry = JSON3.read(log_body).entries[1]
            @test entry.details.user == "writer"
            @test length(entry.details.token_fingerprint) == 16

            # --- admin ----------------------------------------------------------------------
            @test status(() -> HTTP.get("$base_url/admin/tokens", bearer(admin))) == 200
            @test status(() -> HTTP.get("$base_url/datasets/auth_ds/log", bearer(admin))) == 200
            @test status(() -> HTTP.delete("$base_url/datasets/auth_other", bearer(admin))) == 200

            # A revoked token stops working immediately: the check is a lookup, not a cache.
            @test status(() -> HTTP.get("$base_url/datasets", bearer(reader))) == 200
            HTTP.delete("$base_url/admin/tokens/$reader", bearer(admin))
            @test status(() -> HTTP.get("$base_url/datasets", bearer(reader))) == 401
        finally
            srv.app.auth_enabled[] = false
        end

        # Disabled again, the same request that answered 401 answers normally.
        @test HTTP.get("$base_url/datasets").status == 200
    end
end
