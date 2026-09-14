using Test
using HTTP
using JSON3

# A declared meta schema over HTTP (PLAN.md §4.5): declared at creation, enforced on append,
# reported back, and used by a filter to compare in the declared type. The declaration itself
# belongs to the engine; what this file checks is that it survives the round trip through the
# API and governs what the endpoints do.
@testset "15. Declared meta schema" begin
    with_test_server() do base_url, workdir
        resp = HTTP.post("$base_url/datasets", [], JSON3.write(Dict(
            "id" => "schema_ds", "index_type" => "searchgraph", "distance" => "L2",
            "meta_schema" => [Dict("name" => "year", "type" => "int64"),
                              Dict("name" => "lang", "type" => "string"),
                              Dict("name" => "when", "type" => "timestamp")])))
        @test resp.status == 201

        # The declaration is reported back from the project, not from a copy in the sidecar
        detail = JSON3.read(String(HTTP.get("$base_url/datasets/schema_ds").body))
        @test [f.name for f in detail.meta_schema] == ["year", "lang", "when"]
        @test [f.type for f in detail.meta_schema] == ["int64", "string", "timestamp"]

        # A dataset that declares nothing reports nothing, and keeps today's behavior
        @test HTTP.post("$base_url/datasets", [], JSON3.write(Dict("id" => "free_ds", "index_type" => "searchgraph"))).status == 201
        @test JSON3.read(String(HTTP.get("$base_url/datasets/free_ds").body)).meta_schema === nothing

        # A malformed declaration is a 400, like any other invalid request
        resp = try
            HTTP.post("$base_url/datasets", [], JSON3.write(Dict("id" => "bad_ds", "meta_schema" => [Dict("name" => "x", "type" => "int32")])))
        catch e
            e.response
        end
        @test resp.status == 400

        # --- what a declared field accepts ------------------------------------------------
        items = [Dict("doc_id" => "a", "vector" => [1.0, 0.0, 0.0, 0.0],
                      "year" => 2024, "lang" => "es", "when" => "2026-01-02T00:00:00", "free" => "kept"),
                 Dict("doc_id" => "b", "vector" => [0.0, 1.0, 0.0, 0.0],
                      "year" => 1999, "lang" => "en", "when" => "1999-06-01T12:00:00")]
        @test HTTP.post("$base_url/datasets/schema_ds/append", [], JSON3.write(Dict("items" => items))).status == 200

        # The value comes back as the declared type, and an undeclared field is stored as it
        # arrived. `/fetch` merges the metadata into each result, so the fields are top level.
        fetched = JSON3.read(String(HTTP.post("$base_url/datasets/schema_ds/fetch", [], JSON3.write(Dict("ids" => ["a"]))).body)).results
        entry = only(fetched)
        @test entry.year === 2024
        @test entry.free == "kept"

        # A value of another type is refused, and the message says which field and which type
        resp = try
            HTTP.post("$base_url/datasets/schema_ds/append", [], JSON3.write(Dict("items" => [
                Dict("doc_id" => "c", "vector" => [0.0, 0.0, 1.0, 0.0], "year" => "2024")])))
        catch e
            e.response
        end
        @test resp.status == 400
        body = JSON3.read(String(resp.body))
        @test occursin("year", body.error)
        @test body.kind == "PayloadMismatch"

        # The same field with no declaration behind it is stored as it arrives
        @test HTTP.post("$base_url/datasets/free_ds/append", [], JSON3.write(Dict("items" => [
            Dict("doc_id" => "c", "vector" => [0.0, 0.0, 1.0, 0.0], "year" => "2024")]))).status == 200

        # --- what the declaration does for a filter ---------------------------------------
        # In a search result, `id` is the caller's identifier and `doc_id` is the internal
        # one: the two names sit the opposite way round from the record they come from.
        query(filter) = JSON3.write(Dict("vector" => [1.0, 0.0, 0.0, 0.0], "k" => 5, "filter" => filter))

        hits = JSON3.read(String(HTTP.post("$base_url/datasets/schema_ds/search", [], query(Dict("year" => Dict("gte" => 2020)))).body)).results
        @test [h.id for h in hits] == ["a"]

        # An instant compares as an instant: the same moment written without seconds matches
        hits = JSON3.read(String(HTTP.post("$base_url/datasets/schema_ds/search", [], query(Dict("when" => "2026-01-02T00:00"))).body)).results
        @test [h.id for h in hits] == ["a"]

        hits = JSON3.read(String(HTTP.post("$base_url/datasets/schema_ds/search", [], query(Dict("when" => Dict("lt" => "2000-01-01T00:00:00")))).body)).results
        @test [h.id for h in hits] == ["b"]

        # --- the declaration survives a restart of the dataset ----------------------------
        @test HTTP.post("$base_url/admin/datasets/schema_ds/unload", [], "").status == 200
        @test HTTP.post("$base_url/admin/datasets/schema_ds/reload", [], "").status == 200
        detail = JSON3.read(String(HTTP.get("$base_url/datasets/schema_ds").body))
        @test [f.name for f in detail.meta_schema] == ["year", "lang", "when"]

        resp = try
            HTTP.post("$base_url/datasets/schema_ds/append", [], JSON3.write(Dict("items" => [
                Dict("doc_id" => "d", "vector" => [0.0, 0.0, 0.0, 1.0], "year" => "nope")])))
        catch e
            e.response
        end
        @test resp.status == 400
    end
end
