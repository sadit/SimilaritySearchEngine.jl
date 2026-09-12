using Test
using HTTP
using JSON3

@testset "8. Calibration and Beamsearch Safety Barrier Tests" begin
    with_test_server() do base_url, workdir

        data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")
        docs = [JSON3.read(line) for line in readlines(data_path)[1:150]]

        resp = HTTP.post("$base_url/datasets", [], JSON3.write(Dict("id" => "calib_test_ds", "index_type" => "searchgraph", "distance" => "L2")))
        @test resp.status == 201
        resp = HTTP.post("$base_url/simsearch/calib_test_ds/append", [], JSON3.write(Dict("items" => docs)))
        @test resp.status == 200

        # 1. Before any calibrate call, the baseline reported is the library's own
        # BeamSearch() default -- construction no longer autotunes on its own
        # (PLAN.md §5.6's warning about the two mechanisms fighting), so this is the real
        # bootstrap value, not a placeholder.
        resp = HTTP.get("$base_url/datasets/calib_test_ds")
        @test resp.status == 200
        before = JSON3.read(String(resp.body)).beamsearch_baseline
        @test before.bsize == 4
        @test before.delta ≈ 1.0
        @test before.maxvisits == 1_000_000

        # 2. POST .../calibrate runs the real sweep and installs a new baseline.
        resp = HTTP.post("$base_url/simsearch/calib_test_ds/calibrate", [], JSON3.write(Dict("minrecall" => 0.9, "numqueries" => 40, "ksearch" => 8)))
        @test resp.status == 200
        calibrated = JSON3.read(String(resp.body))
        @test calibrated.status == "calibrated"
        baseline = calibrated.baseline
        @test baseline.bsize > 0
        @test baseline.delta > 0
        @test baseline.maxvisits > 0

        # 3. The dataset descriptor reflects the same calibrated baseline afterward.
        resp = HTTP.get("$base_url/datasets/calib_test_ds")
        @test resp.status == 200
        after = JSON3.read(String(resp.body)).beamsearch_baseline
        @test after.bsize == baseline.bsize
        @test after.maxvisits == baseline.maxvisits

        # 4. A plain search (no beamsearch_overrides) is completely unaffected.
        resp = HTTP.post("$base_url/simsearch/calib_test_ds/search", [], JSON3.write(Dict("vector" => docs[1].vector, "k" => 5)))
        @test resp.status == 200
        @test !any(h -> h.first == "X-Beamsearch-Warning", resp.headers)
        plain = JSON3.read(String(resp.body))
        @test length(plain.results) == 5
        @test plain.results[1].id == docs[1].doc_id

        # 5. An override AT the calibrated baseline (not above it) -> no warning header.
        resp = HTTP.post("$base_url/simsearch/calib_test_ds/search", [], JSON3.write(Dict(
            "vector" => docs[1].vector, "k" => 5,
            "beamsearch_overrides" => Dict("bsize" => baseline.bsize),
        )))
        @test resp.status == 200
        @test !any(h -> h.first == "X-Beamsearch-Warning", resp.headers)

        # 6. An override ABOVE baseline but within the 3x safety multiplier -> allowed,
        # with a warning header (PLAN.md §3's "log WARN + return an HTTP warning header").
        resp = HTTP.post("$base_url/simsearch/calib_test_ds/search", [], JSON3.write(Dict(
            "vector" => docs[1].vector, "k" => 5,
            "beamsearch_overrides" => Dict("bsize" => baseline.bsize * 2),
        )))
        @test resp.status == 200
        @test any(h -> h.first == "X-Beamsearch-Warning", resp.headers)

        # 7. An override far beyond the safety multiplier -> hard-rejected, not just warned.
        resp = try
            HTTP.post("$base_url/simsearch/calib_test_ds/search", [], JSON3.write(Dict(
                "vector" => docs[1].vector, "k" => 5,
                "beamsearch_overrides" => Dict("bsize" => baseline.bsize * 100),
            )))
        catch e
            e.response
        end
        @test resp.status == 400

        # 8. calibrate / beamsearch_overrides are meaningless for an exact index -> 400,
        # not a silent no-op.
        resp = HTTP.post("$base_url/datasets", [], JSON3.write(Dict("id" => "calib_exact_ds", "index_type" => "exhaustive_search", "distance" => "L2")))
        @test resp.status == 201
        resp = HTTP.post("$base_url/simsearch/calib_exact_ds/append", [], JSON3.write(Dict("items" => docs[1:5])))
        @test resp.status == 200

        resp = try
            HTTP.post("$base_url/simsearch/calib_exact_ds/calibrate", [], JSON3.write(Dict()))
        catch e
            e.response
        end
        @test resp.status == 400

        resp = try
            HTTP.post("$base_url/simsearch/calib_exact_ds/search", [], JSON3.write(Dict("vector" => docs[1].vector, "k" => 3, "beamsearch_overrides" => Dict("bsize" => 5))))
        catch e
            e.response
        end
        @test resp.status == 400

        # 9. calibrate against a dataset with no data yet -> 400 (nothing to sweep over).
        resp = HTTP.post("$base_url/datasets", [], JSON3.write(Dict("id" => "calib_empty_ds", "index_type" => "searchgraph")))
        @test resp.status == 201
        resp = try
            HTTP.post("$base_url/simsearch/calib_empty_ds/calibrate", [], JSON3.write(Dict()))
        catch e
            e.response
        end
        @test resp.status == 400
    end
end
