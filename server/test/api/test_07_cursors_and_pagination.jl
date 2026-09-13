using Test
using HTTP
using JSON3

@testset "7. Cursors and Pagination Tests" begin
    with_test_server() do base_url, workdir

        data_path = joinpath(@__DIR__, "..", "..", "..", "test", "data", "frankenstein.jsonl")
        docs = [JSON3.read(line) for line in readlines(data_path)[1:30]]

        resp = HTTP.post("$base_url/datasets", [], JSON3.write(Dict("id" => "cursor_ds", "index_type" => "searchgraph", "distance" => "L2")))
        @test resp.status == 201
        resp = HTTP.post("$base_url/datasets/cursor_ds/append", [], JSON3.write(Dict("items" => docs)))
        @test resp.status == 200

        # 1. Search without `page_size` is unchanged: no cursor_id, full results inline.
        resp = HTTP.post("$base_url/datasets/cursor_ds/search", [], JSON3.write(Dict("vector" => docs[1].vector, "k" => 5)))
        @test resp.status == 200
        plain = JSON3.read(String(resp.body))
        @test length(plain.results) == 5
        @test !haskey(plain, :cursor_id)

        # 2. Search with `page_size` materializes a cursor and returns its first page.
        resp = HTTP.post("$base_url/datasets/cursor_ds/search", [], JSON3.write(Dict("vector" => docs[1].vector, "k" => 13, "page_size" => 5)))
        @test resp.status == 200
        first_page = JSON3.read(String(resp.body))
        @test length(first_page.results) == 5
        @test first_page.total == 13
        @test first_page.exhausted == false
        cursor_id = first_page.cursor_id
        @test !isempty(cursor_id)

        # 3. Poll the remaining pages (5 + 5 + 3 = 13) until exhausted, and check every
        # doc_id across all pages together is the full, non-overlapping candidate set.
        seen_doc_ids = Set(r.doc_id for r in first_page.results)
        exhausted = first_page.exhausted
        pages_polled = 0
        while !exhausted
            resp = HTTP.get("$base_url/cursors/$cursor_id")
            @test resp.status == 200
            page = JSON3.read(String(resp.body))
            for r in page.results
                @test !(r.doc_id in seen_doc_ids)  # no page repeats an earlier id
                push!(seen_doc_ids, r.doc_id)
            end
            exhausted = page.exhausted
            pages_polled += 1
            @test pages_polled <= 10  # guard against an infinite loop if exhaustion logic regresses
        end
        @test length(seen_doc_ids) == 13

        # 4. Polling an already-exhausted cursor -> 409, not a silent empty page.
        resp = try
            HTTP.get("$base_url/cursors/$cursor_id")
        catch e
            e.response
        end
        @test resp.status == 409

        # 5. Polling a cursor id that never existed -> 404 (distinct from 409 above).
        resp = try
            HTTP.get("$base_url/cursors/not_a_real_cursor")
        catch e
            e.response
        end
        @test resp.status == 404

        # 6. `?limit=` on a poll overrides the cursor's own page_size for that call.
        resp = HTTP.post("$base_url/datasets/cursor_ds/search", [], JSON3.write(Dict("vector" => docs[1].vector, "k" => 10, "page_size" => 2)))
        cursor_id2 = JSON3.read(String(resp.body)).cursor_id
        resp = HTTP.get("$base_url/cursors/$cursor_id2?limit=100")
        @test resp.status == 200
        big_page = JSON3.read(String(resp.body))
        @test length(big_page.results) == 8  # the 10 - 2 already consumed by the first page
        @test big_page.exhausted == true

        # 7. Plain offset/limit pagination on GET /api/v1/datasets (PLAN.md §5.4's other
        # pagination mechanism -- cheap listings don't need a Cursors round-trip).
        resp = HTTP.post("$base_url/datasets", [], JSON3.write(Dict("id" => "pg_extra_ds")))
        @test resp.status == 201
        resp = HTTP.get("$base_url/datasets")
        @test resp.status == 200
        all_listed = JSON3.read(String(resp.body))
        total_datasets = all_listed.total
        @test total_datasets >= 2

        resp = HTTP.get("$base_url/datasets?limit=1")
        @test resp.status == 200
        first_dataset_page = JSON3.read(String(resp.body))
        @test length(first_dataset_page.datasets) == 1
        @test first_dataset_page.total == total_datasets

        resp = HTTP.get("$base_url/datasets?offset=$(total_datasets)&limit=10")
        @test resp.status == 200
        @test isempty(JSON3.read(String(resp.body)).datasets)

        # 8. GET /api/v1/jobs listing, with status/kind filters and offset/limit.
        resp = HTTP.post("$base_url/jobs/allknn", [], JSON3.write(Dict("dataset" => "cursor_ds", "k" => 3)))
        @test resp.status == 202
        job_id = JSON3.read(String(resp.body)).job_id

        resp = HTTP.get("$base_url/jobs?kind=allknn")
        @test resp.status == 200
        by_kind = JSON3.read(String(resp.body))
        @test any(j -> j.id == job_id, by_kind.jobs)

        resp = HTTP.get("$base_url/jobs?kind=neardup")
        @test resp.status == 200
        @test !any(j -> j.id == job_id, JSON3.read(String(resp.body)).jobs)

        resp = try
            HTTP.get("$base_url/jobs?status=not_a_real_status")
        catch e
            e.response
        end
        @test resp.status == 400

        # Drain the submitted job so it doesn't linger past the test.
        for _ in 1:180
            r = HTTP.get("$base_url/jobs/$job_id")
            s = string(JSON3.read(String(r.body)).status)
            s in ("completed", "failed") && break
            sleep(0.3)
        end
    end
end
