"""SimilaritySearchServer stress/smoke client.

A real, end-to-end exercise of `similarity_search`'s full API surface against a live
`similarity-search-serve` -- doubles as this package's own correctness check (PLAN.md
§8.2's "ship a worked example of using the library" ask) and as a quick way to confirm a
running server is actually healthy end-to-end, not just that `/healthz` returns 200.
"""

import argparse
import json
import os
import sys

from similarity_search import SimilaritySearchClient, run_job, iter_cursor


def _load_docs(n=30):
    """Real dense+text docs from this repo's own test fixtures (128-dim LSI vectors,
    see PLAN.md's chunk on regenerating test/data/*.jsonl) -- not synthetic random data,
    so search/hybrid_search actually return meaningful, checkable hits."""
    # `server/python/` -> `server/` -> the repository root, where the fixtures live: they
    # belong to the engine's own test data and are shared by both packages.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(repo_root, "test", "data", "frankenstein.jsonl")
    docs = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            docs.append(json.loads(line))
    return docs


def run(base_url):
    docs = _load_docs()
    print(f"Loaded {len(docs)} docs from test fixtures.")

    with SimilaritySearchClient(base_url) as c:
        # 1. health
        print("readyz:", c.readyz())

        # 2. dense dataset: create, append, search
        c.create_dataset(id="stress_dense", index_type="searchgraph", distance="L2")
        print("Created dense dataset 'stress_dense'.")
        resp = c.append("stress_dense", docs)
        print("Appended:", resp)
        results = c.search("stress_dense", docs[0]["vector"], k=5)["results"]
        assert results, "search returned no results"
        assert results[0]["doc_id"] == docs[0]["doc_id"], "self-search didn't return itself first"
        print(f"Search top hit: {results[0]}")

        # 3. text dataset: create, append, ftsearch
        c.create_dataset(id="stress_text", index_type="bm25_invfile")
        c.append("stress_text", docs)
        ft_results = c.ftsearch("stress_text", "Chapter", k=5)["results"]
        assert ft_results, "ftsearch returned no results"
        print(f"ftsearch hits: {len(ft_results)}")

        # 4. hybrid_search over both (same doc_ids, since the same docs were appended
        # to both in the same order)
        hybrid = c.hybrid_search("stress_dense", "stress_text", vector=docs[0]["vector"], text="Chapter", k=5)
        assert hybrid["results"], "hybrid_search returned no results"
        print(f"hybrid_search results: {len(hybrid['results'])}")

        # 5. fetch / delete / exists
        fetched = c.fetch("stress_dense", [docs[0]["doc_id"]])
        assert fetched["results"], "fetch found nothing for a known id"
        c.delete_item("stress_dense", 1)
        exists = c.exists("stress_dense", ["1"])
        assert exists["results"][0]["deleted"] is True
        print("fetch/delete/exists OK.")

        # 6. calibrate (searchgraph only)
        baseline = c.calibrate("stress_dense", numqueries=8)
        print("calibrate baseline:", baseline["baseline"])

        # 7. a real heavy job via run_job (allknn)
        job = run_job(c, "allknn", dataset="stress_dense", k=5)
        job_id = job["job"]["id"]
        print(f"allknn job {job_id} completed.")
        result_text = c.get_job_result(job_id)
        print("allknn result (first line):", result_text.splitlines()[0])

        # 8. cursors: paginate through a search's remaining pages
        page1 = c.search("stress_dense", docs[0]["vector"], k=len(docs), page_size=5)
        print(f"page 1: {len(page1['results'])} results, cursor {page1['cursor_id']}")
        for page in iter_cursor(c, page1["cursor_id"], limit=5):
            print(f"  next page: {len(page)} results")

        # 9. admin: tokens, unload/reload, jobs gc
        token = c.create_token(user="stress-test", permissions=["search"])["token"]
        assert any(t["token"] == token for t in c.list_tokens()["tokens"])
        c.revoke_token(token)
        print("token create/list/revoke OK.")

        c.unload_dataset("stress_text")
        c.reload_dataset("stress_text")
        print("unload/reload OK.")

        gc = c.jobs_gc(retention_seconds=0)
        print("jobs_gc:", gc)

        # cleanup
        c.delete_dataset("stress_dense")
        c.delete_dataset("stress_text")
        print("Cleaned up. Stress test complete.")


def main():
    parser = argparse.ArgumentParser(description="SimilaritySearchServer stress client")
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="Server base URL")
    args = parser.parse_args()
    run(args.url)


if __name__ == "__main__":
    main()
