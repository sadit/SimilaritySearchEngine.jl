"""The one piece of real ergonomic value PLAN.md §8.1 calls out: a `run_job` helper that
submits a Job and blocks until it reaches a terminal state, instead of every script
hand-writing the same submit-then-poll loop.
"""

import time
import asyncio


def run_job(client, kind, command=None, poll_interval=1.0, **params):
    """Submits a `kind` job (friendly `params`, e.g. `dataset=`/`k=`/`epsilon=`/`queries=`
    for allknn/fft/neardup/hsp, `dataset=` for dump, `bundle=`/`dataset=` for load -- or an
    explicit `command=[...]` list, the server's raw escape hatch), then polls
    `client.get_job(job_id)` every `poll_interval` seconds until `status` is `completed` or
    `failed`. Returns the job record (`client.get_job`'s own response) on success; raises
    `RuntimeError` on failure. Fetch the actual result payload afterward with
    `client.get_job_result(job_id)` -- its shape (JSONL vs. a dump/load JSON pointer)
    depends on `kind`, so this doesn't guess at it for you.
    """
    submitted = client.submit_job(kind, command=command, **params)
    job_id = submitted["job_id"]

    while True:
        job = client.get_job(job_id)
        status = job["status"]
        if status == "completed":
            return job
        if status == "failed":
            raise RuntimeError(f"job {job_id} ({kind}) failed: {job['job'].get('error')}")
        time.sleep(poll_interval)


async def run_job_async(client, kind, command=None, poll_interval=1.0, **params):
    """Async counterpart of `run_job`, for `AsyncSimilaritySearchClient` -- lets a script
    run several jobs concurrently via `asyncio.gather(*(run_job_async(...) for ...))`
    instead of blocking on each one in turn.
    """
    submitted = await client.submit_job(kind, command=command, **params)
    job_id = submitted["job_id"]

    while True:
        job = await client.get_job(job_id)
        status = job["status"]
        if status == "completed":
            return job
        if status == "failed":
            raise RuntimeError(f"job {job_id} ({kind}) failed: {job['job'].get('error')}")
        await asyncio.sleep(poll_interval)
