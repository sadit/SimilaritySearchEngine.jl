"""Async twin of `client.py` -- same method surface, same `_payloads` builders, just
`httpx.AsyncClient` + `async def` throughout. Exists so a script can fire off several
Jobs/searches concurrently (`asyncio.gather(...)`) instead of blocking on each one in turn
-- the actual ergonomic payoff of choosing `httpx` (PLAN.md §8.1) over a sync-only library.
"""

import httpx

from . import _payloads as p
from .client import _RAW_TEXT_PATHS


class AsyncSimilaritySearchClient:
    """Async counterpart of `SimilaritySearchClient`. Use as an async context manager
    (`async with AsyncSimilaritySearchClient(...) as c: ...`) or call `await c.aclose()`.
    """

    def __init__(self, base_url="http://127.0.0.1:8080", token=None, timeout=30.0):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)

    async def aclose(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def _call(self, method, path, json=None, params=None):
        resp = await self._client.request(method, path, json=json, params=params)
        resp.raise_for_status()
        if path in _RAW_TEXT_PATHS or path.endswith("/result"):
            return resp.text
        if not resp.content:
            return None
        return resp.json()

    # --- health ---------------------------------------------------------------------

    async def healthz(self):
        return await self._call(*p.healthz())

    async def readyz(self):
        return await self._call(*p.readyz())

    async def metrics(self):
        return await self._call(*p.metrics())

    # --- datasets ---------------------------------------------------------------------

    async def create_dataset(self, id=None, index_type="searchgraph", distance="L2", meta_schema=None,
                              partition_by=None, partition_graph_min=None, storage=None, max_k=None):
        return await self._call(*p.create_dataset(id, index_type, distance, meta_schema, partition_by,
                                                partition_graph_min, storage, max_k))

    async def list_datasets(self, offset=0, limit=None):
        return await self._call(*p.list_datasets(offset, limit))

    async def get_dataset(self, id):
        return await self._call(*p.get_dataset(id))

    async def delete_dataset(self, id):
        return await self._call(*p.delete_dataset(id))

    async def get_log(self, id, offset=0, limit=None):
        return await self._call(*p.get_log(id, offset, limit))

    # --- dataset operations ----------------------------------------------------------------

    async def append(self, index, items):
        return await self._call(*p.append(index, items))

    async def search(self, index, vector, k=10, filter=None, beamsearch_overrides=None, partition=None):
        return await self._call(*p.search(index, vector, k, filter, beamsearch_overrides, partition))

    async def ftsearch(self, index, text, k=10, partition=None, filter=None):
        return await self._call(*p.ftsearch(index, text, k, partition, filter))

    async def get_partitions(self, id):
        return await self._call(*p.get_partitions(id))

    async def hybrid_search(self, dense_index, lexical_index, vector=None, text=None, k=10, alpha=None, filter=None):
        return await self._call(*p.hybrid_search(dense_index, lexical_index, vector, text, k, alpha, filter))

    async def delete_item(self, index, doc_id):
        return await self._call(*p.delete_item(index, doc_id))

    async def fetch(self, index, ids):
        return await self._call(*p.fetch(index, ids))

    async def exists(self, index, ids):
        return await self._call(*p.exists(index, ids))

    async def calibrate(self, index, minrecall=None, numqueries=None, ksearch=None, queries=None, partition=None):
        return await self._call(*p.calibrate(index, minrecall, numqueries, ksearch, queries, partition))

    # --- followers (PLAN.md §5.12) ------------------------------------------------------

    async def create_follower(self, index, name, kind, source, disjoint=None, graph_min=None, distance=None, policy=None):
        return await self._call(*p.create_follower(index, name, kind, source, disjoint, graph_min, distance, policy))

    async def list_followers(self, index):
        return await self._call(*p.list_followers(index))

    async def get_follower(self, index, name):
        return await self._call(*p.get_follower(index, name))

    async def delete_follower(self, index, name):
        return await self._call(*p.delete_follower(index, name))

    async def sync_follower(self, index, name):
        return await self._call(*p.sync_follower(index, name))

    async def rebuild_follower(self, index, name):
        return await self._call(*p.rebuild_follower(index, name))

    async def follower_partitions(self, index, name):
        return await self._call(*p.follower_partitions(index, name))

    async def follower_search(self, index, follower, vector=None, keys=None, k=10, t=None, filter=None, minrecall=None):
        return await self._call(*p.follower_search(index, follower, vector, keys, k, t, filter, minrecall))

    async def follower_ftsearch(self, index, follower, text, k=10, filter=None):
        return await self._call(*p.follower_ftsearch(index, follower, text, k, filter))

    async def follower_calibrate(self, index, follower, minrecall=None, numqueries=None, ksearch=None):
        return await self._call(*p.follower_calibrate(index, follower, minrecall, numqueries, ksearch))

    async def posting_list(self, index, key, follower=None, limit=None):
        return await self._call(*p.posting_list(index, key, follower, limit))

    async def migrate_partitions(self, index, name, disjoint):
        return await self._call(*p.migrate_partitions(index, name, disjoint))

    async def get_group(self, index):
        return await self._call(*p.get_group(index))

    async def list_plugins(self):
        return await self._call(*p.list_plugins())

    # --- jobs -------------------------------------------------------------------------

    async def submit_job(self, kind, command=None, **params):
        return await self._call(*p.submit_job(kind, command, **params))

    async def get_job(self, job_id):
        return await self._call(*p.get_job(job_id))

    async def get_job_result(self, job_id):
        return await self._call(*p.get_job_result(job_id))

    async def block_job(self, job_id):
        return await self._call(*p.block_job(job_id))

    async def resume_job(self, job_id):
        return await self._call(*p.resume_job(job_id))

    async def kill_job(self, job_id):
        return await self._call(*p.kill_job(job_id))

    async def cancel_job(self, job_id):
        return await self._call(*p.cancel_job(job_id))

    async def list_jobs(self, status=None, kind=None, offset=0, limit=None):
        return await self._call(*p.list_jobs(status, kind, offset, limit))

    # --- admin ------------------------------------------------------------------------

    async def create_token(self, user="anonymous", permissions=None, expires_at=None):
        return await self._call(*p.create_token(user, permissions, expires_at))

    async def list_tokens(self):
        return await self._call(*p.list_tokens())

    async def prune_tokens(self):
        return await self._call(*p.prune_tokens())

    async def revoke_token(self, token):
        return await self._call(*p.revoke_token(token))

    async def jobs_gc(self, retention_seconds=86400):
        return await self._call(*p.jobs_gc(retention_seconds))

    async def unload_dataset(self, id):
        return await self._call(*p.unload_dataset(id))

    async def reload_dataset(self, id, max_k=None):
        return await self._call(*p.reload_dataset(id, max_k))
