"""Synchronous client for SimilaritySearchServer's HTTP API.

Plain functions over the wire, not a class-per-endpoint SDK (PLAN.md §8.1) -- every method
here just delegates to a matching builder in ``_payloads.py`` then makes the call. No
local workdir access anywhere: this only ever talks HTTP to a running
``similarity-search-serve``.
"""

import httpx

from . import _payloads as p

_RAW_TEXT_PATHS = {"/metrics"}  # Prometheus text, not JSON -- everything else is JSON


class SimilaritySearchClient:
    """A thin `httpx.Client` wrapper. ``token``, if given, is sent as `Authorization:
    Bearer {token}` on every request -- recorded by the server's `op_log` telemetry, but
    NOT validated anywhere server-side yet (no request-auth enforcement exists in this
    codebase as of this writing). Use as a context manager to close the underlying
    connection pool, or call `.close()` directly.
    """

    def __init__(self, base_url="http://127.0.0.1:8080", token=None, timeout=30.0):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _call(self, method, path, json=None, params=None):
        resp = self._client.request(method, path, json=json, params=params)
        resp.raise_for_status()
        if path in _RAW_TEXT_PATHS or path.endswith("/result"):
            # /result's content is whatever a heavy job's CLI subprocess wrote (JSONL for
            # allknn/fft/neardup/hsp, a JSON pointer dict for dump) -- not safe to assume
            # a single JSON document here, so return raw text and let the caller decide.
            return resp.text
        if not resp.content:
            return None
        return resp.json()

    # --- health ---------------------------------------------------------------------

    def healthz(self):
        return self._call(*p.healthz())

    def readyz(self):
        return self._call(*p.readyz())

    def metrics(self):
        return self._call(*p.metrics())

    # --- datasets ---------------------------------------------------------------------

    def create_dataset(self, id=None, index_type="searchgraph", distance="L2", meta_schema=None,
                        partition_by=None, partition_graph_min=None, storage=None):
        return self._call(*p.create_dataset(id, index_type, distance, meta_schema, partition_by,
                                                partition_graph_min, storage))

    def list_datasets(self, offset=0, limit=None):
        return self._call(*p.list_datasets(offset, limit))

    def get_dataset(self, id):
        return self._call(*p.get_dataset(id))

    def delete_dataset(self, id):
        return self._call(*p.delete_dataset(id))

    def get_log(self, id, offset=0, limit=None):
        return self._call(*p.get_log(id, offset, limit))

    # --- dataset operations ----------------------------------------------------------------

    def append(self, index, items):
        return self._call(*p.append(index, items))

    def search(self, index, vector, k=10, filter=None, beamsearch_overrides=None, page_size=None,
               partition=None):
        return self._call(*p.search(index, vector, k, filter, beamsearch_overrides, page_size, partition))

    def ftsearch(self, index, text, k=10, partition=None, filter=None):
        return self._call(*p.ftsearch(index, text, k, partition, filter))

    def get_partitions(self, id):
        return self._call(*p.get_partitions(id))

    def hybrid_search(self, dense_index, lexical_index, vector=None, text=None, k=10, alpha=None, filter=None):
        return self._call(*p.hybrid_search(dense_index, lexical_index, vector, text, k, alpha, filter))

    def delete_item(self, index, doc_id):
        return self._call(*p.delete_item(index, doc_id))

    def fetch(self, index, ids):
        return self._call(*p.fetch(index, ids))

    def exists(self, index, ids):
        return self._call(*p.exists(index, ids))

    def calibrate(self, index, minrecall=None, numqueries=None, ksearch=None, queries=None, partition=None):
        return self._call(*p.calibrate(index, minrecall, numqueries, ksearch, queries, partition))

    # --- followers (PLAN.md §5.12) ------------------------------------------------------

    def create_follower(self, index, name, kind, source, disjoint=None, graph_min=None, distance=None, policy=None):
        return self._call(*p.create_follower(index, name, kind, source, disjoint, graph_min, distance, policy))

    def list_followers(self, index):
        return self._call(*p.list_followers(index))

    def get_follower(self, index, name):
        return self._call(*p.get_follower(index, name))

    def delete_follower(self, index, name):
        return self._call(*p.delete_follower(index, name))

    def sync_follower(self, index, name):
        return self._call(*p.sync_follower(index, name))

    def rebuild_follower(self, index, name):
        return self._call(*p.rebuild_follower(index, name))

    def follower_partitions(self, index, name):
        return self._call(*p.follower_partitions(index, name))

    def follower_search(self, index, follower, vector=None, keys=None, k=10, t=None, filter=None, minrecall=None,
                        page_size=None):
        return self._call(*p.follower_search(index, follower, vector, keys, k, t, filter, minrecall, page_size))

    def follower_ftsearch(self, index, follower, text, k=10, filter=None):
        return self._call(*p.follower_ftsearch(index, follower, text, k, filter))

    def follower_calibrate(self, index, follower, minrecall=None, numqueries=None, ksearch=None):
        return self._call(*p.follower_calibrate(index, follower, minrecall, numqueries, ksearch))

    def posting_list(self, index, key, follower=None, limit=None):
        return self._call(*p.posting_list(index, key, follower, limit))

    def migrate_partitions(self, index, name, disjoint):
        return self._call(*p.migrate_partitions(index, name, disjoint))

    def get_group(self, index):
        return self._call(*p.get_group(index))

    def list_plugins(self):
        return self._call(*p.list_plugins())

    # --- jobs -------------------------------------------------------------------------

    def submit_job(self, kind, command=None, **params):
        return self._call(*p.submit_job(kind, command, **params))

    def get_job(self, job_id):
        return self._call(*p.get_job(job_id))

    def get_job_result(self, job_id):
        return self._call(*p.get_job_result(job_id))

    def block_job(self, job_id):
        return self._call(*p.block_job(job_id))

    def resume_job(self, job_id):
        return self._call(*p.resume_job(job_id))

    def kill_job(self, job_id):
        return self._call(*p.kill_job(job_id))

    def cancel_job(self, job_id):
        return self._call(*p.cancel_job(job_id))

    def list_jobs(self, status=None, kind=None, offset=0, limit=None):
        return self._call(*p.list_jobs(status, kind, offset, limit))

    # --- cursors ------------------------------------------------------------------

    def poll_cursor(self, cursor_id, limit=None):
        return self._call(*p.poll_cursor(cursor_id, limit))

    # --- admin ------------------------------------------------------------------------

    def create_token(self, user="anonymous", permissions=None, expires_at=None):
        return self._call(*p.create_token(user, permissions, expires_at))

    def list_tokens(self):
        return self._call(*p.list_tokens())

    def prune_tokens(self):
        return self._call(*p.prune_tokens())

    def revoke_token(self, token):
        return self._call(*p.revoke_token(token))

    def jobs_gc(self, retention_seconds=86400):
        return self._call(*p.jobs_gc(retention_seconds))

    def unload_dataset(self, id):
        return self._call(*p.unload_dataset(id))

    def reload_dataset(self, id):
        return self._call(*p.reload_dataset(id))
