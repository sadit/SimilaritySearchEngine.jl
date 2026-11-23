"""Pure request-shape builders for every SimilaritySearchServer HTTP endpoint.

No HTTP calls happen here -- each function just returns
``(method, path, json_body_or_None, query_params_or_None)`` for one endpoint, built from
plain keyword arguments matching the real wire shapes in ``src/server.jl``'s route table
(read directly against a running implementation, not against PLAN.md's original
aspirational text -- see the package README for what that means in practice, e.g. no Avro
binary mode, no long-query 429 fallback: neither exists server-side today).

Both ``client.py`` (sync) and ``async_client.py`` (async) call these same functions, so the
two clients can never drift apart on what a request actually looks like.
"""

from urllib.parse import quote as _quote

try:
    import numpy as _np
except ImportError:  # pragma: no cover - numpy is an optional convenience only
    _np = None


def _as_vector(vector):
    """Accepts a numpy array or a plain sequence, returns a plain ``list[float]``."""
    if vector is None:
        return None
    if _np is not None and isinstance(vector, _np.ndarray):
        return [float(x) for x in vector.tolist()]
    return [float(x) for x in vector]


def _compact(d):
    """Drops ``None``-valued keys -- the server treats an absent field and an explicit
    ``null`` differently in a few places (e.g. optional job params), so omitting is safer
    than sending ``null`` everywhere."""
    return {k: v for k, v in d.items() if v is not None}


# --- health -------------------------------------------------------------------------

def healthz():
    return "GET", "/healthz", None, None


def readyz():
    return "GET", "/readyz", None, None


def metrics():
    return "GET", "/metrics", None, None


# --- datasets -------------------------------------------------------------------------

def create_dataset(id=None, index_type="searchgraph", distance="L2", meta_schema=None,
                    partition_by=None, partition_graph_min=None):
    body = _compact({
        "id": id, "index_type": index_type, "distance": distance,
        "meta_schema": meta_schema, "partition_by": partition_by,
        "partition_graph_min": partition_graph_min,
    })
    return "POST", "/api/v1/datasets", body, None


def list_datasets(offset=0, limit=None):
    params = _compact({"offset": offset, "limit": limit})
    return "GET", "/api/v1/datasets", None, params


def get_dataset(id):
    return "GET", f"/api/v1/datasets/{id}", None, None


def delete_dataset(id):
    return "DELETE", f"/api/v1/datasets/{id}", None, None


def get_log(id, offset=0, limit=None):
    params = _compact({"offset": offset, "limit": limit})
    return "GET", f"/api/v1/datasets/{id}/log", None, params


# --- dataset operations --------------------------------------------------------------------

def append(index, items):
    return "POST", f"/api/v1/datasets/{index}/append", {"items": items}, None


def search(index, vector, k=10, filter=None, beamsearch_overrides=None, page_size=None, partition=None):
    body = _compact({
        "vector": _as_vector(vector), "k": k, "filter": filter,
        "beamsearch_overrides": beamsearch_overrides, "page_size": page_size,
        "partition": partition,
    })
    return "POST", f"/api/v1/datasets/{index}/search", body, None


def ftsearch(index, text, k=10, partition=None, filter=None):
    body = _compact({"text": text, "k": k, "partition": partition, "filter": filter})
    return "POST", f"/api/v1/datasets/{index}/ftsearch", body, None


def get_partitions(id):
    return "GET", f"/api/v1/datasets/{id}/partitions", None, None


def hybrid_search(dense_index, lexical_index, vector=None, text=None, k=10, alpha=None, filter=None):
    body = _compact({
        "dense_index": dense_index, "lexical_index": lexical_index,
        "vector": _as_vector(vector), "text": text, "k": k, "alpha": alpha, "filter": filter,
    })
    return "POST", "/api/v1/search/hybrid", body, None


def delete_item(index, doc_id):
    return "POST", f"/api/v1/datasets/{index}/delete", {"_id": doc_id}, None


def fetch(index, ids):
    return "POST", f"/api/v1/datasets/{index}/fetch", {"ids": ids}, None


def exists(index, ids):
    params = {"ids": ",".join(str(i) for i in ids)}
    return "GET", f"/api/v1/datasets/{index}/exists", None, params


def calibrate(index, minrecall=None, numqueries=None, ksearch=None, queries=None, partition=None):
    body = _compact({
        "minrecall": minrecall, "numqueries": numqueries, "ksearch": ksearch, "queries": queries,
        "partition": partition,
    })
    return "POST", f"/api/v1/datasets/{index}/calibrate", body, None


# --- followers (PLAN.md §5.12) ------------------------------------------------------

def _follower_path(index, follower, rest=""):
    """The path of a route of a follower. ``follower`` is a name, or ``name=value`` for one
    partition; it is escaped as one path segment."""
    return f"/api/v1/datasets/{index}/followers/{_quote(follower, safe='')}{rest}"


def create_follower(index, name, kind, source, disjoint=None, graph_min=None, distance=None, policy=None):
    """``kind`` is ``partition``, ``keyword`` or ``Plugin.name``; ``source`` is ``keywords``,
    ``refs``, ``doc_id``, ``meta:<field>`` or ``Plugin.name``. A kind or a key source of a plugin
    with options is a dict, ``{"name": "Decades.decade", "field": "year"}``. ``policy`` is
    ``on_index``, ``on_write``, ``manual`` or ``{"periodic": {"seconds": 60, "items": 1000}}``."""
    body = _compact({
        "name": name, "kind": kind, "source": source, "disjoint": disjoint,
        "graph_min": graph_min, "distance": distance, "policy": policy,
    })
    return "POST", f"/api/v1/datasets/{index}/followers", body, None


def list_followers(index):
    return "GET", f"/api/v1/datasets/{index}/followers", None, None


def get_follower(index, name):
    return "GET", _follower_path(index, name), None, None


def delete_follower(index, name):
    return "DELETE", _follower_path(index, name), None, None


def sync_follower(index, name):
    return "POST", _follower_path(index, name, "/sync"), None, None


def rebuild_follower(index, name):
    return "POST", _follower_path(index, name, "/rebuild"), None, None


def follower_partitions(index, name):
    return "GET", _follower_path(index, name, "/partitions"), None, None


def follower_search(index, follower, vector=None, keys=None, k=10, t=None, filter=None, minrecall=None,
                    page_size=None):
    """A vector in one partition (``follower="name=value"``), or keys in a keyword follower
    (``follower="name"``) -- one of ``vector`` and ``keys``."""
    if (vector is None) == (keys is None):
        raise ValueError("follower_search takes a vector (a partition, name=value) or keys (a keyword follower), not both")
    body = _compact({
        "vector": _as_vector(vector), "keys": None if keys is None else list(keys), "k": k, "t": t,
        "filter": filter, "minrecall": minrecall, "page_size": page_size,
    })
    return "POST", _follower_path(index, follower, "/search"), body, None


def follower_ftsearch(index, follower, text, k=10, filter=None):
    body = _compact({"text": text, "k": k, "filter": filter})
    return "POST", _follower_path(index, follower, "/ftsearch"), body, None


def follower_calibrate(index, follower, minrecall=None, numqueries=None, ksearch=None):
    body = _compact({"minrecall": minrecall, "numqueries": numqueries, "ksearch": ksearch})
    return "POST", _follower_path(index, follower, "/calibrate"), body, None


def posting_list(index, key, follower=None, limit=None):
    """The ids of the items that hold ``key``: a token of a text dataset, a dimension of a sparse
    dataset, or with ``follower`` a key of a keyword follower. A whole list of a frequent key in
    a large dataset can hold most of its items; ``limit`` returns the first ``limit`` ids."""
    params = _compact({"key": str(key), "limit": limit})
    path = f"/api/v1/datasets/{index}/postings" if follower is None else _follower_path(index, follower, "/postings")
    return "GET", path, None, params


def migrate_partitions(index, name, disjoint):
    """``disjoint`` is required: the migration does not choose the kind of partition."""
    if not isinstance(disjoint, bool):
        raise ValueError("migrate_partitions needs disjoint=True or disjoint=False")
    return "POST", f"/api/v1/datasets/{index}/migrate-partitions", {"name": name, "disjoint": disjoint}, None


def get_group(index):
    return "GET", f"/api/v1/datasets/{index}/group", None, None


def list_plugins():
    """The plugins that the server loaded, with the wire names of their key sources and kinds."""
    return "GET", "/api/v1/plugins", None, None


# --- jobs -------------------------------------------------------------------------

def submit_job(kind, command=None, **params):
    """Either an explicit ``command`` list (the server's raw escape hatch) or the
    friendly per-kind params body (``dataset``/``k``/``epsilon``/``queries`` for
    allknn/fft/neardup/hsp, or ``dataset``/``bundle`` for dump/load) -- never both."""
    body = {"command": command} if command is not None else _compact(params)
    return "POST", f"/api/v1/jobs/{kind}", body, None


def get_job(job_id):
    return "GET", f"/api/v1/jobs/{job_id}", None, None


def get_job_result(job_id):
    return "GET", f"/api/v1/jobs/{job_id}/result", None, None


def block_job(job_id):
    return "POST", f"/api/v1/jobs/{job_id}/block", None, None


def resume_job(job_id):
    return "POST", f"/api/v1/jobs/{job_id}/resume", None, None


def kill_job(job_id):
    return "POST", f"/api/v1/jobs/{job_id}/kill", None, None


def cancel_job(job_id):
    return "DELETE", f"/api/v1/jobs/{job_id}", None, None


def list_jobs(status=None, kind=None, offset=0, limit=None):
    params = _compact({"status": status, "kind": kind, "offset": offset, "limit": limit})
    return "GET", "/api/v1/jobs", None, params


# --- cursors ------------------------------------------------------------------------

def poll_cursor(cursor_id, limit=None):
    params = _compact({"limit": limit})
    return "GET", f"/api/v1/cursors/{cursor_id}", None, params


# --- admin -------------------------------------------------------------------------

def create_token(user="anonymous", permissions=None, expires_at=None):
    body = _compact({"user": user, "permissions": permissions or [], "expires_at": expires_at})
    return "POST", "/api/v1/admin/tokens", body, None


def list_tokens():
    return "GET", "/api/v1/admin/tokens", None, None


def prune_tokens():
    return "POST", "/api/v1/admin/tokens/prune", None, None


def revoke_token(token):
    return "DELETE", f"/api/v1/admin/tokens/{token}", None, None


def jobs_gc(retention_seconds=86400):
    return "POST", "/api/v1/admin/jobs/gc", {"retention_seconds": retention_seconds}, None


def unload_dataset(id):
    return "POST", f"/api/v1/admin/datasets/{id}/unload", None, None


def reload_dataset(id):
    return "POST", f"/api/v1/admin/datasets/{id}/reload", None, None
