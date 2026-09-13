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

def create_dataset(id=None, index_type="searchgraph", distance="L2", join_group=None,
                    holds_metadata=False, key=None, meta_schema=None):
    body = _compact({
        "id": id, "index_type": index_type, "distance": distance,
        "join_group": join_group, "holds_metadata": holds_metadata or None,
        "key": key, "meta_schema": meta_schema,
    })
    return "POST", "/api/v1/datasets", body, None


def list_datasets(offset=0, limit=None):
    params = _compact({"offset": offset, "limit": limit})
    return "GET", "/api/v1/datasets", None, params


def get_dataset(id):
    return "GET", f"/api/v1/datasets/{id}", None, None


def delete_dataset(id):
    return "DELETE", f"/api/v1/datasets/{id}", None, None


def get_join_group(id):
    return "GET", f"/api/v1/datasets/{id}/join_group", None, None


def get_log(id, offset=0, limit=None):
    params = _compact({"offset": offset, "limit": limit})
    return "GET", f"/api/v1/datasets/{id}/log", None, params


# --- dataset operations --------------------------------------------------------------------

def append(index, items):
    return "POST", f"/api/v1/datasets/{index}/append", {"items": items}, None


def search(index, vector, k=10, filter=None, beamsearch_overrides=None, page_size=None):
    body = _compact({
        "vector": _as_vector(vector), "k": k, "filter": filter,
        "beamsearch_overrides": beamsearch_overrides, "page_size": page_size,
    })
    return "POST", f"/api/v1/datasets/{index}/search", body, None


def ftsearch(index, text, k=10):
    return "POST", f"/api/v1/datasets/{index}/ftsearch", {"text": text, "k": k}, None


def ftsearch_group(join_group, key, text, k=10):
    body = {"join_group": join_group, "key": key, "text": text, "k": k}
    return "POST", "/api/v1/search/group", body, None


def hybrid_search(dense_index, lexical_index, vector=None, text=None, k=10, alpha=None, filter=None):
    body = _compact({
        "dense_index": dense_index, "lexical_index": lexical_index,
        "vector": _as_vector(vector), "text": text, "k": k, "alpha": alpha, "filter": filter,
    })
    return "POST", "/api/v1/search/hybrid", body, None


def delete_item(index, doc_id):
    return "POST", f"/api/v1/datasets/{index}/delete", {"doc_id": doc_id}, None


def fetch(index, ids):
    return "POST", f"/api/v1/datasets/{index}/fetch", {"ids": ids}, None


def exists(index, ids):
    params = {"ids": ",".join(str(i) for i in ids)}
    return "GET", f"/api/v1/datasets/{index}/exists", None, params


def calibrate(index, minrecall=None, numqueries=None, ksearch=None, queries=None):
    body = _compact({
        "minrecall": minrecall, "numqueries": numqueries, "ksearch": ksearch, "queries": queries,
    })
    return "POST", f"/api/v1/datasets/{index}/calibrate", body, None


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
