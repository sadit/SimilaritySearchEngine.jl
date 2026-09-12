"""similarity_search: plain functions over similarity-search-serve's HTTP API.

A scripting library, not a full SDK (PLAN.md §8) -- write a script that calls these
functions, without hand-rolling Job-polling loops or cursor bookkeeping yourself.
"""

from .client import SimilaritySearchClient
from .async_client import AsyncSimilaritySearchClient
from .jobs import run_job, run_job_async
from .cursors import iter_cursor

__version__ = "0.2.0"

__all__ = [
    "SimilaritySearchClient",
    "AsyncSimilaritySearchClient",
    "run_job",
    "run_job_async",
    "iter_cursor",
]
