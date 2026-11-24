"""similarity_search: plain functions over similarity-search-serve's HTTP API.

A scripting library, not a full SDK (CHANGELOG.md §8) -- write a script that calls these
functions, without hand-rolling Job-polling loops yourself. A search answers its k
results in one response.
"""

from .client import SimilaritySearchClient
from .async_client import AsyncSimilaritySearchClient
from .jobs import run_job, run_job_async
from .fields import ftsearch_fields, ftsearch_fields_async

__version__ = "0.2.0"

__all__ = [
    "SimilaritySearchClient",
    "AsyncSimilaritySearchClient",
    "run_job",
    "run_job_async",
    "ftsearch_fields",
    "ftsearch_fields_async",
]
