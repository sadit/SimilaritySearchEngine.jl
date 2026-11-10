"""Search several text datasets with one query, one dataset per field of a record.

The server has no route for this: join groups were removed on 2026-11-10 (PLAN.md §0, item
11). Each field of a record is a text dataset of its own, appended with the same `doc_id` in
every field dataset, and the caller keeps the map from field name to dataset id. These
helpers send one `ftsearch` request per dataset and return the results by field name.
"""

import asyncio


def ftsearch_fields(client, fields, text, k=10, partition=None):
    """Searches `text` in every dataset of `fields`, a dict from field name to dataset id,
    with a `SimilaritySearchClient`. Sends the requests one after the other. Returns a dict
    from field name to the `results` list of that dataset's `ftsearch` response. Each result
    carries the `doc_id` the item was appended with, which identifies the record across
    fields. `partition`, if given, is sent to every dataset. The first request that fails
    raises `httpx.HTTPStatusError`.
    """
    return {name: client.ftsearch(dataset, text, k, partition)["results"]
            for name, dataset in fields.items()}


async def ftsearch_fields_async(client, fields, text, k=10, partition=None):
    """Same as `ftsearch_fields`, with an `AsyncSimilaritySearchClient`: sends all the
    requests at once and waits for all of them. The time is that of the slowest request,
    instead of the sum of all of them.
    """
    names = list(fields)
    responses = await asyncio.gather(
        *(client.ftsearch(fields[name], text, k, partition) for name in names))
    return {name: resp["results"] for name, resp in zip(names, responses)}
