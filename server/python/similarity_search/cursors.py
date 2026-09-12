"""Hides `cursor_id`/`exhausted` bookkeeping behind a plain generator (PLAN.md §8.1)."""


def iter_cursor(client, cursor_id, limit=None):
    """Yields each *remaining* page's `results` list in turn until the cursor is
    exhausted -- the page-1 result already embedded in the response that gave you
    `cursor_id` in the first place (e.g. `search(..., page_size=N)`) is NOT re-yielded
    here; the server's own cursor already advanced past it the moment it was created
    (`Cursors.create_cursor!` immediately materializes page 1 via one `poll_cursor!`
    call), so this generator picks up from page 2 onward.

    `client` may be a sync `SimilaritySearchClient` -- for `AsyncSimilaritySearchClient`,
    poll `await client.poll_cursor(...)` directly in an `async for`-friendly loop instead,
    since a plain generator can't `await`.
    """
    while True:
        page = client.poll_cursor(cursor_id, limit=limit)
        yield page["results"]
        if page["exhausted"]:
            return
