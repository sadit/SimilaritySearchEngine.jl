"""
    SearchPages

The pages of one search, as returned by [`search_pages`](@ref). Iterating it gives one
`Vector{SearchResult}` per page, in order. The first page arrives with the search. Each later
page is one request to the cursor of the server (`GET /api/v1/cursors/{id}`).

`total` is the number of results of the search, and `cursor_id` is the id of the cursor.

A `SearchPages` can be iterated once. The server closes the cursor after its last page, so a
second pass raises [`StateConflict`](@ref) when it asks for its second page. The cursor also expires 300 seconds after the
search that created it, whether it is read or not. A page requested after that raises
[`StateConflict`](@ref) too.
"""
struct SearchPages
    conn::Connection
    cursor_id::String
    total::Int
    first::Vector{SearchResult}
    exhausted::Bool
end

Base.IteratorSize(::Type{SearchPages}) = Base.SizeUnknown()
Base.eltype(::Type{SearchPages}) = Vector{SearchResult}

Base.iterate(p::SearchPages) = p.total == 0 ? nothing : (p.first, p.exhausted)

function Base.iterate(p::SearchPages, exhausted::Bool)
    exhausted && return nothing
    resp = request(p.conn, "GET", "/api/v1/cursors/" * HTTP.escapeuri(p.cursor_id))
    return (_results(resp), Bool(resp[:exhausted]))
end

Base.show(io::IO, p::SearchPages) = print(io, "SearchPages(", p.total, " results, cursor ", repr(p.cursor_id), ")")

"""
    search_pages(h::RemoteProject, vector, k=10; page_size, filter=nothing, partition=nothing,
                 beamsearch_overrides=nothing) -> SearchPages

The `k` nearest neighbors of `vector`, as [`search`](@ref) finds them, in pages of `page_size`
results. The server computes all `k` results at once and keeps them in a cursor, and each page
is one request. It is useful for a large `k`, where one response would be large.

```julia
for page in search_pages(h, q, 10_000; page_size=500)
    # page::Vector{SearchResult}
end
```

The engine has no counterpart: it returns every result at once. The server pages only dense and
sparse searches, so there is no counterpart for [`ftsearch`](@ref).
"""
function search_pages(h::RemoteProject, vector::AbstractVector{<:Real}, k::Integer=10; page_size::Integer,
                      filter=nothing, partition::Union{Nothing,AbstractString}=nothing, beamsearch_overrides=nothing)
    page_size > 0 || throw(ArgumentError("page_size must be positive; got $page_size"))
    body = _search_body(vector, k, filter, partition, beamsearch_overrides)
    body["page_size"] = Int(page_size)
    resp = request(h.conn, "POST", dataset_path(h.id, "/search"); body)
    return SearchPages(h.conn, string(resp[:cursor_id]), Int(resp[:total]), _results(resp), Bool(resp[:exhausted]))
end
