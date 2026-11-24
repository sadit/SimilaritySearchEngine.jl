"""
    append_items!(h::RemoteProject, items) -> Int

Appends `items` (a vector of [`DenseItem`](@ref), [`SparseItem`](@ref) or [`TextItem`](@ref),
or one item) in one request, and returns the number of items the server inserted.

The server indexes the items before it answers, so they are searchable when this call returns.
[`index!`](@ref) is therefore not needed, unlike in the engine.
"""
function append_items!(h::RemoteProject, items::AbstractVector{<:AbstractItem})
    resp = request(h.conn, "POST", dataset_path(h.id, "/append"); body=Dict("items" => map(wire_item, items)))
    return Int(resp[:inserted])
end

append_items!(h::RemoteProject, item::AbstractItem) = append_items!(h, [item])

"""
    index!(h::RemoteProject) -> nothing

Does nothing: the server indexes the items of every [`append_items!`](@ref) call before it
answers. It exists so that a script written for the engine runs unchanged.
"""
index!(::RemoteProject) = nothing

"""
    search(h::RemoteProject, vector, k=10; filter=nothing, partition=nothing,
           beamsearch_overrides=nothing) -> Vector{SearchResult}

The `k` nearest neighbors of `vector` in a dense or sparse dataset, by ascending distance
(`POST /api/v1/datasets/{id}/search`).

`filter` is a filter object of the server, for example `Dict("year" => Dict("gte" => 2020))`,
and not the `(record, meta) -> Bool` function the engine takes: the server cannot run a
function of the client. The server manual describes the filter objects. With a filter, the
server reads a limited number of candidates, so the result can have fewer than `k` items.

`partition` searches one partition of a dense dataset. `beamsearch_overrides` replaces fields
of the beam search of the global graph for this request, for example `(bsize=64,)`. It cannot
be combined with `partition`.

The engine's `minrecall`, `candidates` and `stats` keywords do not exist here, because the
server does not accept them.
"""
function search(h::RemoteProject, vector::AbstractVector{<:Real}, k::Integer=10; filter=nothing,
                partition::Union{Nothing,AbstractString}=nothing, beamsearch_overrides=nothing)
    body = _search_body(vector, k, filter, partition, beamsearch_overrides)
    return _results(request(h.conn, "POST", dataset_path(h.id, "/search"); body))
end

"The body of `POST .../search`."
function _search_body(vector, k, filter, partition, beamsearch_overrides)
    _check_filter(filter)
    body = Dict{String,Any}("vector" => convert(Vector{Float32}, vector), "k" => Int(k))
    filter === nothing || (body["filter"] = filter)
    partition === nothing || (body["partition"] = partition)
    beamsearch_overrides === nothing || (body["beamsearch_overrides"] = Dict(pairs(beamsearch_overrides)))
    return body
end

"""
    search(h::RemoteProject, queries::AbstractVector{<:AbstractVector{<:Real}}, k=10; ntasks=4)
        -> Vector{Vector{SearchResult}}

The `k` nearest neighbors of each query, one inner vector per query, in the order of `queries`.

The server has no route for a batch of queries. The client sends one request per query, with at
most `ntasks` requests at the same time.
"""
function search(h::RemoteProject, queries::AbstractVector{<:AbstractVector{<:Real}}, k::Integer=10;
                ntasks::Integer=4)
    asyncmap(q -> search(h, q, k), queries; ntasks)
end

"""
    ftsearch(h::RemoteProject, text, k=10; filter=nothing, partition=nothing) -> Vector{SearchResult}

The `k` items of a text dataset with the best score for `text`
(`POST /api/v1/datasets/{id}/ftsearch`). `distance` is the value the engine reports, and the
results are in the engine's order. `filter` and `partition` work as in [`search`](@ref).

The engine also accepts a list of tokens. The server accepts only a string, so this function
does too.
"""
function ftsearch(h::RemoteProject, text::AbstractString, k::Integer=10; filter=nothing,
                  partition::Union{Nothing,AbstractString}=nothing)
    _check_filter(filter)
    body = Dict{String,Any}("text" => String(text), "k" => Int(k))
    filter === nothing || (body["filter"] = filter)
    partition === nothing || (body["partition"] = partition)
    return _results(request(h.conn, "POST", dataset_path(h.id, "/ftsearch"); body))
end

_check_filter(::Nothing) = nothing
_check_filter(::AbstractDict) = nothing
_check_filter(f) = throw(ArgumentError(
    "a remote filter is a filter object of the server, such as Dict(\"year\" => Dict(\"gte\" => 2020)); " *
    "got $(typeof(f)). A function cannot be sent to the server."))

_results(resp) = SearchResult[SearchResult(Int32(r[:_id]), _doc_id(r), Float32(r[:distance]), false)
                              for r in resp[:results]]

_doc_id(r) = (d = get(r, :doc_id, nothing); d === nothing ? nothing : string(d))

"""
    fetch_items(h::RemoteProject, ids) -> Vector{StoredItem}

The items with the given ids. Each id is an internal `_id` or a `doc_id`, as in the engine. An
id that matches no item is left out, so the result can be shorter than `ids`.

The server does not return the vector or the text of an item, so `payload` is `nothing` (see
[`StoredItem`](@ref)).
"""
function fetch_items(h::RemoteProject, ids)
    resp = request(h.conn, "POST", dataset_path(h.id, "/fetch"); body=Dict("ids" => collect(ids)))
    return StoredItem[_stored_item(r) for r in resp[:results]]
end

const RECORD_FIELDS = (:_id, :doc_id, :keywords, :refs, :schema_version)

function _stored_item(r)
    meta = Dict{String,Any}(string(k) => _plain(v) for (k, v) in r if k ∉ RECORD_FIELDS)
    StoredItem(Int32(r[:_id]), Int(get(r, :schema_version, 1)), _doc_id(r),
               String[string(x) for x in get(r, :keywords, String[])],
               String[string(x) for x in get(r, :refs, String[])], nothing, meta)
end

"""
    exists(h::RemoteProject, ids) -> Vector{ExistsResult}

For each id, whether an item has it and whether that item is deleted, one result per id in the
order of `ids`. Each id is an internal `_id` or a `doc_id`. The ids travel in the query string
separated by commas, so an id that contains a comma raises `ArgumentError`.
"""
function exists(h::RemoteProject, ids)
    strs = String[string(i) for i in ids]
    bad = findfirst(s -> occursin(',', s), strs)
    bad === nothing || throw(ArgumentError("exists cannot send an id that contains a comma: $(repr(strs[bad]))"))
    resp = request(h.conn, "GET", dataset_path(h.id, "/exists"); query=Dict("ids" => join(strs, ",")))
    return ExistsResult[ExistsResult(string(r[:id]), Bool(r[:exists]), Bool(r[:deleted])) for r in resp[:results]]
end

"""
    delete_item!(h::RemoteProject, _id::Integer) -> nothing

Marks the item `_id` as deleted. Later searches do not return it.
"""
function delete_item!(h::RemoteProject, _id::Integer)
    request(h.conn, "POST", dataset_path(h.id, "/delete"); body=Dict("_id" => Int(_id)))
    return nothing
end

"""
    calibrate!(h::RemoteProject; levels=nothing, numqueries=nothing, ksearch=nothing, queries=nothing,
               tradeoff=nothing, transition=nothing, partition=nothing) -> Vector{CalibrationLevel}

Tunes the beam search of a `searchgraph` dataset for one recall level, and installs the result
as the default of the dataset (`POST /api/v1/datasets/{id}/calibrate`).

The server calibrates one level per request, so `levels` is one number, and the result has one
element. The server decides every keyword that is `nothing`: `levels` is then `0.9`.
`partition` calibrates the graph of one partition of a dense dataset, as in the engine.
"""
function calibrate!(h::RemoteProject; levels::Union{Nothing,Real}=nothing, numqueries=nothing, ksearch=nothing,
                    queries=nothing, tradeoff=nothing, transition=nothing,
                    partition::Union{Nothing,AbstractString}=nothing)
    body = Dict{String,Any}()
    for (key, value) in (("minrecall", levels), ("numqueries", numqueries), ("ksearch", ksearch),
                         ("tradeoff", tradeoff), ("partition", partition))
        value === nothing || (body[key] = value)
    end
    queries === nothing || (body["queries"] = [convert(Vector{Float32}, q) for q in queries])
    transition === nothing || (body["transition"] = collect(transition))
    return [_calibration_level(request(h.conn, "POST", dataset_path(h.id, "/calibrate"); body))]
end

# The `CalibrationLevel` of a calibration response, of a dataset or of a partition of a follower.
function _calibration_level(resp)
    b, c = resp[:baseline], resp[:calibration]
    CalibrationLevel(Float32(c[:minrecall]), (bsize=Int(b[:bsize]), delta=Float64(b[:delta]), maxvisits=Int(b[:maxvisits])),
                     Float64(c[:recall]), Float64(c[:recall_lo]), Float64(c[:recall_hi]), Float64(c[:confidence]),
                     Int(c[:numqueries]), Float64(c[:distance_evaluations]), Bool(c[:heldout]))
end

"""
    hybrid_search(dense::RemoteProject, lexical::RemoteProject; vector=nothing, text=nothing, k=10)
        -> Vector{@NamedTuple{_id::Int32, doc_id::Union{Nothing,String}, score::Float64}}

Searches `vector` in the dense dataset and `text` in the text dataset, and combines the two
lists of results by reciprocal rank fusion (`POST /api/v1/search/hybrid`). A higher `score` is
a better result. The two datasets must be on the same server and share the `_id`s of their
items. The engine has no counterpart of this function.
"""
function hybrid_search(dense::RemoteProject, lexical::RemoteProject; vector=nothing,
                       text::Union{Nothing,AbstractString}=nothing, k::Integer=10)
    dense.conn.url == lexical.conn.url ||
        throw(ArgumentError("hybrid_search needs two datasets of the same server; got $(dense.conn.url) and $(lexical.conn.url)"))
    body = Dict{String,Any}("dense_index" => dense.id, "lexical_index" => lexical.id, "k" => Int(k))
    vector === nothing || (body["vector"] = convert(Vector{Float32}, vector))
    text === nothing || (body["text"] = String(text))
    resp = request(dense.conn, "POST", "/api/v1/search/hybrid"; body)
    return [(_id=Int32(r[:_id]), doc_id=_doc_id(r), score=Float64(r[:score])) for r in resp[:results]]
end
