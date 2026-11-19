"""
    AbstractItem

An item to append: [`DenseItem`](@ref), [`SparseItem`](@ref) or [`TextItem`](@ref). The three
types have the fields and the constructors of the engine's types of the same name.
"""
abstract type AbstractItem end

"""
    DenseItem(vector; doc_id=nothing, keywords=String[], refs=String[], meta=Dict{String,Any}())

A dense vector to append to a dense dataset, with its metadata. `vector` is converted to
`Vector{Float32}`.
"""
struct DenseItem <: AbstractItem
    vector::Vector{Float32}
    doc_id::Union{String,Nothing}
    keywords::Vector{String}
    refs::Vector{String}
    meta::Dict{String,Any}
end

DenseItem(vector::AbstractVector{<:Real}; doc_id=nothing, keywords=String[], refs=String[],
          meta::AbstractDict=Dict{String,Any}()) =
    DenseItem(convert(Vector{Float32}, vector), _as_doc_id(doc_id), _as_strings(keywords),
              _as_strings(refs), _as_meta(meta))

"""
    TextItem(text; doc_id=nothing, keywords=String[], refs=String[], meta=Dict{String,Any}())

A text to append to a text dataset, with its metadata.
"""
struct TextItem <: AbstractItem
    text::String
    doc_id::Union{String,Nothing}
    keywords::Vector{String}
    refs::Vector{String}
    meta::Dict{String,Any}
end

TextItem(text::AbstractString; doc_id=nothing, keywords=String[], refs=String[],
         meta::AbstractDict=Dict{String,Any}()) =
    TextItem(String(text), _as_doc_id(doc_id), _as_strings(keywords), _as_strings(refs), _as_meta(meta))

"""
    SparseItem(vector; doc_id=nothing, keywords=String[], refs=String[], meta=Dict{String,Any}())

A sparse vector to append to a sparse dataset (`index_type="sparse_invfile"`), with its
metadata. The length of `vector` is its dimension, and it must be the dimension of the dataset.
"""
struct SparseItem <: AbstractItem
    vector::SparseVector{Float32,Int32}
    doc_id::Union{String,Nothing}
    keywords::Vector{String}
    refs::Vector{String}
    meta::Dict{String,Any}
end

SparseItem(vector::SparseVector; doc_id=nothing, keywords=String[], refs=String[],
           meta::AbstractDict=Dict{String,Any}()) =
    SparseItem(convert(SparseVector{Float32,Int32}, vector), _as_doc_id(doc_id), _as_strings(keywords),
               _as_strings(refs), _as_meta(meta))

_as_doc_id(::Nothing) = nothing
_as_doc_id(id) = string(id)
_as_strings(xs) = String[string(x) for x in xs]
_as_meta(m::AbstractDict) = Dict{String,Any}(string(k) => v for (k, v) in m)

"""
    wire_item(item::AbstractItem) -> Dict{String,Any}

`item` in the shape of one element of the `items` array of `POST .../append`.
"""
function wire_item(item::AbstractItem)
    d = Dict{String,Any}("keywords" => item.keywords, "refs" => item.refs, "meta" => item.meta)
    item.doc_id === nothing || (d["doc_id"] = item.doc_id)
    _wire_payload!(d, item)
    return d
end

_wire_payload!(d, item::DenseItem) = (d["vector"] = item.vector; d)
_wire_payload!(d, item::TextItem) = (d["text"] = item.text; d)
function _wire_payload!(d, item::SparseItem)
    d["indices"] = SparseArrays.nonzeroinds(item.vector)
    d["values"] = SparseArrays.nonzeros(item.vector)
    d["dimension"] = length(item.vector)
    return d
end

"""
    SearchResult

One result of [`search`](@ref) or [`ftsearch`](@ref): the internal `_id` of the item, its
`doc_id`, and its distance to the query. The fields are those of the engine's `SearchResult`.

`deleted` is always `false`. The engine reports a deleted item with `deleted=true`, and the
server removes those results before it answers.
"""
struct SearchResult
    _id::Int32
    doc_id::Union{String,Nothing}
    distance::Float32
    deleted::Bool
end

"""
    StoredItem

One item of [`fetch_items`](@ref). The fields are those of the engine's `StoredItem`.

`payload` is always `nothing`: the server's `fetch` route does not return the vector or the
text of an item. `meta` holds every field of the response that is not one of the record's
fields (`_id`, `doc_id`, `keywords`, `refs`, `schema_version`). The server sends the `meta` of
an item and the fields of its record in one object, so a `meta` key with the name of a record
field is lost.
"""
struct StoredItem
    _id::Int32
    schema_version::Int
    doc_id::Union{String,Nothing}
    keywords::Vector{String}
    refs::Vector{String}
    payload::Nothing
    meta::Dict{String,Any}
end

"""
    ExistsResult

One result of [`exists`](@ref): the id that was asked for, whether an item has it, and whether
that item is deleted. The fields are those of the engine's `ExistsResult`.
"""
struct ExistsResult
    id::String
    exists::Bool
    deleted::Bool
end

"""
    CalibrationLevel

The result of [`calibrate!`](@ref) for one recall level. The fields are those of the engine's
`CalibrationLevel`, except two:

- `beamsearch` is a named tuple `(bsize, delta, maxvisits)`, because the client does not load
  SimilaritySearch and its `BeamSearch` type.
- There is no `recall_std`, because the server does not send it.
"""
struct CalibrationLevel
    minrecall::Float32
    beamsearch::@NamedTuple{bsize::Int, delta::Float64, maxvisits::Int}
    recall::Float64
    recall_lo::Float64
    recall_hi::Float64
    confidence::Float64
    numqueries::Int
    distance_evaluations::Float64
    heldout::Bool
end

function Base.show(io::IO, c::CalibrationLevel)
    print(io, "CalibrationLevel(minrecall=", c.minrecall, ", recall=", round(c.recall; digits=4),
          ", ", round(100 * c.confidence; digits=1), "% [", round(c.recall_lo; digits=4), ", ",
          round(c.recall_hi; digits=4), "], ", c.numqueries, c.heldout ? " held-out" : " tuning",
          " queries, ", round(c.distance_evaluations; digits=1), " evaluations/query)")
end
