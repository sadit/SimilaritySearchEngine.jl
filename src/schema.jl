module Schema

using JSON3
using SparseArrays: SparseVector, sparsevec, nonzeros, nonzeroinds

export AbstractItem, DenseItem, SparseItem, TextItem, MetadataRecord, StoredItem
export payload, metadata_record, encode_meta, decode_meta, raw_meta
export get_field, matches_filter
export be_key, decode_be_key

"""
    be_key(id::Integer) -> Vector{UInt8}
    decode_be_key(key) -> UInt32

An integer id as a RocksDB key, big-endian, and back.

**One rule for every integer key this package writes**, in any column family: big-endian, four
bytes. Not because of the machine -- every platform this runs on is little-endian, and `hton`
is a byte swap that costs nothing next to the point read that follows it (measured: 0.41s
against 0.54s per 10 million keys, both dominated by allocating the 4-byte vector) -- but
because RocksDB compares keys **bytewise**. Native little-endian bytes sort as
`[65536, 256, 1, 257, 2, 1000, 255, 65535]`; big-endian ones sort numerically.

The package no longer *depends* on that ordering to be correct -- the loaders that read a
whole column family back sort by the id they decode from each key
(`Persistence.load_invfile_docvecs`, `Persistence.load_object_blocks`) -- but iterating a range
of ids in order is a property worth keeping, and having one rule means nobody has to ask which
encoding a given column family uses.

`collect` materializes a concrete `Vector{UInt8}` rather than a lazy `reinterpret` view over a
temporary array: `WriteBatch` defers the actual write until `write!`, and a lazy view is not
reliably kept alive across that gap -- confirmed with a 200-key round-trip repro where a
handful of keys came back missing from `get` on every run.
"""
be_key(id::Integer) = collect(reinterpret(UInt8, [hton(UInt32(id))]))

"See [`be_key`](@ref)."
decode_be_key(key::AbstractVector{UInt8}) = ntoh(only(reinterpret(UInt32, Vector{UInt8}(key))))

"""
    AbstractItem

Something to append to a project: [`DenseItem`](@ref) (a dense vector), [`SparseItem`](@ref)
(a sparse one) or [`TextItem`](@ref) (a string), each carrying its own external id, tags,
references and free-form metadata.

This package takes typed values in and hands typed values back. It does not accept a
JSON-shaped `Dict` as an item and pick it apart, which is what it used to do -- the caller
splits its own data, because the caller is the one who knows what its fields mean. The single
exception is `meta`, which is free-form by definition and stays a `Dict{String,Any}`. On the
way out the same rule holds: [`StoredItem`](@ref), not a merged dictionary. The one raw-JSON
path that remains is [`raw_meta`](@ref), reserved for an HTTP layer forwarding stored bytes it
never inspects.

Which concrete type an item is *is* the check. A project indexing dense vectors takes
`DenseItem`s, one indexing sparse vectors takes `SparseItem`s and one indexing text takes
`TextItem`s; handing over the wrong one raises, where the dictionary form silently skipped any
item whose expected key was missing and returned a count that quietly disagreed with what the
caller passed.
"""
abstract type AbstractItem end

"""
    DenseItem(vector; doc_id=nothing, keywords=String[], refs=String[], meta=Dict{String,Any}())

One dense vector to index, with its metadata.

`vector` is converted to `Vector{Float32}` on construction -- the element type the engine
indexes in -- so a caller holding `Float64`s or a `SubArray` does not have to convert first,
and the conversion happens once, here, rather than per insertion.
"""
struct DenseItem <: AbstractItem
    vector::Vector{Float32}
    doc_id::Union{String,Nothing}
    keywords::Vector{String}
    refs::Vector{String}
    meta::Dict{String,Any}
end

function DenseItem(vector::AbstractVector;
                   doc_id=nothing, keywords=String[], refs=String[],
                   meta::AbstractDict=Dict{String,Any}())
    DenseItem(convert(Vector{Float32}, vector), _as_doc_id(doc_id),
              _as_strings(keywords), _as_strings(refs), _as_meta(meta))
end

"""
    TextItem(text; doc_id=nothing, keywords=String[], refs=String[], meta=Dict{String,Any}())

One text document to index, with its metadata. In a paragraph-level project this is one
paragraph, not one book: the item is whatever unit a search should hand back.
"""
struct TextItem <: AbstractItem
    text::String
    doc_id::Union{String,Nothing}
    keywords::Vector{String}
    refs::Vector{String}
    meta::Dict{String,Any}
end

function TextItem(text::AbstractString;
                  doc_id=nothing, keywords=String[], refs=String[],
                  meta::AbstractDict=Dict{String,Any}())
    TextItem(String(text), _as_doc_id(doc_id),
             _as_strings(keywords), _as_strings(refs), _as_meta(meta))
end

_as_doc_id(::Nothing) = nothing
_as_doc_id(id) = string(id)
_as_strings(xs) = String[string(x) for x in xs]
_as_meta(m::Dict{String,Any}) = m
_as_meta(m::AbstractDict) = Dict{String,Any}(string(k) => v for (k, v) in m)

"""
    SparseItem(vector; doc_id=nothing, keywords=String[], refs=String[], meta=Dict{String,Any}())

One sparse vector to index, with its metadata.

A `SparseVector{Float32,Int32}`, and only that: the element and index types an inverted file
indexes in, converted once here rather than per insertion. A bag of counts is *not* accepted in
its place, deliberately -- a `Dict{UInt32,Int32}` and a sparse vector mean different things to a
scorer (presence-and-count against weight), and taking both would make the item's type stop
answering which one this is.

The vector's length is its dimension, and every item in a project must agree with the dimension
that project was created with: an inverted file is a fixed array of posting lists, so the
dimension is structural rather than descriptive.
"""
struct SparseItem <: AbstractItem
    vector::SparseVector{Float32,Int32}
    doc_id::Union{String,Nothing}
    keywords::Vector{String}
    refs::Vector{String}
    meta::Dict{String,Any}
end

function SparseItem(vector::SparseVector;
                    doc_id=nothing, keywords=String[], refs=String[],
                    meta::AbstractDict=Dict{String,Any}())
    SparseItem(convert(SparseVector{Float32,Int32}, vector), _as_doc_id(doc_id),
               _as_strings(keywords), _as_strings(refs), _as_meta(meta))
end

"""
    payload(item::AbstractItem)

The part of `item` that gets indexed -- the vector of a [`DenseItem`](@ref), the text of a
[`TextItem`](@ref) -- as opposed to the metadata that travels beside it.

Exists so code that treats both kinds uniformly (staging, persistence) does not branch on the
concrete type just to reach the one field whose name differs.
"""
payload(item::DenseItem) = item.vector
payload(item::SparseItem) = item.vector
payload(item::TextItem) = item.text

"""
    MetadataRecord

Fixed-shape record every indexed object carries, independent of whatever `meta` a
caller attaches to it (see [`metadata_record`](@ref)/`Project`'s own dedicated `meta`
column family) -- there is no per-project typed-field schema to declare or validate
against anymore: every record has exactly these fields, always. All-concrete field
types, so `Vector{MetadataRecord}` writes/reads directly through `Avro.writetable`/
`readtable` (confirmed empirically -- unlike a `meta`/`Dict{String,Any}` field, which
`Avro.jl` schema inference rejects outright since `Any` isn't a concrete value type;
that's why `meta` is a separate JSON blob, never a field on this struct).

# Fields
- `_id::Int32`: internal sequential id, 1..n -- the same key the engine's own raw
  vector/text column family is indexed by (see `Persistence`), so a record and its
  vector/text share one id space, never duplicated across the two. Leading
  underscore flags it as the internal handle (vs. `doc_id` below), but it's kept as
  a real, always-present field -- callers need it back to line up with result
  matrices/ids from search, not just to recover data by key.
- `schema_version::Int`: version tag for whatever shape the accompanying `meta`
  happens to follow -- carried here purely for the caller's own interpretation of
  `meta`, not validated or enforced by this module.
- `doc_id::Union{String,Nothing}`: the caller-supplied external id, if the appended
  item had one (its raw `"doc_id"` key) -- `nothing` if it didn't.
- `keywords::Vector{String}`: free tags, indexed for lookup (see `Project`).
- `refs::Vector{String}`: `doc_id`s of related records -- *not* validated against
  anything at write time (a ref may point at a `doc_id` that doesn't exist yet, or
  ever, e.g. a forward reference inserted before its target), so this is a plain
  unchecked list, not a foreign key.
"""
struct MetadataRecord
    _id::Int32
    schema_version::Int
    doc_id::Union{String,Nothing}
    keywords::Vector{String}
    refs::Vector{String}
end

MetadataRecord(_id::Int32, schema_version::Int) = MetadataRecord(_id, schema_version, nothing, String[], String[])

"""
    metadata_record(item::AbstractItem, _id::Int32, schema_version::Int) -> MetadataRecord

`item`'s fixed-shape half, stamped with the id the project assigned it.

The payload is deliberately not in here and not duplicated anywhere near it: a vector or a
text already lives in the engine's own dedicated, 1..n-indexed storage (see `Persistence`),
and this record shares that id space with it.

This replaces the `split_item(_id, version, ::AbstractDict)` that used to pull `doc_id`,
`keywords` and `ref` out of a raw dictionary by key and sweep the rest into `meta`. There is
nothing left to split: an [`AbstractItem`](@ref) arrives already separated, by a caller who
knows which of its fields are tags and which are payload. What was a parsing step with a
reserved-key list is now a projection.
"""
metadata_record(item::AbstractItem, _id::Int32, schema_version::Int) =
    MetadataRecord(_id, schema_version, item.doc_id, item.keywords, item.refs)

"""
    StoredItem

One indexed item read back out: its [`MetadataRecord`](@ref) fields, its free-form `meta`, and
its `payload` -- the very text, dense vector or sparse vector that was indexed.

`payload` is what a search result needs and what the dictionary-returning `fetch_items` could
not give: `text` was a reserved key, stripped before anything reached the metadata store, so a
hit came back with everything *about* the paragraph and not the paragraph. It is `nothing` only
when the payload cannot be read back -- a `DenseEngine{<:ExactBackend}` whose id is out of range, or an
engine kind with no per-item storage to consult.
"""
struct StoredItem
    _id::Int32
    schema_version::Int
    doc_id::Union{String,Nothing}
    keywords::Vector{String}
    refs::Vector{String}
    payload::Union{String,Vector{Float32},SparseVector{Float32,Int32},Nothing}
    meta::Dict{String,Any}
end

StoredItem(record::MetadataRecord, payload, meta::Dict{String,Any}) =
    StoredItem(record._id, record.schema_version, record.doc_id, record.keywords, record.refs,
               payload, meta)

"""
    encode_meta(meta) -> String

JSON3-encodes `meta` (a `Dict`, a `Vector`, or any other JSON3-writable value),
for storage in `Project`'s dedicated meta column family. Returns the `String`
`JSON3.write` itself produces rather than converting it to a `Vector{UInt8}` --
that conversion isn't a free reinterpretation, it's a real copy (confirmed
empirically: `pointer` differs before/after, with allocation to match), so it'd
just be paying twice to get back to bytes `JSON3.write` already builds internally
before wrapping them as a `String`. No need to pay it at all here: `RocksDB.jl`'s
own `put!` already accepts a plain `String` and converts it to bytes via
`codeunits` (zero-copy) right before handing it to the C API.
"""
encode_meta(meta) = JSON3.write(meta)

"""
    decode_meta(bytes::Union{Vector{UInt8},Nothing}; lazy::Bool) -> Any

Inverse of [`encode_meta`](@ref). `nothing`/empty input decodes to `nothing`.

With `lazy=true` (the default), returns `JSON3`'s own lazy `Object`/`Array` view
straight over `bytes` -- cheap to produce (confirmed empirically: allocates roughly
as much as a full `Dict` parse just to index top-level keys, but nothing nested is
materialized until actually accessed) and useful for a caller that only reads a
field or two out of `meta`. This is *not* a free round-trip on write, though: writing
a parsed `JSON3.Object` back out with `JSON3.write` still walks and reserializes
every value -- empirically no cheaper than writing a plain `Dict` (slightly more
allocation, in fact). A caller that only needs to pass `meta` through unmodified
(e.g. straight into an HTTP response body) should skip decoding entirely and reuse
the raw `bytes` from storage directly -- that's the only actually-zero-cost path.

Pass `lazy=false` for a fully materialized `Dict{String,Any}`/`Vector{Any}` instead,
when the caller needs to mutate or merge `meta` rather than just read from it.
"""
function decode_meta(bytes::Vector{UInt8}; lazy::Bool)
    isempty(bytes) && return nothing
    lazy ? JSON3.read(String(copy(bytes))) : JSON3.read(String(copy(bytes)), Dict{String,Any})
end
decode_meta(::Nothing; lazy::Bool) = nothing

"""
    raw_meta(bytes::Union{Vector{UInt8},Nothing}) -> Union{String,Nothing}

The actually-zero-cost path for a caller that only forwards `meta` unmodified (a
fetch/HTTP handler splicing it into a hand-built response, or writing it straight
out as a response body) and never inspects its fields -- unlike either
[`decode_meta`](@ref) mode, this never touches JSON3 at all: no lazy `Object`, no
`Dict`, nothing parsed or walked, just the stored bytes handed back as a `String`.

`String(::Vector{UInt8})` takes ownership of `bytes` and empties it as a side
effect (a real move, not a copy -- that's the whole point here), so pass a `bytes`
this call is free to consume; if the caller still needs `bytes` afterwards (e.g. to
also call `decode_meta` on the same underlying data), copy it first.
"""
raw_meta(bytes::Vector{UInt8}) = isempty(bytes) ? nothing : String(bytes)
raw_meta(::Nothing) = nothing

"""
    get_field(record::MetadataRecord, meta, name::String) -> Any

Retrieves a field by name either from `record`'s fixed fields (`_id`, `doc_id`, `keywords`, `refs`, `schema_version`)
or from free-form `meta` dictionary / object.
"""
function get_field(record::MetadataRecord, meta, name::String)
    name == "_id" && return record._id
    name == "doc_id" && return record.doc_id
    name == "keywords" && return record.keywords
    name == "refs" && return record.refs
    name == "schema_version" && return record.schema_version
    meta isa AbstractDict && return get(meta, name, nothing)
    meta !== nothing && hasproperty(meta, Symbol(name)) && return getproperty(meta, Symbol(name))
    return nothing
end
get_field(record::MetadataRecord, name::String) = get_field(record, nothing, name)

"""
    matches_filter(record::MetadataRecord, meta, filter::AbstractDict) -> Bool

Post-filter predicate for `/search`-family endpoints (PLAN.md §5.4). `filter` maps field
names to either a bare value (equality) or a spec dict supporting `gte`/`lte`/`gt`/`lt`
(range), `in`/`nin` (set membership), and `eq`/`neq`. A field missing from the record/meta fails the filter.
"""
function matches_filter(record::MetadataRecord, meta, filter::AbstractDict)
    for (field, spec) in filter
        value = get_field(record, meta, string(field))
        value === nothing && return false

        if spec isa AbstractDict
            haskey(spec, "gte") && !(value >= spec["gte"]) && return false
            haskey(spec, "lte") && !(value <= spec["lte"]) && return false
            haskey(spec, "gt") && !(value > spec["gt"]) && return false
            haskey(spec, "lt") && !(value < spec["lt"]) && return false
            haskey(spec, "in") && !(value in spec["in"]) && return false
            haskey(spec, "nin") && (value in spec["nin"]) && return false
            haskey(spec, "eq") && !(value == spec["eq"]) && return false
            haskey(spec, "neq") && !(value != spec["neq"]) && return false
        elseif value != spec
            return false
        end
    end
    return true
end
matches_filter(record::MetadataRecord, filter::AbstractDict) = matches_filter(record, nothing, filter)

end # module
