module Schema

using JSON3

export MetadataRecord
export split_item, encode_meta, decode_meta, raw_meta

"""
    MetadataRecord

Fixed-shape record every indexed object carries, independent of whatever `meta` a
caller attaches to it (see [`split_item`](@ref)/`Project`'s own dedicated `meta`
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
    split_item(_id::Int32, schema_version::Int, raw_dict::AbstractDict) -> (MetadataRecord, meta)

Splits a raw appended item into its fixed [`MetadataRecord`](@ref) (`_id`,
`schema_version`, `doc_id` from `raw_dict["doc_id"]`, `keywords` from
`raw_dict["keywords"]`, `refs` from `raw_dict["ref"]`) and everything else as
free-form `meta` -- a plain `Dict{String,Any}` of whatever raw_dict entries are left.

`"vector"`/`"text"` are dropped entirely, not carried into either half: both already
live in the engine's own dedicated, 1..n-indexed column family (see `Persistence`),
so keeping a second copy here would just be redundant storage.
"""
function split_item(_id::Int32, schema_version::Int, raw_dict::AbstractDict)
    reserved = ("vector", "text", "keywords", "ref", "doc_id")
    keywords = String.(get(raw_dict, "keywords", String[]))
    refs = String.(get(raw_dict, "ref", String[]))
    doc_id = haskey(raw_dict, "doc_id") ? string(raw_dict["doc_id"]) : nothing
    meta = Dict{String,Any}(string(k) => v for (k, v) in raw_dict if !(string(k) in reserved))
    return MetadataRecord(_id, schema_version, doc_id, keywords, refs), meta
end

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
    decode_meta(bytes::Union{Vector{UInt8},Nothing}; lazy::Bool=true) -> Any

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
function decode_meta(bytes::Vector{UInt8}; lazy::Bool=true)
    isempty(bytes) && return nothing
    lazy ? JSON3.read(String(copy(bytes))) : JSON3.read(String(copy(bytes)), Dict{String,Any})
end
decode_meta(::Nothing; lazy::Bool=true) = nothing

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

end # module
