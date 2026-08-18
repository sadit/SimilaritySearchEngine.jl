module Persistence

using JLD2
using Avro
using SimilaritySearch
using TextSearch

export save_snapshot, load_snapshot, DumpRecord, write_dump_records, read_dump_records

# ---------------------------------------------------------
# JLD2 Snapshotting for Index Structs
# ---------------------------------------------------------

"""
    save_snapshot(filepath::String, obj) -> String

Saves an object to disk as a JLD2 snapshot. `obj` is typically a SimilaritySearch/TextSearch
index, or a `NamedTuple` bundling an index together with the extra state needed to rebuild a
`SearchEngineWrapper` (text engines also need their trained `Vocabulary` and `text_kind`,
since neither survives inside `BM25InvertedFile`/`WeightedInvertedFile` alone for the latter).
The path should follow the convention: `{uuid}.snapshot.jld2`

# Arguments
- `filepath::String`: The output path.
- `obj`: The object to snapshot.

# Returns
- `String`: The path where the snapshot was saved.
"""
function save_snapshot(filepath::String, obj)
    JLD2.save(filepath, "index", obj)
    return filepath
end

"""
    load_snapshot(filepath::String) -> Any

Loads an index from a JLD2 snapshot.

# Arguments
- `filepath::String`: The path to the JLD2 file.

# Returns
- `Any`: The loaded index object.
"""
function load_snapshot(filepath::String)
    return JLD2.load(filepath, "index")
end

# ---------------------------------------------------------
# Avro Serialization for Datasets (Dump / Load, PLAN.md §4.4)
# ---------------------------------------------------------
#
# !!! note "Corrected against the real Avro.jl API (empirically verified, not just read from docs)"
#     Avro.jl's `writetable`/`readtable` are a plain `Tables.jl` reader/writer: given a
#     `Vector` of a concrete Julia struct, `writetable` derives the Avro schema directly
#     from the struct's field types (via `StructTypes.jl`), and `readtable` hands back
#     property-accessible rows of that same shape. There is no `schema=` keyword to
#     `writetable`, and `Avro.parseschema` is for *reading* raw bytes against an
#     externally-known schema, not for dictating what `writetable` writes -- the previous
#     version of this file called `Avro.writetable(path, records; schema=parsed_schema)`,
#     which doesn't match any real keyword `writetable` accepts. Confirmed via a direct
#     round-trip test (`Vector{Float32}`, `String`, `Int64`, `Bool` fields) before writing
#     `DumpRecord` below.

"""
    DumpRecord

One dataset row in a `dump` bundle's Avro export (PLAN.md §4.4) -- the cross-language
interop format `load` (or any non-Julia Avro reader) consumes. Deliberately does *not*
carry a separate `vector`/`text` field: `Schema.MetadataRecord`'s `extra` already holds
the complete original raw item (including its `"vector"`/`"text"` key) for every dataset
in this app, since `execute_build`/`handle_append` always store the full posted item, not
just its declared fields -- so `declared_json`/`extra_json` alone are a faithful,
losslessly-restorable copy of the record, with no redundant second copy of the payload.

# Fields
- `doc_id::Int64`: matches the index position `add_item!`/`push_item!` assigned it.
- `declared_json::String`: JSON-encoded `MetadataRecord.declared_fields`.
- `extra_json::String`: JSON-encoded `MetadataRecord.extra` (already JSON bytes internally
  -- just decoded to a `String` here since Avro has no reason to double-encode it).
- `tombstoned::Bool`: whether this doc_id was in the engine's `deleted_ids` at dump time.
  `load` restores tombstones from the copied JLD2 snapshot's own `deleted_ids` (already
  authoritative, see `Server.handle_delete_item`), not by recomputing them from this flag
  -- it's carried here purely for a non-Julia consumer that only has the Avro file.
"""
struct DumpRecord
    doc_id::Int64
    declared_json::String
    extra_json::String
    tombstoned::Bool
end

"""
    write_dump_records(filepath::String, records::Vector{DumpRecord}) -> String

Writes `records` as an Avro object container file. Errors if `records` is empty --
`Avro.writetable` has no schema to infer from a table with no rows, so an empty dataset's
dump must be handled by the caller before reaching here (see `execute_dump`).
"""
function write_dump_records(filepath::String, records::Vector{DumpRecord})
    isempty(records) && error("write_dump_records: cannot write an Avro file with zero rows (nothing to infer a schema from)")
    Avro.writetable(filepath, records)
    return filepath
end

"""
    read_dump_records(filepath::String) -> Vector{DumpRecord}

Reads back an Avro file written by `write_dump_records`, reconstructing plain
`DumpRecord`s (rather than handing back `Avro.jl`'s own row-view type) so callers don't
need to depend on `Avro.jl`'s internals beyond this module.
"""
function read_dump_records(filepath::String)
    return [DumpRecord(Int64(row.doc_id), String(row.declared_json), String(row.extra_json), Bool(row.tombstoned)) for row in Avro.readtable(filepath)]
end

end # module
