module Persistence

using Avro
using RocksDB
using SimilaritySearch
using TextSearch

export EngineStore, ENGINE_CF, open_engine_store, save_field!, load_field, has_field, save_fields!
export append_block!, load_blocks
export AdjacencyStore, ADJACENCY_CF, open_adjacency_store, save_neighbors!, load_neighbors
export InvertedFileObjectStore, INVFILE_DB_CF, open_invertedfile_object_store, append_objects!, load_object_blocks
export DumpRecord, write_dump_records, read_dump_records

# ---------------------------------------------------------
# Shared low-level primitives, operating directly on a RocksDBDict -- every named store
# below (EngineStore, AdjacencyStore, InvertedFileObjectStore) is a thin wrapper around
# one of these, scoped to its own column family, so this is the one place the actual
# key layout (plain keys, or a `"<prefix>:block:<0-padded index>"` append-only sequence
# plus a small `"<prefix>:block_count"` counter) is defined.
# ---------------------------------------------------------

_save_key!(dict, key::String, value) = (dict[key] = value; nothing)
_load_key(dict, key::String, default=nothing) = get(dict, key, default)

function _append_block!(dict, prefix::String, value)
    count = _load_key(dict, "$(prefix):block_count", 0) + 1
    _save_key!(dict, "$(prefix):block:$(lpad(count, 12, '0'))", value)
    _save_key!(dict, "$(prefix):block_count", count)
    return count
end

function _load_blocks(dict, prefix::String)
    count = _load_key(dict, "$(prefix):block_count", 0)
    return [_load_key(dict, "$(prefix):block:$(lpad(i, 12, '0'))") for i in 1:count]
end

# ---------------------------------------------------------
# Per-field engine persistence (RocksDB column family)
# ---------------------------------------------------------

"""
    ENGINE_CF

Name of the RocksDB column family an [`EngineStore`](@ref) is backed by. Pass this in
`Project.open_project`'s `extra_cf_names` so the column family is listed on every open,
not just the first (see that keyword's docstring).
"""
const ENGINE_CF = "engine"

"""
    EngineStore

A `field_name::Symbol => value` store for one `IndexEngine.AbstractSearchEngine`'s state,
backed by one RocksDB column family (`ENGINE_CF`) with one key per struct field --
replacing a single JLD2 blob holding the whole engine (index included) so that a mutation
touching only one field (e.g. `deleted_ids` on a soft-delete) only ever rewrites that one
key, not the whole index.

Shares its underlying `RocksDB.DB` connection (typically `Project.ProjectManager.db`) --
closing that `DB` (e.g. via `Project.close_project`) is what closes this store too; it has
no separate `close` of its own to call.
"""
struct EngineStore
    dict::RocksDB.RocksDBDict{String, Any}
end

"""
    open_engine_store(db::RocksDB.DB) -> EngineStore

Wraps `db`'s [`ENGINE_CF`](@ref) column family (which must already be open on `db` --
see `Project.open_project`'s `extra_cf_names`) as an [`EngineStore`](@ref).
"""
open_engine_store(db::RocksDB.DB) = EngineStore(RocksDB.RocksDBDict{String, Any}(db, ENGINE_CF))

"""
    save_key!(store::EngineStore, key::String, value)

Writes just `value` under the literal RocksDB key `key`. Low-level counterpart of
[`save_field!`](@ref) (which just does `save_key!(store, String(field), value)`) for
callers that need a dynamically-built key -- see [`append_block!`](@ref).
"""
save_key!(store::EngineStore, key::String, value) = _save_key!(store.dict, key, value)

"""
    load_key(store::EngineStore, key::String, default=nothing)

Reads back the value last saved under the literal RocksDB key `key`, or `default` if it
was never saved. Low-level counterpart of [`load_field`](@ref).
"""
load_key(store::EngineStore, key::String, default=nothing) = _load_key(store.dict, key, default)

"""
    save_field!(store::EngineStore, field::Symbol, value)

Writes just `value` under `field`'s key, leaving every other field's key untouched.
"""
save_field!(store::EngineStore, field::Symbol, value) = save_key!(store, String(field), value)

"""
    load_field(store::EngineStore, field::Symbol, default=nothing)

Reads back the value last saved under `field`'s key, or `default` if it was never saved.
"""
load_field(store::EngineStore, field::Symbol, default=nothing) = load_key(store, String(field), default)

"""
    has_field(store::EngineStore, field::Symbol) -> Bool
"""
has_field(store::EngineStore, field::Symbol) = haskey(store.dict, String(field))

"""
    append_block!(store::EngineStore, field::Symbol, value) -> Int

Appends `value` as a new, never-again-rewritten block in `field`'s sequence (key
`"<field>:block:<0-padded index>"`), bumping the small `"<field>:block_count"` counter
alongside it, and returns the assigned (1-based) block index. Unlike [`save_field!`](@ref)
(one key, always overwritten) this is for state that grows by *appending* -- e.g. a
`SearchGraph`'s insertion-range vectors blocks (see `IndexEngine.searchgraph_vectors`;
its per-object adjacency is a separate, dedicated [`AdjacencyStore`](@ref) instead) --
where writing the new block must never require rewriting any earlier one. See
[`load_blocks`](@ref) to read the whole sequence back in order.
"""
append_block!(store::EngineStore, field::Symbol, value) = _append_block!(store.dict, String(field), value)

"""
    load_blocks(store::EngineStore, field::Symbol) -> Vector

Every block [`append_block!`](@ref) has written for `field`, in the order they were
appended.
"""
load_blocks(store::EngineStore, field::Symbol) = _load_blocks(store.dict, String(field))

"""
    save_fields!(store::EngineStore, fields::NamedTuple)

Writes every `field => value` pair in `fields` -- used for a full initial save (e.g.
right after `create_engine`) or a one-off multi-field transition (e.g. a text engine's
first training batch setting `voc`/`model`/`index` together), as opposed to
[`save_field!`](@ref)'s single-key writes for routine mutations.

Writes one key at a time via [`save_field!`](@ref) rather than `RocksDB.batch` -- that
Tier-3 helper's `RocksDBBatchProxy` currently writes every key to the `"default"` column
family regardless of which column family the wrapped `RocksDBDict` is scoped to (verified
directly against a throwaway `db`: `RocksDB.batch(d) do b; b["x"] = 1; end` on a dict
scoped to a non-`"default"` column family lands nothing there), which would silently
misplace every field this function saves. Not atomic across fields as a result; each
field in this NamedTuple is small and this is only ever called for rare, one-shot
transitions, so a partial write on a crash mid-loop is an acceptable risk here.
"""
function save_fields!(store::EngineStore, fields::NamedTuple)
    for (k, v) in pairs(fields)
        save_field!(store, k, v)
    end
end

# ---------------------------------------------------------
# Per-object adjacency persistence (SearchGraph only, its own RocksDB column family)
# ---------------------------------------------------------

"""
    ADJACENCY_CF

Name of the RocksDB column family a `SearchGraph`'s per-object direct-neighbor lists are
stored in (see [`AdjacencyStore`](@ref)) -- separate from [`ENGINE_CF`](@ref) so this
(potentially one-entry-per-indexed-object) data doesn't share a keyspace with the
handful of other engine fields (`deleted_ids`, `minrecall`, ...). Pass this in
`Project.open_project`'s `extra_cf_names` alongside `ENGINE_CF`, for the same reason
(every column family that exists on disk must be listed on every open, not just the
first).
"""
const ADJACENCY_CF = "searchgraph_adj"

"""
    AdjacencyStore

A `object_id::Integer => direct_neighbors::Vector{UInt32}` store, one entry per indexed
object, backed by its own RocksDB column family ([`ADJACENCY_CF`](@ref)). Only ever used
for a `SearchGraph`'s direct links (see `IndexEngine.direct_neighbors`/
`build_searchgraph`) -- `InvertedFile`/`BM25InvertedFile` have no equivalent per-object
adjacency concept; their own incremental persistence is
[`InvertedFileObjectStore`](@ref) instead.

Shares its underlying `RocksDB.DB` connection, same as [`EngineStore`](@ref); no separate
`close` of its own to call.
"""
struct AdjacencyStore
    dict::RocksDB.RocksDBDict{String, Vector{UInt32}}
end

"""
    open_adjacency_store(db::RocksDB.DB) -> AdjacencyStore

Wraps `db`'s [`ADJACENCY_CF`](@ref) column family (which must already be open on `db` --
see `Project.open_project`'s `extra_cf_names`) as an [`AdjacencyStore`](@ref).
"""
open_adjacency_store(db::RocksDB.DB) = AdjacencyStore(RocksDB.RocksDBDict{String, Vector{UInt32}}(db, ADJACENCY_CF))

"""
    save_neighbors!(store::AdjacencyStore, object_id::Integer, neighbors::Vector{UInt32})

Writes `object_id`'s direct neighbor list, leaving every other object's entry untouched.
"""
save_neighbors!(store::AdjacencyStore, object_id::Integer, neighbors::Vector{UInt32}) =
    (store.dict[string(object_id)] = neighbors; nothing)

"""
    load_neighbors(store::AdjacencyStore, object_id::Integer) -> Vector{UInt32}

Reads back `object_id`'s direct neighbor list, or an empty vector if it was never saved.
"""
load_neighbors(store::AdjacencyStore, object_id::Integer) = get(store.dict, string(object_id), UInt32[])

# ---------------------------------------------------------
# Incremental object persistence for InvertedFile/BM25InvertedFile (their own RocksDB
# column family, a different shape from SearchGraph's -- see the module note below)
# ---------------------------------------------------------

"""
    INVFILE_DB_CF

Name of the RocksDB column family an [`InvertedFileObjectStore`](@ref) is backed by.
Pass this in `Project.open_project`'s `extra_cf_names` alongside `ENGINE_CF`/
`ADJACENCY_CF`, for the same reason (every column family that exists on disk must be
listed on every open, not just the first).
"""
const INVFILE_DB_CF = "invfile_db"

"""
    InvertedFileObjectStore

An append-only sequence of raw indexed objects (see `IndexEngine.invertedfile_objects`)
for one `BM25Engine`/`InvertedFileEngine`, backed by its own RocksDB column family
([`INVFILE_DB_CF`](@ref)) -- kept separate from [`EngineStore`](@ref) for the same reason
[`AdjacencyStore`](@ref) is: this can grow to one entry per indexed object, and shouldn't
share a keyspace with the handful of other, small engine fields.

Unlike a `SearchGraph`, an inverted file's posting lists have no direct/reverse-link
split to worry about -- `push_item!` fully finalizes each object's contribution before
`LOG` even fires (see `IndexEngine.CallbackLog`'s docstring), so what's saved here is
simply every object ever indexed, in insertion order; reloading rebuilds the whole index
by replaying them through the library's own `push_item!` again (see
`IndexEngine.build_bm25invertedfile`/`build_invertedfile`) rather than trying to persist
posting lists directly.

!!! warning "Scaling: a rebuild-by-reinsertion design, not an on-disk index format"
    Reloading this way means every raw object saved here, and the whole rebuilt index,
    has to fit in memory at once, and reloading costs the full original indexing time
    over again (proportional to the object count) rather than being O(1) or streamed.
    That's a fine trade for collections up to a few million documents -- it is *not* a
    design for billion-document corpora, which need genuinely disk-backed posting lists
    (paged/streamed from disk on demand, never fully materialized in memory) -- a
    different architecture this project does not attempt.

Shares its underlying `RocksDB.DB` connection, same as [`EngineStore`](@ref); no separate
`close` of its own to call.
"""
struct InvertedFileObjectStore
    dict::RocksDB.RocksDBDict{String, Any}
end

"""
    open_invertedfile_object_store(db::RocksDB.DB) -> InvertedFileObjectStore

Wraps `db`'s [`INVFILE_DB_CF`](@ref) column family (which must already be open on `db` --
see `Project.open_project`'s `extra_cf_names`) as an [`InvertedFileObjectStore`](@ref).
"""
open_invertedfile_object_store(db::RocksDB.DB) = InvertedFileObjectStore(RocksDB.RocksDBDict{String, Any}(db, INVFILE_DB_CF))

"""
    append_objects!(store::InvertedFileObjectStore, objects::Vector) -> Int

Appends `objects` (the raw indexed objects for one `push_item!`/`append_items!` report --
see `IndexEngine.invertedfile_objects`) as a new, never-again-rewritten block, returning
the assigned (1-based) block index. See [`load_object_blocks`](@ref) to read the whole
sequence back in order.
"""
append_objects!(store::InvertedFileObjectStore, objects::Vector) = _append_block!(store.dict, "objects", objects)

"""
    load_object_blocks(store::InvertedFileObjectStore) -> Vector{<:Vector}

Every block [`append_objects!`](@ref) has written, in the order they were appended --
concatenate them (in order) to get back every object ever indexed, in original insertion
order.
"""
load_object_blocks(store::InvertedFileObjectStore) = _load_blocks(store.dict, "objects")

# ---------------------------------------------------------
# Avro Serialization for Projects (Dump / Load, PLAN.md §4.4)
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

One project row in a `dump` bundle's Avro export (PLAN.md §4.4) -- the cross-language
interop format `load` (or any non-Julia Avro reader) consumes. Deliberately does *not*
carry a separate `vector`/`text` field: `Schema.MetadataRecord`'s `extra` already holds
the complete original raw item (including its `"vector"`/`"text"` key) for every project
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
`Avro.writetable` has no schema to infer from a table with no rows, so an empty project's
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
