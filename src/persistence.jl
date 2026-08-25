module Persistence

using Avro
using RocksDB
using SimilaritySearch
using TextSearch
using ..Schema

export EngineStore, ENGINE_CF, open_engine_store, save_field!, load_field, has_field, save_fields!
export AdjacencyStore, ADJACENCY_CF, open_adjacency_store, save_neighbors!, load_neighbors
export DENSE_VECTORS_FILENAME, dense_vectors_path, open_dense_vectors, load_dense_vector_blocks
export InvertedFileObjectStore, INVFILE_DB_CF, open_invertedfile_object_store, append_objects!, load_object_blocks
export StagedTextStore, STAGED_TEXT_CF, open_staged_text_store, append_staged_texts!, load_staged_text_blocks
export DumpRecord, write_dump_records, read_dump_records

# ---------------------------------------------------------
# Shared low-level primitives, operating directly on a RocksDBDict -- every named store
# below (EngineStore, AdjacencyStore, InvertedFileObjectStore) is a thin wrapper around
# one of these, scoped to its own column family.
# ---------------------------------------------------------

_save_key!(dict, key::String, value) = (dict[key] = value; nothing)
_load_key(dict, key::String, default) = get(dict, key, default)

"""
    _be_key(id::Integer) -> Vector{UInt8}

Big-endian (`hton`) bytes of `UInt32(id)`, for a `RocksDBDict{Vector{UInt8},Any}` keyed
directly by a batch's/object's own numeric id (see [`InvertedFileObjectStore`](@ref))
instead of a composite `"<prefix>:block:<0-padded index>"` string plus a separate
`"<prefix>:block_count"` counter key. Neither piece of that older scheme is needed once a
store has its *own* dedicated column family and keys each block by its own numeric
position: there is no `prefix` to disambiguate (nothing else shares this column family)
and no counter to maintain, since `getindex`/iteration already needs the block's
position, not a count -- big-endian specifically (*not*
`reinterpret(UInt8, [UInt32(id)])`, which is native/little-endian on every platform this
runs on) so that RocksDB's own byte-lexicographic key ordering already matches ascending
numeric order -- iterating the column family directly (see `RocksDBDict`'s own "iteration
order is sorted key order") yields entries in the right sequence with no re-sorting after
the fact. `EngineStore` still can't use this scheme for its own per-field keys, since it
*does* share one column family across every engine struct field -- but nothing in this
package currently needs an appending/growing field there (a `DenseEngine{GraphBackend}`'s vectors
now live in their own `MMapMatrixDatabase` file, see below, not RocksDB blocks), so that
generic composite-key scheme has been removed rather than kept around unused.
"""
_be_key(id::Integer) = collect(reinterpret(UInt8, [hton(UInt32(id))]))

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
    save_field!(store::EngineStore, field::Symbol, value)

Writes just `value` under `field`'s key, leaving every other field's key untouched.
"""
save_field!(store::EngineStore, field::Symbol, value) = _save_key!(store.dict, String(field), value)

"""
    load_field(store::EngineStore, field::Symbol, default)

Reads back the value last saved under `field`'s key, or `default` if it was never saved.
"""
load_field(store::EngineStore, field::Symbol, default) = _load_key(store.dict, String(field), default)

"""
    has_field(store::EngineStore, field::Symbol) -> Bool
"""
has_field(store::EngineStore, field::Symbol) = haskey(store.dict, String(field))

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
# Dense vector persistence for SearchGraph -- NOT a RocksDB column family (unlike every
# other store in this module): a dedicated `SimilaritySearch.MMapMatrixDatabase` file
# alongside the project's RocksDB directory. Vectors are fixed-dimension dense Float32
# data, exactly what MMapMatrixDatabase is for -- its append-only counter/growth discipline
# gives in-order, crash-safe storage (see its own docstring) with no need for the
# batch-keying scheme AdjacencyStore/InvertedFileObjectStore use; durability itself is
# opt-in on that type (see its docstring's "Durability" section) and is arranged by
# `embedded.jl`'s `append_items!`, which flushes it right after every staged batch. And
# `getindex` on a memory-mapped file is far cheaper than a RocksDB point read (confirmed
# empirically against `VectorDatabase`/`BlockMatrixDatabase` in SimilaritySearch.jl's own
# test suite). This replaces the previous RocksDB-backed `DenseVectorStore`/`DENSE_DB_CF`.
# ---------------------------------------------------------

"""
    DENSE_VECTORS_FILENAME

Filename (relative to a project's directory, alongside its RocksDB files) of the
`SimilaritySearch.MMapMatrixDatabase` file backing a `DenseEngine{GraphBackend}`'s dense vectors.
"""
const DENSE_VECTORS_FILENAME = "dense_vectors.mmapdb"

"""
    dense_vectors_path(dir::String) -> String

The on-disk path of a project's dense-vector file (see [`open_dense_vectors`](@ref)).
"""
dense_vectors_path(dir::String) = joinpath(dir, DENSE_VECTORS_FILENAME)

"""
    open_dense_vectors(dir::String; read_only::Bool) -> Union{SimilaritySearch.MMapMatrixDatabase, Nothing}

Reopens this project's dense-vector file if it exists, or `nothing` if it doesn't -- which
is the normal state for a project that has never had a dense item appended yet (the file
is created lazily, on the very first append, since its dimension isn't known before then;
see `embedded.jl`'s `_searchgraph_on_change`), not an error condition.
"""
function open_dense_vectors(dir::String; read_only::Bool)
    path = dense_vectors_path(dir)
    isfile(path) || return nothing
    MMapMatrixDatabase(path; read_only)
end

"""
    load_dense_vector_blocks(dense_db) -> Vector

Reads back every vector in `dense_db` (as returned by [`open_dense_vectors`](@ref), or
`nothing`) as a single `(sp=1, ep=n, vectors)` block -- the same shape
`IndexEngine.build_searchgraph` already expects from a `vector_blocks` sequence (see
`Persistence`'s older RocksDB-block scheme this replaces), except there's exactly one
block here: an mmap'd file has no batch boundaries of its own worth preserving, unlike the
old per-append-call RocksDB blocks. Returns an empty `Vector` (no blocks at all) for
`dense_db === nothing` (nothing ever appended).

Copies every vector out of the memory-mapped file (`copy(dense_db[i])`, not the raw view
`getindex` normally returns) since these vectors are handed to a fresh, independent, RAM-
backed `VectorDatabase` that must outlive this read and never alias the mapped memory.
"""
function load_dense_vector_blocks(dense_db)
    dense_db === nothing && return []
    n = length(dense_db)
    n == 0 && return []
    return [(sp=1, ep=n, vectors=[copy(dense_db[i]) for i in 1:n])]
end

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
const INVFILE_DB_CF = "sparse_db"

"""
    InvertedFileObjectStore

An append-only sequence of raw indexed objects (see `IndexEngine.invertedfile_objects`)
for one `FullTextEngine`, backed by its own RocksDB column family
([`INVFILE_DB_CF`](@ref)) -- kept separate from [`EngineStore`](@ref) for the same reason
[`AdjacencyStore`](@ref) is: this can grow to one entry per indexed object, and shouldn't
share a keyspace with the handful of other, small engine fields.

Unlike a `SearchGraph`, an inverted file's posting lists have no direct/reverse-link
split to worry about -- `push_item!` fully finalizes each object's contribution before
`LOG` even fires (see `SimilaritySearch.CallbackLog`'s docstring), so what's saved here is
simply every object ever indexed, in insertion order; reloading rebuilds the whole index
by replaying them through the library's own `push_item!` again (see
`IndexEngine.build_bm25invertedfile`/`build_textinvertedfile`) rather than trying to persist
posting lists directly.

!!! warning "Scaling: a rebuild-by-reinsertion design, not an on-disk index format"
    Reloading this way means every raw object saved here, and the whole rebuilt index,
    has to fit in memory at once, and reloading costs the full original indexing time
    over again (proportional to the object count) rather than being O(1) or streamed.
    That's a fine trade for collections up to a few million documents -- it is *not* a
    design for billion-document corpora, which need genuinely disk-backed posting lists
    (paged/streamed from disk on demand, never fully materialized in memory) -- a
    different architecture this project does not attempt.

Keyed directly by each block's own starting id (`sp`, see [`_be_key`](@ref)) rather than a
generic block-count-and-prefix scheme: its own dedicated column family has no need for
that extra indirection, the id *is* the key, and iterating the column family already
yields blocks back in ascending `sp` order for free (see [`_be_key`](@ref)'s docstring).
`IndexEngine.invertedfile_objects(index, sp, ep)` returns a plain `Vector` with no `sp` of
its own, so [`append_objects!`](@ref) takes `sp` as a separate argument instead.

Shares its underlying `RocksDB.DB` connection, same as [`EngineStore`](@ref); no separate
`close` of its own to call.
"""
struct InvertedFileObjectStore
    dict::RocksDB.RocksDBDict{Vector{UInt8}, Any}
end

"""
    open_invertedfile_object_store(db::RocksDB.DB) -> InvertedFileObjectStore

Wraps `db`'s [`INVFILE_DB_CF`](@ref) column family (which must already be open on `db` --
see `Project.open_project`'s `extra_cf_names`) as an [`InvertedFileObjectStore`](@ref).
"""
open_invertedfile_object_store(db::RocksDB.DB) = InvertedFileObjectStore(RocksDB.RocksDBDict{Vector{UInt8}, Any}(db, INVFILE_DB_CF))

"""
    append_objects!(store::InvertedFileObjectStore, sp::Integer, objects::Vector)

Saves `objects` (the raw indexed objects for one `push_item!`/`append_items!` report over
range `sp:ep` -- see `IndexEngine.invertedfile_objects`) under `sp` as key (see
[`_be_key`](@ref)) -- never rewriting any earlier block. See [`load_object_blocks`](@ref)
to read the whole sequence back in order.
"""
function append_objects!(store::InvertedFileObjectStore, sp::Integer, objects::Vector)
    store.dict[_be_key(sp)] = objects
    return nothing
end

"""
    load_object_blocks(store::InvertedFileObjectStore) -> Vector{<:Vector}

Every block [`append_objects!`](@ref) has written, in the order they were appended -- a
plain iteration over `store.dict` already comes back in ascending `sp` order (see
[`_be_key`](@ref)'s docstring), so this needs no separate counter key or re-sorting.
"""
load_object_blocks(store::InvertedFileObjectStore) = [objects for (_, objects) in store.dict]

# ---------------------------------------------------------
# Staged (raw, not-yet-encoded) text persistence for FullTextEngine -- their
# own RocksDB column family, holding what append_items! stages before any Vocabulary
# exists to encode it against. Structurally identical to InvertedFileObjectStore (same
# block-keyed-by-sp scheme, see _be_key) but a different concern: InvertedFileObjectStore
# holds already-encoded objects (bags-of-words/SparseVectors), written from inside
# CallbackLog once push_item! has fully indexed them; StagedTextStore holds plain raw
# strings, written directly by embedded.jl's append_items! at stage time -- durable the
# instant they're staged, before any Vocabulary/encoding work happens, the same treatment
# append_items! already gives a DenseEngine{GraphBackend}'s raw vectors (see DENSE_VECTORS_FILENAME
# above), just via RocksDB instead of an mmap file since text isn't fixed-size Float32 data.
# ---------------------------------------------------------

"""
    STAGED_TEXT_CF

Name of the RocksDB column family a [`StagedTextStore`](@ref) is backed by. Pass this in
`Project.open_project`'s `extra_cf_names` alongside `ENGINE_CF`/`ADJACENCY_CF`/
`INVFILE_DB_CF`, for the same reason (every column family that exists on disk must be
listed on every open, not just the first).
"""
const STAGED_TEXT_CF = "staged_text"

"""
    StagedTextStore

An append-only sequence of raw, not-yet-encoded text blocks for one `FullTextEngine`/
`FullTextEngine`, backed by its own RocksDB column family ([`STAGED_TEXT_CF`](@ref)) --
the text-engine counterpart of a `DenseEngine{GraphBackend}`'s `dense_vectors.mmapdb`: every item
`add_item!`/`append_items!` has ever staged (see `IndexEngine.FullTextEngine`/
`IndexEngine.FullTextEngine`'s `staged` field), whether or not
`IndexEngine.index!(engine::IndexEngine.FullTextEngine)` has caught it up into the real
`BM25InvertedFile`/`InvertedFile` yet.

Keyed by each block's own starting position (`sp`, see [`_be_key`](@ref)) exactly like
[`InvertedFileObjectStore`](@ref) -- same reasoning: no prefix/counter needed, and
iteration already comes back in ascending order.

Shares its underlying `RocksDB.DB` connection, same as [`EngineStore`](@ref); no separate
`close` of its own to call.
"""
struct StagedTextStore
    dict::RocksDB.RocksDBDict{Vector{UInt8}, Any}
end

"""
    open_staged_text_store(db::RocksDB.DB) -> StagedTextStore

Wraps `db`'s [`STAGED_TEXT_CF`](@ref) column family (which must already be open on `db` --
see `Project.open_project`'s `extra_cf_names`) as a [`StagedTextStore`](@ref).
"""
open_staged_text_store(db::RocksDB.DB) = StagedTextStore(RocksDB.RocksDBDict{Vector{UInt8}, Any}(db, STAGED_TEXT_CF))

"""
    append_staged_texts!(store::StagedTextStore, sp::Integer, texts::Vector{String})

Saves `texts` (one `append_items!` batch's worth of freshly staged raw text) under `sp`
as key (see [`_be_key`](@ref)) -- never rewriting any earlier block. See
[`load_staged_text_blocks`](@ref) to read the whole sequence back in order.
"""
function append_staged_texts!(store::StagedTextStore, sp::Integer, texts::Vector{String})
    store.dict[_be_key(sp)] = texts
    return nothing
end

"""
    load_staged_text_blocks(store::StagedTextStore) -> Vector{Vector{String}}

Every block [`append_staged_texts!`](@ref) has written, in the order they were appended --
a plain iteration over `store.dict` already comes back in ascending `sp` order (see
[`_be_key`](@ref)'s docstring), so this needs no separate counter key or re-sorting. A
caller wants `vcat(load_staged_text_blocks(store)...)` for the flat, restore-ready
`Vector{String}` `IndexEngine.restore_engine` expects as `state.staged`.
"""
load_staged_text_blocks(store::StagedTextStore) = [texts for (_, texts) in store.dict]

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
interop format `load` (or any non-Julia Avro reader) consumes. Mirrors
`Schema.MetadataRecord` field-for-field (all concrete types, so they go straight into
Avro columns -- confirmed empirically -- rather than through a JSON-encoded
`declared_json` blob the way the pre-`Schema` redesign version of this struct needed) plus
`meta_json` for the one part that *can't* be a native Avro column (`meta`'s values are
heterogeneous/`Any`-typed, which `Avro.jl`'s schema inference rejects outright -- see
`Schema.MetadataRecord`'s own docstring) and `tombstoned`.

Deliberately does *not* carry `vector`/`text`: unlike the pre-`Schema` redesign version of
this struct (where `MetadataRecord.extra` held the complete original raw item, `vector`/
`text` included, making `declared_json`/`extra_json` alone a lossless copy), `meta` here
never contains `vector`/`text` at all -- both live only in the engine's own dedicated,
1..n-indexed column family (see `EngineStore`/`InvertedFileObjectStore`). A caller wiring
up a real `execute_dump`/`execute_load` on top of this must separately export/import that
column family (already exactly what `EngineStore`'s own block-based fields are for) for a
dump bundle to be losslessly restorable -- this struct alone is the metadata half only.

# Fields
- `_id::Int32`: matches [`Schema.MetadataRecord`](@ref)'s `_id` -- the index position
  `add_item!`/`push_item!` assigned it.
- `schema_version::Int`: copied from the record, see [`Schema.MetadataRecord`](@ref).
- `doc_id::Union{String,Nothing}`: copied from the record.
- `keywords::Vector{String}`: copied from the record.
- `refs::Vector{String}`: copied from the record.
- `meta_json::Union{String,Nothing}`: the record's `meta`, already JSON-encoded (see
  `Schema.encode_meta`/`Project.get_raw_meta`) -- `nothing` if this `_id` had no `meta`
  stored at all (not the same as an empty JSON object).
- `tombstoned::Bool`: whether this `_id` was in the engine's `deleted_ids` at dump time.
  `load` restores tombstones from the copied JLD2 snapshot's own `deleted_ids` (already
  authoritative, see `Server.handle_delete_item`), not by recomputing them from this flag
  -- it's carried here purely for a non-Julia consumer that only has the Avro file.
"""
struct DumpRecord
    _id::Int32
    schema_version::Int
    doc_id::Union{String,Nothing}
    keywords::Vector{String}
    refs::Vector{String}
    meta_json::Union{String,Nothing}
    tombstoned::Bool
end

"""
    DumpRecord(record::Schema.MetadataRecord, meta_json::Union{String,Nothing}, tombstoned::Bool) -> DumpRecord

Convenience constructor for a real `execute_dump` caller: pulls every
[`Schema.MetadataRecord`](@ref) field across as-is, pairing it with its already-encoded
`meta_json` (e.g. `Project.get_raw_meta(project, record._id)`) and `tombstoned` flag.
"""
DumpRecord(record::Schema.MetadataRecord, meta_json::Union{String,Nothing}, tombstoned::Bool) =
    DumpRecord(record._id, record.schema_version, record.doc_id, record.keywords, record.refs, meta_json, tombstoned)

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
_optional_string(x) = x === missing ? nothing : String(x)

function read_dump_records(filepath::String)
    return [
        DumpRecord(
            Int32(row._id),
            Int(row.schema_version),
            _optional_string(row.doc_id),
            Vector{String}(row.keywords),
            Vector{String}(row.refs),
            _optional_string(row.meta_json),
            Bool(row.tombstoned),
        )
        for row in Avro.readtable(filepath)
    ]
end

end # module
