module Persistence

using Avro
using RocksDB
using SimilaritySearch
using TextSearch
using SparseArrays: SparseArrays, AbstractSparseVector
using ..Schema
import ..IndexEngine
using ..Errors

export EngineStore, ENGINE_CF, open_engine_store, save_field!, load_field, has_field, save_fields!
export AdjacencyStore, ADJACENCY_CF, open_adjacency_store, save_neighbors!, load_neighbors
export DENSE_VECTORS_FILENAME, dense_vectors_path, open_dense_vectors, load_dense_vector_blocks
export InvertedFileObjectStore, INVFILE_DB_CF, open_invertedfile_object_store, append_objects!, load_object_blocks
export StagedTextStore, STAGED_TEXT_CF, open_staged_text_store, append_staged_texts!, load_staged_text_blocks
export InvertedIndexStore, INVFILE_POSTINGS_CF, INVFILE_DOCVECS_CF, open_inverted_index_store, has_inverted_index,
       write_invfile_block!, load_invfile_docvecs, docvec_nnz, DocVec, WeightedDocVec, BagDocVec
export LazyPostings, trim_postings_cache!, invalidate_postings!
export DumpRecord, write_dump_records, read_dump_records

# ---------------------------------------------------------
# Shared low-level primitives, operating directly on a RocksDBDict -- every named store
# below (EngineStore, AdjacencyStore, InvertedFileObjectStore) is a thin wrapper around
# one of these, scoped to its own column family.
# ---------------------------------------------------------

_save_key!(dict, key::String, value) = (dict[key] = value; nothing)
_load_key(dict, key::String, default) = get(dict, key, default)

# The one key encoding, defined once in `Schema.be_key` (which documents why it is big-endian)
# and aliased here so this module's own call sites read as they always did.
const _be_key = Schema.be_key
const _decode_be_key = Schema.decode_be_key


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
for one inverted-file-backed project -- a `FullTextEngine` or a `SparseEngine` -- backed by its
own RocksDB column family
([`INVFILE_DB_CF`](@ref)) -- kept separate from [`EngineStore`](@ref) for the same reason
[`AdjacencyStore`](@ref) is: this can grow to one entry per indexed object, and shouldn't
share a keyspace with the handful of other, small engine fields.

Unlike a `SearchGraph`, an inverted file's posting lists have no direct/reverse-link
split to worry about -- `push_item!` fully finalizes each object's contribution before
`LOG` even fires (see `SimilaritySearch.CallbackLog`'s docstring), so what's saved here is
simply every object ever indexed, in insertion order; reloading rebuilds the whole index
by replaying them through the library's own `append_items!` again (see
`IndexEngine.build_sparseinvertedfile`) rather than trying to persist posting lists directly.

**Only a sparse project still works this way.** A text project persists its index instead --
see [`InvertedIndexStore`](@ref) -- which is what removed the rebuild from its every open.

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

Every block [`append_objects!`](@ref) has written, in the order they were appended -- which is
ascending `sp`, decoded from each key rather than taken from the order the iteration happens to
produce.

The two coincide today, because [`_be_key`](@ref) is big-endian and RocksDB compares keys
bytewise (see its docstring). Sorting anyway is three lines and removes the dependency: a block
order that silently depended on the key encoding would corrupt a project's document numbering,
not fail it, and would do so only past the first 256 blocks -- exactly the kind of bug a small
test corpus never reaches.
"""
function load_object_blocks(store::InvertedFileObjectStore)
    blocks = [(_decode_be_key(k), objects) for (k, objects) in store.dict]
    sort!(blocks; by=first)
    [objects for (_, objects) in blocks]
end

# ---------------------------------------------------------
# Staged (raw, not-yet-encoded) text persistence for a text project -- its
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

An append-only sequence of raw, not-yet-encoded text blocks for one `FullTextEngine`,
backed by its own RocksDB column family ([`STAGED_TEXT_CF`](@ref)) --
the text-engine counterpart of a `DenseEngine{GraphBackend}`'s `dense_vectors.mmapdb`: every item
`add_item!`/`append_items!` has ever staged (see `IndexEngine.FullTextEngine`'s
`staged` field), whether or not
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
    isempty(records) && invalid_option(:records, "write_dump_records: cannot write an Avro file with zero rows (nothing to infer a schema from)")
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


# ---------------------------------------------------------
# Persisted BM25 index: posting lists and per-document term vectors
# ---------------------------------------------------------

"""
    INVFILE_POSTINGS_CF
    INVFILE_DOCVECS_CF

Names of the two RocksDB column families a [`InvertedIndexStore`](@ref) is backed by. Pass them
in `Project.open_project`'s `extra_cf_names` alongside the other stores' own, for the same
reason (every column family that exists on disk must be listed on every open, not just the
first).

They hold the *index*, not the objects that were indexed: `INVFILE_POSTINGS_CF` maps a token id
to the ascending ids of the documents carrying it, and `INVFILE_DOCVECS_CF` maps a document id
to the vector it was indexed as -- term frequencies under BM25, model weights under the
weighted backend, each value tagged with which (see [`_encode_docvec`](@ref)). [`InvertedFileObjectStore`](@ref), which a BM25 project used to
write instead, stores the raw bags-of-words and leaves the whole index to be *recomputed* on
every open -- 20.4s for 265k paragraphs of Project Gutenberg, measured 2026-09-11, growing
linearly. Reading these two back costs 1.1s for the same corpus, and the posting half of it
need not be read at all (see [`LazyPostings`](@ref)).
"""
const INVFILE_POSTINGS_CF = "invfile_postings"

"$(INVFILE_POSTINGS_CF)'s companion, keyed by document id. See [`INVFILE_POSTINGS_CF`](@ref)."
const INVFILE_DOCVECS_CF = "invfile_docvecs"

"""
    DocVec
    WeightedDocVec

The two shapes a document is stored as, one per text backend: `BM25InvertedFile` keeps integer
term frequencies (`bm25score` recomputes the query-document intersection from them), while the
weighted backend keeps the model's `Float32` weights. Both are named here because this module
writes and reconstructs them, and the first lives several modules deep inside
`SimilaritySearch.jl`.
"""
const DocVec = SimilaritySearch.Special.Sparse.SparseVecView{Vector{Int32},Vector{UInt32}}

"See [`DocVec`](@ref)."
const WeightedDocVec = SparseArrays.SparseVector{Float32,Int32}

"""
    BagDocVec

The third shape, and the reason the stored values carry a tag: a weighted project under a *set*
distance (`Dist.Sets.*`) indexes bags of words rather than weighted vectors, because those
distances read set membership and not weights at all. `TextSearch.jl` picks the encoding from
the distance, so the store has to accept whichever one the project ended up with.
"""
const BagDocVec = Dict{UInt32,Int32}

const _DOCVEC_TAG_FREQ = 0x01
const _DOCVEC_TAG_WEIGHT = 0x02
const _DOCVEC_TAG_BAG = 0x03

"""
    InvertedIndexStore

The two column families of [`INVFILE_POSTINGS_CF`](@ref)/[`INVFILE_DOCVECS_CF`](@ref) as one
handle, sharing the project's own `RocksDB.DB` connection like every other store here.
"""
struct InvertedIndexStore
    db::RocksDB.DB
    postings::RocksDB.ColumnFamily
    docvecs::RocksDB.ColumnFamily
end

"""
    open_inverted_index_store(db::RocksDB.DB) -> InvertedIndexStore

Wraps `db`'s [`INVFILE_POSTINGS_CF`](@ref) and [`INVFILE_DOCVECS_CF`](@ref) column families (both
of which must already be open on `db`) as a [`InvertedIndexStore`](@ref).
"""
open_inverted_index_store(db::RocksDB.DB) =
    InvertedIndexStore(db, db.column_families[INVFILE_POSTINGS_CF], db.column_families[INVFILE_DOCVECS_CF])

_encode_ids(ids::AbstractVector{<:Integer}) = Vector{UInt8}(reinterpret(UInt8, Vector{UInt32}(ids)))
_decode_ids(bytes::Vector{UInt8}) = Vector{UInt32}(reinterpret(UInt32, bytes))

"""
    _encode_docvec(v) -> Vector{UInt8}
    _decode_docvec(bytes) -> DocVec

A document's term vector as `(n::Int32, nnz::Int32, nzind::Vector{Int32}, nzval::Vector{UInt32})`,
written by hand rather than through a generic serializer.

This is the high-volume, hot path of the whole store -- one value per document, read once per
scored candidate -- which is exactly the case where a generic object-graph format is not worth
its overhead (the same reasoning that keeps the vocabulary in a hand-written format rather
than JLD2).
"""
function _encode_docvec(v::DocVec)
    io = IOBuffer()
    write(io, _DOCVEC_TAG_FREQ)
    write(io, Int32(v.n))
    write(io, Int32(length(v.nzind)))
    write(io, reinterpret(UInt8, Vector{Int32}(v.nzind)))
    write(io, reinterpret(UInt8, Vector{UInt32}(v.nzval)))
    take!(io)
end

function _encode_docvec(v::AbstractSparseVector)
    io = IOBuffer()
    write(io, _DOCVEC_TAG_WEIGHT)
    write(io, Int32(length(v)))
    write(io, Int32(length(SparseArrays.nonzeroinds(v))))
    write(io, reinterpret(UInt8, Vector{Int32}(SparseArrays.nonzeroinds(v))))
    write(io, reinterpret(UInt8, Vector{Float32}(SparseArrays.nonzeros(v))))
    take!(io)
end

function _encode_docvec(v::AbstractDict)
    io = IOBuffer()
    write(io, _DOCVEC_TAG_BAG)
    write(io, Int32(length(v)))
    write(io, reinterpret(UInt8, UInt32[UInt32(k) for k in keys(v)]))
    write(io, reinterpret(UInt8, Int32[Int32(x) for x in values(v)]))
    take!(io)
end

function _decode_docvec(bytes::Vector{UInt8})
    io = IOBuffer(bytes)
    tag = read(io, UInt8)
    if tag == _DOCVEC_TAG_BAG
        nnz = read(io, Int32)
        ids = Vector{UInt32}(undef, nnz)
        read!(io, ids)
        freqs = Vector{Int32}(undef, nnz)
        read!(io, freqs)
        return BagDocVec(zip(ids, freqs))
    end
    n = read(io, Int32)
    nnz = read(io, Int32)
    nzind = Vector{Int32}(undef, nnz)
    read!(io, nzind)
    if tag == _DOCVEC_TAG_FREQ
        nzval = Vector{UInt32}(undef, nnz)
        read!(io, nzval)
        return DocVec(Int(n), nzind, nzval)
    end
    tag == _DOCVEC_TAG_WEIGHT ||
        corrupted_storage("unknown document-vector tag $(repr(tag)) in $(INVFILE_DOCVECS_CF)")
    nzval = Vector{Float32}(undef, nnz)
    read!(io, nzval)
    SparseArrays.sparsevec(nzind, nzval, Int(n))
end

"""
    has_inverted_index(store::InvertedIndexStore) -> Bool

Whether anything has ever been written to this store -- the question `open_project` asks to
tell a project whose index is persisted from one written before that existed (which has to be
migrated, once) or one that is simply empty.
"""
function has_inverted_index(store::InvertedIndexStore)
    for _ in RocksDB.DBIterator(store.db; cf=store.docvecs)
        return true
    end
    return false
end

"""
    write_invfile_block!(store::InvertedIndexStore, base::Integer, docvecs, postings::Dict{UInt32,Vector{UInt32}})

Persists one freshly indexed block: the `docvecs` of documents `base+1 … base+length(docvecs)`,
and the *new* document ids each token in `postings` gained (already offset to global ids and
ascending).

The posting half is a read-modify-write per touched token -- get the token's current list,
append, put -- which is what makes an incremental `index!` possible at all: a block adds
documents to the lists of the tokens it happens to contain, and nothing else in the index
changes. The whole block goes in one `WriteBatch`, so a crash mid-block leaves the store at
the previous block's boundary rather than half-updated.

Appending is correct only because document ids grow monotonically: every id in this block is
larger than every id already in any list, so appending preserves the ascending order the
merge in `TextSearch.jl`'s search relies on.
"""
function write_invfile_block!(store::InvertedIndexStore, base::Integer, docvecs, postings::Dict{UInt32,Vector{UInt32}})
    b = RocksDB.WriteBatch()
    for (i, v) in enumerate(docvecs)
        RocksDB.put!(b, _be_key(base + i), _encode_docvec(v); cf=store.docvecs)
    end
    for (token, newids) in postings
        raw = RocksDB.get(store.db, _be_key(token); cf=store.postings)
        merged = raw === nothing ? Vector{UInt32}(newids) : vcat(_decode_ids(raw), UInt32.(newids))
        RocksDB.put!(b, _be_key(token), _encode_ids(merged); cf=store.postings)
    end
    RocksDB.write!(store.db, b)
    return nothing
end

"""
    docvec_nnz(v) -> Int

How many terms a stored document vector holds, whichever of the three shapes it is -- what the
weighted backend keeps in its `sizes` array, recomputed at assembly instead of stored twice.
"""
docvec_nnz(v::AbstractDict) = length(v)
docvec_nnz(v::AbstractSparseVector) = length(SparseArrays.nonzeroinds(v))
docvec_nnz(v) = length(v.nzind)

"""
    load_invfile_docvecs(store::InvertedIndexStore) -> Vector{DocVec}

Every persisted document vector, in document-id order.

Kept resident, unlike the posting lists, and that asymmetry is the measured heart of this
design rather than an accident. `TextSearch.jl` scores a candidate by reading its term vector
(`bm25score(..., idx.db[docID])` in `onmatch!`), and the merge makes *every* document holding
*any* query token a candidate -- so a query containing a frequent token turns a lazy `db` into
one point read per document in the collection. Measured on 265k Gutenberg paragraphs
(2026-09-11): 66.8 ms per query with the vectors resident, 1771.9 ms with them read on
demand, for the same query and the same index. The posting lists, read once per query *term*,
cost 1.8% (see [`LazyPostings`](@ref)).

Loading them back costs 0.76s for that corpus, against 20.4s to recompute the index from the
raw objects.
"""
function load_invfile_docvecs(store::InvertedIndexStore)
    ids = Int[]
    out = nothing
    for (k, v) in RocksDB.DBIterator(store.db; cf=store.docvecs)
        d = _decode_docvec(v)
        # The element type follows the first value's tag rather than being passed in: the store
        # says what it holds, so a project cannot be reassembled under the wrong backend by
        # mistake -- the index it builds would simply not accept the vectors.
        out === nothing && (out = Vector{typeof(d)}())
        push!(ids, Int(_decode_be_key(k)))
        push!(out, d)
    end
    out === nothing && return DocVec[]
    # Placed by the id in the key, not by the order the iteration produced them. Those agree
    # under big-endian keys, and a document vector filed under the wrong id would corrupt every
    # score rather than fail -- see [`load_object_blocks`](@ref) for the same reasoning.
    issorted(ids) ? out : out[sortperm(ids)]
end

"""
    LazyPostings(store::InvertedIndexStore, vocsize::Int, maxlists::Int, baselists::Int)

A `BM25InvertedFile`'s posting lists, read from RocksDB on demand instead of held in memory,
with a bounded cache in front.

This is the extension point `SimilaritySearch.jl` documents: an `AbstractAdjList{UInt32}`
needs only `neighbors`/`neighbors_length`/`eachindex`, so the search code neither knows nor
cares that a list came from disk. Being lazy here is close to free -- a query reads one list
per *term*, not one per candidate -- which is why this half is lazy and the document vectors
are not (see [`load_invfile_docvecs`](@ref)). Measured on 265k Gutenberg paragraphs, 2026-09-11:
66.8 ms per query fully resident against 68.0 ms with these lists on disk, while the resident
lists cost 55.8 MB of the index's 237.5 MB.

# The cache, and why trimming is not part of a read

`cache` holds decoded lists; `uses` counts how often each was asked for. When
`trim_postings_cache!` finds more than `maxlists` entries it evicts the least frequently used
down to `baselists`, so the cost is paid in bulk rather than on the unlucky read that crossed
the threshold, and a hot list is never evicted by a burst of one-off lookups within a single
query.

**Eviction happens only between queries, never inside one.** `trim_postings_cache!` is called
by the engine once a search has fully resolved; a read never evicts. The reason is
correctness under concurrency as much as latency: several searches run against this index at
once (they take a read lock, not an exclusive one), and a list evicted mid-query would be
re-read and re-decoded by whichever of them was still walking it.

Counting *lists* rather than bytes is deliberate too: what the cache is protecting against is
re-reading and re-decoding, and that cost tracks the number of lookups, not their size.
"""
mutable struct LazyPostings <: SimilaritySearch.AbstractAdjList{UInt32}
    store::InvertedIndexStore
    n::Int
    cache::Dict{UInt32,Vector{UInt32}}
    uses::Dict{UInt32,Int}
    lk::Threads.ReentrantLock
    maxlists::Int
    baselists::Int
end

function LazyPostings(store::InvertedIndexStore, vocsize::Integer, maxlists::Integer, baselists::Integer)
    baselists <= maxlists ||
        invalid_option(:postings_cache_base, "the posting cache's base size ($baselists) cannot exceed its maximum ($maxlists)")
    LazyPostings(store, Int(vocsize), Dict{UInt32,Vector{UInt32}}(), Dict{UInt32,Int}(),
                 Threads.ReentrantLock(), Int(maxlists), Int(baselists))
end

function SimilaritySearch.neighbors(a::LazyPostings, i)
    token = UInt32(i)
    hit = lock(a.lk) do
        cached = get(a.cache, token, nothing)
        cached === nothing || (a.uses[token] = get(a.uses, token, 0) + 1)
        cached
    end
    hit === nothing || return hit

    raw = RocksDB.get(a.store.db, _be_key(token); cf=a.store.postings)
    ids = raw === nothing ? UInt32[] : _decode_ids(raw)
    lock(a.lk) do
        a.cache[token] = ids
        a.uses[token] = get(a.uses, token, 0) + 1
    end
    ids
end

SimilaritySearch.neighbors_length(a::LazyPostings, i) = length(SimilaritySearch.neighbors(a, i))
Base.eachindex(a::LazyPostings) = Base.OneTo(a.n)
Base.length(a::LazyPostings) = a.n

SimilaritySearch.add!(::LazyPostings, args...) =
    unsupported_operation(:add!, "LazyPostings is read-only from the index's side: a block's new postings are written " *
          "by Persistence.write_invfile_block! and reach this list through invalidate_postings!")

"""
    trim_postings_cache!(a::LazyPostings) -> Int

Evicts the least frequently used cached lists down to `a.baselists`, but only if there are
more than `a.maxlists` of them; returns how many it dropped.

Call it after a query (or a batch of them) has fully resolved -- never from inside one. See
[`LazyPostings`](@ref).
"""
function trim_postings_cache!(a::LazyPostings)
    lock(a.lk) do
        length(a.cache) <= a.maxlists && return 0
        victims = sort!(collect(keys(a.cache)); by=t -> get(a.uses, t, 0))
        ndrop = length(a.cache) - a.baselists
        for t in view(victims, 1:ndrop)
            delete!(a.cache, t)
            delete!(a.uses, t)
        end
        ndrop
    end
end

"""
    invalidate_postings!(a::LazyPostings, tokens)

Drops the cached lists of `tokens`, which a just-written block has lengthened on disk.

Dropping rather than patching: a token whose list just grew is not necessarily one anybody
queries, and the next read pays a single point lookup either way.
"""
function invalidate_postings!(a::LazyPostings, tokens)
    lock(a.lk) do
        for t in tokens
            delete!(a.cache, UInt32(t))
        end
    end
    return nothing
end

# The two hooks `IndexEngine` leaves open for an adjacency list that lives somewhere other than
# memory (see its `persist_block!`/`maybe_trim_cache!`): this module is loaded after that one
# precisely so it can add these methods without `IndexEngine` ever naming RocksDB.
function IndexEngine.persist_block!(a::LazyPostings, base::Int, docvecs, postings::Dict{UInt32,Vector{UInt32}})
    write_invfile_block!(a.store, base, docvecs, postings)
    invalidate_postings!(a, keys(postings))
    return nothing
end

IndexEngine.maybe_trim_cache!(a::LazyPostings) = trim_postings_cache!(a)

end # module
