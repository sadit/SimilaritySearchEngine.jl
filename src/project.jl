module Project

using RocksDB
using JSON3
using ..Schema

export ProjectManager, open_project, close_project, compact_all!, put_metadata!, get_metadata,
       get_meta, get_raw_meta, find_by_doc_id, find_all_by_doc_id, generate_id

"""
    DOCID_CF

Column family holding the secondary index from `doc_id` to `_id`. Its keys are
`doc_id`'s bytes followed by the 4 bytes of [`_id_key`](@ref), the same composite shape
`cf_idx_keywords`/`cf_idx_refs` already use, so a seek to a `doc_id` lands on the run of
entries that carry it -- see [`find_all_by_doc_id`](@ref). Composite because `doc_id` is not
unique: a plain `doc_id` key would keep only the last item appended under it.
"""
const DOCID_CF = "idx_docid"

"""
    DOCID_INDEX_MARKER

Sentinel key written into [`DOCID_CF`](@ref) once its backfill has run, so a project whose
`doc_id`s are all empty is not rescanned on every open. It starts with a NUL byte, which no
`doc_id` produced by a caller realistically does, and [`find_all_by_doc_id`](@ref) ignores
it anyway (its key is not `length(doc_id) + 4` bytes long).
"""
const DOCID_INDEX_MARKER = Vector{UInt8}("\0__docid_index_built__")

"""
    generate_id() -> String

Generates a collision-safe 64-bit identifier (16 hex chars).

# Returns
- `String`: A randomly generated hex string.
"""
function generate_id()
    return string(rand(UInt64), base=16, pad=16)
end

"""
    ProjectManager

Manages a project backed by RocksDB.

# Fields
- `dataset::String`: Unique identifier of the dataset stored in this project.
- `db::RocksDB.DB`: The main RocksDB database handle.
- `cf_records::RocksDB.ColumnFamily`: Column family for the fixed `Schema.MetadataRecord`
  half of each item (see [`put_metadata!`](@ref)).
- `cf_meta::RocksDB.ColumnFamily`: Column family for the free-form `meta` half -- kept
  apart from `cf_records` so a caller that only wants `meta` (or only the record) never
  pays to read/decode the other.
- `cf_op_log::RocksDB.ColumnFamily`: Column family for operation logs.
- `cf_idx_keywords::RocksDB.ColumnFamily`: Secondary index from a keyword value to the
  `_id`s of records carrying it (see [`put_metadata!`](@ref)).
- `cf_idx_refs::RocksDB.ColumnFamily`: Secondary index from a referenced `doc_id` to the
  `_id`s of records whose `refs` list it.

There is no per-project typed-field schema carried here anymore -- every project has
exactly these column families, always; only `keywords`/`refs` are indexed, and that's
fixed, not configurable per project.
"""
mutable struct ProjectManager
    dataset::String
    db::RocksDB.DB
    cf_records::RocksDB.ColumnFamily
    cf_meta::RocksDB.ColumnFamily
    cf_op_log::RocksDB.ColumnFamily
    cf_idx_keywords::RocksDB.ColumnFamily
    cf_idx_refs::RocksDB.ColumnFamily
    cf_idx_docid::RocksDB.ColumnFamily
end

"""
    open_project(path::String, dataset::String; read_only::Bool=false, extra_cf_names::Vector{String}=String[]) -> ProjectManager

Opens or creates a RocksDB-backed project at the given path.

# Arguments
- `path::String`: The directory path where the project will be stored.
- `dataset::String`: The unique identifier of the dataset held by this project.

# Keyword Arguments
- `read_only::Bool`: Opens without acquiring RocksDB's exclusive per-process write lock
  (`RocksDB.jl`'s `opendb(...; read_only=true)`) -- lets a read-only caller (e.g. the CLI
  `describe` command) inspect a project a `similarity-search-serve` process still has open
  for writing, instead of failing with a lock-contention error. Never pass this when the
  caller intends to write (e.g. `rebuild`): a read-only handle can't `put!`/`delete!`.
- `extra_cf_names::Vector{String}`: additional column family names to open alongside this
  module's own (`"records"`/`"meta"`/`"op_log"`/`"idx_keywords"`/`"idx_refs"`), for a
  caller (e.g. `IndexEngine`'s persistence layer) that wants to share this same RocksDB
  connection/directory for its own column family rather than opening a second `DB`.
  `Project` doesn't interpret these names at all -- it just makes sure RocksDB knows about
  them, since every column family that exists on disk must be listed at `opendb` time on
  *every* open, not only the first (see `RocksDB.create_column_family`'s docstring): a
  caller that creates one ad hoc after the fact and never adds it here would find the
  *next* `open_project` call failing.

# Returns
- `ProjectManager`: The initialized project manager object.
"""
function open_project(path::String, dataset::String; read_only::Bool=false, extra_cf_names::Vector{String}=String[])
    # Note: In a production environment, we would also need to gracefully handle
    # the known ColumnFamily handle leak in RocksDB.jl when closing/reopening frequently.
    cf_names = ["default", "records", "meta", "op_log", "idx_keywords", "idx_refs", DOCID_CF]
    append!(cf_names, extra_cf_names)

    db = read_only ?
        RocksDB.opendb(path, column_families=cf_names, read_only=true) :
        RocksDB.opendb(path, column_families=cf_names, create_if_missing=true, create_missing_column_families=true)

    cfs = db.column_families
    manager = ProjectManager(dataset, db, cfs["records"], cfs["meta"], cfs["op_log"],
                             cfs["idx_keywords"], cfs["idx_refs"], cfs[DOCID_CF])
    read_only || backfill_docid_index!(manager)
    return manager
end

"""
    backfill_docid_index!(manager::ProjectManager) -> Int

Populates [`DOCID_CF`](@ref) from `cf_records` for a project written before that index
existed, and returns how many entries it wrote (`0` when there was nothing to do).

Runs at most once per project: the marker key ([`DOCID_INDEX_MARKER`](@ref)) is written at
the end, and its presence is what later opens check -- not whether the column family holds
anything, which would rescan forever a project whose records all carry an empty `doc_id`.
The scan itself costs exactly one pass over `cf_records`, which is what a *single*
`doc_id` lookup used to cost before this index existed (see [`find_by_doc_id`](@ref)), so
the one-off migration is cheaper than the first lookup it replaces.

Never called on a `read_only` handle: it writes, and a read-only handle cannot. Such a
handle simply has no index to consult until a writer opens the project once.
"""
function backfill_docid_index!(manager::ProjectManager)
    RocksDB.get(manager.db, DOCID_INDEX_MARKER, cf=manager.cf_idx_docid) === nothing || return 0

    n = 0
    b = RocksDB.WriteBatch()
    for (_, v) in RocksDB.DBIterator(manager.db; cf=manager.cf_records)
        record = JSON3.read(String(v), MetadataRecord)
        isempty(record.doc_id) && continue
        RocksDB.put!(b, docid_key(record.doc_id, record._id), UInt8[], cf=manager.cf_idx_docid)
        n += 1
    end
    RocksDB.put!(b, DOCID_INDEX_MARKER, UInt8[], cf=manager.cf_idx_docid)
    RocksDB.write!(manager.db, b)
    return n
end

"""
    close_project(manager::ProjectManager)

Closes the project and its underlying RocksDB connection.

# Arguments
- `manager::ProjectManager`: The project manager to close.
"""
function close_project(manager::ProjectManager)
    # WARNING: RocksDB.jl currently leaks CF handles on close.
    # Documenting here as per design decisions.
    close(manager.db)
end

"""
    compact_all!(manager::ProjectManager)

Forces a full manual compaction of every column family in the project.

Every one of them, not just the default: `RocksDB.compact!(db)` without a `cf` compacts the
default column family alone, and a project keeps its records, metadata, secondary indexes and
the engine's own state in separate ones.

This is what turns a bulk write session's leftovers into flat SST files. Without it the next
open pays for replaying the write-ahead log, which is not a small effect: measured
2026-09-10 on a 50k-vector dense project, reopening took 2.97s before compacting and 0.28s
after, for a compaction that itself cost 0.03s.
"""
function compact_all!(manager::ProjectManager)
    for (_, cf) in manager.db.column_families
        RocksDB.compact!(manager.db; cf=cf)
    end
    return nothing
end

# `collect` materializes a concrete Vector{UInt8} rather than a lazy reinterpret view
# over a temporary array -- WriteBatch defers the actual write until write!(), and a lazy
# view isn't reliably kept alive/rooted across that gap, causing intermittent
# silently-dropped keys under GC pressure (confirmed via a targeted 200-key round-trip
# repro: a handful of keys came back missing on `get` every run, with materialized keys
# the issue disappears).
_id_key(id::Int32) = collect(reinterpret(UInt8, [id]))

"""
    docid_key(doc_id::AbstractString, id::Int32) -> Vector{UInt8}

The [`DOCID_CF`](@ref) key for `(doc_id, _id)`: `doc_id`'s bytes, then [`_id_key`](@ref)'s
4. Composite rather than plain `doc_id` because nothing in this package makes `doc_id`
unique -- two items may legitimately carry the same one -- and a plain key would silently
keep only the last of them.
"""
docid_key(doc_id::AbstractString, id::Int32) = vcat(Vector{UInt8}(String(doc_id)), _id_key(id))

"""
    put_metadata!(manager::ProjectManager, record::MetadataRecord, meta)

Inserts or updates `record` (and its accompanying free-form `meta`, if any -- see
[`Schema.metadata_record`](@ref)) in the project, keyed by `record._id` in both `cf_records`
and `cf_meta`. Also (re)writes `record`'s `keywords`/`refs` entries into their respective
secondary index column families -- this is *not* incremental: any previous index entries
for this `_id` are left in place (an update that removes a keyword/ref leaks a stale
index row pointing at an `_id` whose current record no longer carries it) -- acceptable
for the append-mostly, rarely-updated workloads this targets, not a general upsert.

# Arguments
- `manager::ProjectManager`: The target project manager.
- `record::MetadataRecord`: The fixed-shape record to insert or update.
- `meta`: The free-form data to store alongside it (a `Dict`, `Vector`, `nothing`, or
  anything else `Schema.encode_meta` accepts) -- omitted entirely from `cf_meta` when
  `nothing` (not written as a JSON `null`).
"""
function put_metadata!(manager::ProjectManager, record::MetadataRecord, meta)
    key_bytes = _id_key(record._id)
    record_bytes = JSON3.write(record)

    b = RocksDB.WriteBatch()
    RocksDB.put!(b, key_bytes, record_bytes, cf=manager.cf_records)
    meta === nothing || RocksDB.put!(b, key_bytes, Schema.encode_meta(meta), cf=manager.cf_meta)

    isempty(record.doc_id) ||
        RocksDB.put!(b, docid_key(record.doc_id, record._id), UInt8[], cf=manager.cf_idx_docid)

    for kw in record.keywords
        idx_key = vcat(Vector{UInt8}(kw), key_bytes)
        RocksDB.put!(b, idx_key, UInt8[], cf=manager.cf_idx_keywords)
    end
    for r in record.refs
        idx_key = vcat(Vector{UInt8}(r), key_bytes)
        RocksDB.put!(b, idx_key, UInt8[], cf=manager.cf_idx_refs)
    end
    RocksDB.write!(manager.db, b)
end

"""
    get_metadata(manager::ProjectManager, id::Integer) -> Union{MetadataRecord, Nothing}

Retrieves a [`Schema.MetadataRecord`](@ref) from the project by its internal `_id`. Does
*not* touch `cf_meta` -- use [`get_meta`](@ref)/[`get_raw_meta`](@ref) for that half.
"""
function get_metadata(manager::ProjectManager, id::Integer)
    val_bytes = get(manager.db, _id_key(Int32(id)), cf=manager.cf_records)
    val_bytes === nothing && return nothing
    return JSON3.read(String(val_bytes), MetadataRecord)
end

"""
    get_meta(manager::ProjectManager, id::Integer; lazy::Bool=true) -> Any

Retrieves and decodes the `meta` half for `id` via [`Schema.decode_meta`](@ref) (see its
docstring for the `lazy` tradeoff). `nothing` if `id` has no stored `meta` at all (either
it never had one, or `id` doesn't exist).
"""
function get_meta(manager::ProjectManager, id::Integer; lazy::Bool=true)
    val_bytes = get(manager.db, _id_key(Int32(id)), cf=manager.cf_meta)
    return Schema.decode_meta(val_bytes; lazy)
end

"""
    get_raw_meta(manager::ProjectManager, id::Integer) -> Union{String, Nothing}

The stored `meta` bytes for `id`, completely unparsed (see [`Schema.raw_meta`](@ref)) --
for a caller that only forwards `meta` unmodified (e.g. straight into an HTTP response).
"""
function get_raw_meta(manager::ProjectManager, id::Integer)
    val_bytes = get(manager.db, _id_key(Int32(id)), cf=manager.cf_meta)
    return Schema.raw_meta(val_bytes)
end

"""
    find_by_doc_id(manager::ProjectManager, doc_id::String) -> Union{MetadataRecord, Nothing}

Linear scan over `cf_records` looking for a record whose `doc_id` matches. There is no
secondary index from `doc_id` to `_id` in this pass, so this is the fallback path for
`fetch` requests that use the caller-supplied external id rather than the internal `_id`
-- acceptable for the project sizes this prototype targets, not meant as a hot-path
lookup. Only decodes the (small, fixed-shape) record for each row -- never touches
`cf_meta` -- so this is cheaper than the old `extra`-scanning version, but still O(n).
"""
function find_by_doc_id(manager::ProjectManager, doc_id::String)
    for (_, v) in RocksDB.DBIterator(manager.db; cf=manager.cf_records)
        record = JSON3.read(String(v), MetadataRecord)
        record.doc_id == doc_id && return record
    end
    return nothing
end

"""
    find_all_by_doc_id(manager::ProjectManager, doc_id::AbstractString) -> Vector{MetadataRecord}

Every record carrying exactly this `doc_id`, in ascending `_id` order.

A vector rather than a single record because nothing in this package makes `doc_id` unique:
it is a caller-chosen external id, and appending two items that share one is allowed. The old
`doc_id` path returned whichever of them a full scan happened to reach first, which is a
coin flip dressed as an answer.

Cost is the number of records it returns, not the number the project holds -- the point of
[`DOCID_CF`](@ref). It seeks to `doc_id`'s bytes and walks while they still prefix the key,
keeping only keys of exactly `length(doc_id) + 4` bytes: the longer ones belong to a
*different*, longer `doc_id` that happens to start the same way (`"doc_2"` walking over
`"doc_20"`'s entries), and they are skipped rather than decoded. A `doc_id` that prefixes
many others therefore costs a short walk over their keys -- never a record read, never a scan
of the collection.

Falls back to [`find_by_doc_id`](@ref)'s full scan when the index has not been built yet,
which happens only on a `read_only` handle opened against a project written before this index
existed: [`backfill_docid_index!`](@ref) fills it in on the first writable open.
"""
function find_all_by_doc_id(manager::ProjectManager, doc_id::AbstractString)
    isempty(doc_id) && return MetadataRecord[]
    if RocksDB.get(manager.db, DOCID_INDEX_MARKER, cf=manager.cf_idx_docid) === nothing
        record = find_by_doc_id(manager, String(doc_id))
        return record === nothing ? MetadataRecord[] : [record]
    end

    dbytes = Vector{UInt8}(String(doc_id))
    n = length(dbytes)
    ids = Int32[]
    it = RocksDB.DBIterator(manager.db; cf=manager.cf_idx_docid)
    RocksDB.seek!(it, dbytes)
    while RocksDB.valid(it)
        k = RocksDB.key(it)
        # Keys are sorted bytewise, so the first one that stops carrying this prefix ends
        # the run that could possibly contain this doc_id's entries.
        (length(k) >= n && view(k, 1:n) == dbytes) || break
        length(k) == n + 4 && push!(ids, only(reinterpret(Int32, k[end-3:end])))
        RocksDB.advance!(it)
    end

    sort!(ids)
    records = MetadataRecord[]
    for id in ids
        record = get_metadata(manager, id)
        record === nothing || push!(records, record)
    end
    return records
end

end # module
