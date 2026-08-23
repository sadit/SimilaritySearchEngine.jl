module Project

using RocksDB
using JSON3
using ..Schema

export ProjectManager, open_project, close_project, put_metadata!, get_metadata,
       get_meta, get_raw_meta, find_by_doc_id, generate_id

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
    cf_names = ["default", "records", "meta", "op_log", "idx_keywords", "idx_refs"]
    append!(cf_names, extra_cf_names)

    db = read_only ?
        RocksDB.opendb(path, column_families=cf_names, read_only=true) :
        RocksDB.opendb(path, column_families=cf_names, create_if_missing=true, create_missing_column_families=true)

    cfs = db.column_families
    return ProjectManager(dataset, db, cfs["records"], cfs["meta"], cfs["op_log"], cfs["idx_keywords"], cfs["idx_refs"])
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

# `collect` materializes a concrete Vector{UInt8} rather than a lazy reinterpret view
# over a temporary array -- WriteBatch defers the actual write until write!(), and a lazy
# view isn't reliably kept alive/rooted across that gap, causing intermittent
# silently-dropped keys under GC pressure (confirmed via a targeted 200-key round-trip
# repro: a handful of keys came back missing on `get` every run, with materialized keys
# the issue disappears).
_id_key(id::Int32) = collect(reinterpret(UInt8, [id]))

"""
    put_metadata!(manager::ProjectManager, record::MetadataRecord, meta=nothing)

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
function put_metadata!(manager::ProjectManager, record::MetadataRecord, meta=nothing)
    key_bytes = _id_key(record._id)
    record_bytes = JSON3.write(record)

    b = RocksDB.WriteBatch()
    RocksDB.put!(b, key_bytes, record_bytes, cf=manager.cf_records)
    meta === nothing || RocksDB.put!(b, key_bytes, Schema.encode_meta(meta), cf=manager.cf_meta)

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

end # module
