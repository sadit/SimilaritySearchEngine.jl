module Project

using RocksDB
using JSON3
using ..Schema

export ProjectManager, open_project, close_project, put_metadata!, get_metadata, find_by_original_id, generate_id

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
- `cf_meta::RocksDB.ColumnFamily`: Column family for metadata.
- `cf_op_log::RocksDB.ColumnFamily`: Column family for operation logs.
- `cf_secondary_indices::Dict{String, RocksDB.ColumnFamily}`: Secondary indices.
- `schema::MetaSchema`: In-memory schema definition.
"""
mutable struct ProjectManager
    dataset::String
    db::RocksDB.DB
    cf_meta::RocksDB.ColumnFamily
    cf_op_log::RocksDB.ColumnFamily
    # Secondary indices can be stored in a dict of field_name => ColumnFamily
    cf_secondary_indices::Dict{String, RocksDB.ColumnFamily}

    # Store schema in memory for quick checks
    schema::MetaSchema
end

"""
    open_project(path::String, dataset::String, schema::MetaSchema=MetaSchema(); read_only::Bool=false) -> ProjectManager

Opens or creates a RocksDB-backed project at the given path.

# Arguments
- `path::String`: The directory path where the project will be stored.
- `dataset::String`: The unique identifier of the dataset held by this project.
- `schema::MetaSchema`: The schema definition for the project's metadata.

# Keyword Arguments
- `read_only::Bool`: Opens without acquiring RocksDB's exclusive per-process write lock
  (`RocksDB.jl`'s `opendb(...; read_only=true)`) -- lets a read-only caller (e.g. the CLI
  `describe` command) inspect a project a `similarity-search-serve` process still has open
  for writing, instead of failing with a lock-contention error. Never pass this when the
  caller intends to write (e.g. `rebuild`): a read-only handle can't `put!`/`delete!`.
- `extra_cf_names::Vector{String}`: additional column family names to open alongside this
  module's own (`"meta"`/`"op_log"`/secondary indices), for a caller (e.g. `IndexEngine`'s
  persistence layer) that wants to share this same RocksDB connection/directory for its
  own column family rather than opening a second `DB`. `Project` doesn't interpret these
  names at all -- it just makes sure RocksDB knows about them, since every column family
  that exists on disk must be listed at `opendb` time on *every* open, not only the first
  (see `RocksDB.create_column_family`'s docstring): a caller that creates one ad hoc after
  the fact and never adds it here would find the *next* `open_project` call failing.

# Returns
- `ProjectManager`: The initialized project manager object.
"""
function open_project(path::String, dataset::String, schema::MetaSchema=MetaSchema(); read_only::Bool=false, extra_cf_names::Vector{String}=String[])
    # Note: In a production environment, we would also need to gracefully handle
    # the known ColumnFamily handle leak in RocksDB.jl when closing/reopening frequently.

    # Base column families
    cf_names = ["default", "meta", "op_log"]

    # Add column families for secondary indices declared in the schema
    for field in schema.fields
        if field.indexed
            push!(cf_names, "meta_idx_$(field.name)")
        end
    end

    append!(cf_names, extra_cf_names)

    # Open DB with all column families
    db = read_only ?
        RocksDB.opendb(path, column_families=cf_names, read_only=true) :
        RocksDB.opendb(path, column_families=cf_names, create_if_missing=true, create_missing_column_families=true)

    # Retrieve handles
    cfs = db.column_families

    cf_meta = cfs["meta"]
    cf_op_log = cfs["op_log"]

    cf_secondary = Dict{String, RocksDB.ColumnFamily}()
    for field in schema.fields
        if field.indexed
            cf_secondary[field.name] = cfs["meta_idx_$(field.name)"]
        end
    end

    return ProjectManager(dataset, db, cf_meta, cf_op_log, cf_secondary, schema)
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
    put_metadata!(manager::ProjectManager, record::MetadataRecord)

Inserts or updates a MetadataRecord in the project.

# Arguments
- `manager::ProjectManager`: The target project manager.
- `record::MetadataRecord`: The metadata record to insert or update.
"""
function put_metadata!(manager::ProjectManager, record::MetadataRecord)
    # Serialize record to JSON bytes (or Avro if fully integrating Avro.jl here)
    # For now, using JSON bytes for the prototype
    data_dict = Dict(
        "doc_id" => record.doc_id,
        "schema_version" => record.schema_version,
        "declared" => record.declared_fields,
        # String(::Vector{UInt8}) empties its argument as a side effect -- copy first
        # so `record.extra` stays intact for the caller after put_metadata! returns.
        "extra" => String(copy(record.extra))
    )
    bytes = Vector{UInt8}(JSON3.write(data_dict))

    # `collect` materializes a concrete Vector{UInt8} rather than a lazy reinterpret
    # view over a temporary array -- WriteBatch defers the actual write until write!(),
    # and a lazy view isn't reliably kept alive/rooted across that gap, causing
    # intermittent silently-dropped keys under GC pressure (confirmed via a targeted
    # 200-key round-trip repro: a handful of keys came back missing on `get` every run,
    # with materialized keys the issue disappears).
    key_bytes = collect(reinterpret(UInt8, [record.doc_id]))

    b = RocksDB.WriteBatch()
    RocksDB.put!(b, key_bytes, bytes, cf=manager.cf_meta)

    for (k, v) in record.declared_fields
        if haskey(manager.cf_secondary_indices, k)
            cf = manager.cf_secondary_indices[k]
            val_bytes = Vector{UInt8}(string(v))
            idx_key = vcat(val_bytes, key_bytes)
            RocksDB.put!(b, idx_key, UInt8[], cf=cf)
        end
    end
    RocksDB.write!(manager.db, b)
end

"""
    get_metadata(manager::ProjectManager, doc_id::Int) -> Union{MetadataRecord, Nothing}

Retrieves a MetadataRecord from the project by its document ID.

# Arguments
- `manager::ProjectManager`: The project manager.
- `doc_id::Int`: The integer document ID to retrieve.

# Returns
- `MetadataRecord`: The parsed record if found, or `nothing`.
"""
function get_metadata(manager::ProjectManager, doc_id::Integer)
    key_bytes = collect(reinterpret(UInt8, [Int(doc_id)]))
    val_bytes = get(manager.db, key_bytes, cf=manager.cf_meta)

    val_bytes === nothing && return nothing
    return _decode_metadata(val_bytes)
end

function _decode_metadata(val_bytes::Vector{UInt8})
    data_dict = JSON3.read(String(val_bytes), Dict{String, Any})
    extra_bytes = Vector{UInt8}(data_dict["extra"])

    return MetadataRecord(
        data_dict["doc_id"],
        data_dict["schema_version"],
        Dict{String, Any}(data_dict["declared"]),
        extra_bytes
    )
end

"""
    find_by_original_id(manager::ProjectManager, id_str::String) -> Union{MetadataRecord, Nothing}

Linear scan over the `meta` column family looking for a record whose caller-supplied
`"id"` field (declared or in `extra`) matches `id_str`. There is no secondary index from
external id to `doc_id` in this pass, so this is the fallback path for `fetch` requests
that use the original document id rather than the internal integer `doc_id` — acceptable
for the project sizes this prototype targets, not meant as a hot-path lookup.
"""
function find_by_original_id(manager::ProjectManager, id_str::String)
    for (_, v) in RocksDB.DBIterator(manager.db; cf=manager.cf_meta)
        record = _decode_metadata(v)
        if Schema.get_field(record, "id") == id_str
            return record
        end
    end
    return nothing
end

end # module
