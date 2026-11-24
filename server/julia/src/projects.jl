"""
    RemoteProject

A dataset of a remote server, as returned by [`create_project`](@ref) and
[`open_project`](@ref). It is the remote counterpart of the engine's `EmbeddedEngine`, and the
functions of this package take it where the engine's functions take an `EmbeddedEngine`.

It holds the [`Connection`](@ref) and the id of the dataset, and nothing else. Every call sends
a request, so two `RemoteProject`s of the same dataset see the same data.
"""
struct RemoteProject
    conn::Connection
    id::String
end

Base.show(io::IO, h::RemoteProject) = print(io, "RemoteProject(", repr(h.conn.url), ", ", repr(h.id), ")")

"""
    create_project(conn::Connection, dataset; index_type=nothing, distance=nothing, dimension=nothing,
                   meta_schema=nothing, partition_by=nothing, partition_graph_min=nothing,
                   edit_correction=nothing, storage=nothing, max_k=nothing) -> RemoteProject

Creates the dataset `dataset` on the server (`POST /api/v1/datasets`) and returns its handle.

The server decides every keyword that is `nothing`: `index_type` is then `"searchgraph"` and
`distance` is `"L2"`. The index types are `"searchgraph"`, `"exhaustive_search"`,
`"parallel_exhaustive_search"`, `"bm25_invfile"`, `"invfile"` and `"sparse_invfile"`, which
also needs `dimension`. `meta_schema` declares fields of `meta` as in the engine, with
`"year" => :int64` pairs or `("year", :int64)` tuples.

`partition_by` names the field of `meta` that partitions the dataset, for a text or a dense
dataset. `partition_graph_min` is the number of members from which a partition of a dense
dataset has its own graph. Both are described in the server manual.

`storage` says how a dense dataset stores its vectors, as the `storage` object of the server: a
`Dict` or a `NamedTuple` with a `kind`, `"float32"`, `"quantized"` or `"sketched"`, and the
fields of that kind. For example, `(kind="quantized", bits=4)` or
`(kind="sketched", hyperplanes=256, bits=8)`. The fields are described in the server manual.
[`project_info`](@ref) reports the storage with every field, and its encoder.

`max_k` is the largest `k` that a search of the dataset can ask for, 100 when it is `nothing`.
A search answers its `k` results in one response, and a larger `k` raises
[`InvalidOption`](@ref). The server manual describes how to change it after the dataset is
created. [`project_info`](@ref) reports it.
"""
function create_project(conn::Connection, dataset::AbstractString; index_type=nothing, distance=nothing,
                        dimension=nothing, meta_schema=nothing, partition_by=nothing,
                        partition_graph_min=nothing, edit_correction=nothing, storage=nothing, max_k=nothing)
    body = Dict{String,Any}("id" => String(dataset))
    for (key, value) in (("index_type", index_type), ("distance", distance), ("dimension", dimension),
                         ("partition_by", partition_by), ("partition_graph_min", partition_graph_min),
                         ("edit_correction", edit_correction), ("storage", storage), ("max_k", max_k))
        value === nothing || (body[key] = value)
    end
    meta_schema === nothing || (body["meta_schema"] = wire_meta_schema(meta_schema))
    request(conn, "POST", "/api/v1/datasets"; body)
    return RemoteProject(conn, String(dataset))
end

"""
    wire_meta_schema(fields) -> Vector{Dict{String,String}}

A declaration of `meta` fields, given as `name => type` pairs or `(name, type)` tuples, in the
shape of the `meta_schema` field of `POST /api/v1/datasets`.
"""
function wire_meta_schema(fields)
    map(collect(fields)) do f
        f isa Pair || (f isa Tuple && length(f) == 2) ||
            throw(ArgumentError("a declared field is `name => type` or `(name, type)`; got $(repr(f))"))
        Dict("name" => string(first(f)), "type" => string(last(f)))
    end
end

"""
    open_project(conn::Connection, dataset) -> RemoteProject

The handle of the dataset `dataset`. Raises [`ResourceNotFound`](@ref) when the server does not
have it. The server must have the dataset loaded to serve it. [`project_info`](@ref) reports
whether it is loaded.
"""
function open_project(conn::Connection, dataset::AbstractString)
    request(conn, "GET", dataset_path(dataset))
    return RemoteProject(conn, String(dataset))
end

"""
    close_project!(h::RemoteProject) -> nothing

Does nothing, because a `RemoteProject` holds no resource. It exists so that a script written for
the engine runs unchanged. The server keeps the dataset open.
"""
close_project!(::RemoteProject) = nothing

"""
    delete_project!(h::RemoteProject) -> nothing

Deletes the dataset from the server, with its files (`DELETE /api/v1/datasets/{id}`). It cannot
be undone. The token needs the `admin` permission.
"""
function delete_project!(h::RemoteProject)
    request(h.conn, "DELETE", dataset_path(h.id))
    return nothing
end

"""
    list_projects(conn::Connection) -> Vector{String}

The ids of every dataset of the server, sorted, loaded or not.
"""
list_projects(conn::Connection) =
    String[string(d[:id]) for d in request(conn, "GET", "/api/v1/datasets")[:datasets]]

"""
    project_info(h::RemoteProject) -> Dict{String,Any}

The descriptor of the dataset (`GET /api/v1/datasets/{id}`): `index_kind`, `distance`,
`created_at`, `loaded`, and for a loaded dataset `doc_count`, `tombstone_count`, `meta_schema`,
`partition_by`, `partition_graph_min`, `beamsearch_baseline`, `storage` and `encoder`.
`storage` has every field of the storage of a dense dataset. `encoder` is `nothing` for a
dataset that stores `Float32` vectors, and otherwise says whether the encoder is `fitted`, its
`dimension`, its range (`min`, `max`), its `bytes_per_vector`, and the number of `staged`
vectors that are not encoded yet.
"""
project_info(h::RemoteProject) = _plain(request(h.conn, "GET", dataset_path(h.id)))

"""
    meta_schema(h::RemoteProject) -> Vector{Pair{String,Symbol}}

The fields of `meta` that the dataset declares, as `name => type` pairs. Empty for a dataset
that declares none, or that the server has not loaded.
"""
function meta_schema(h::RemoteProject)
    declared = get(request(h.conn, "GET", dataset_path(h.id)), :meta_schema, nothing)
    declared === nothing && return Pair{String,Symbol}[]
    return Pair{String,Symbol}[string(f[:name]) => Symbol(f[:type]) for f in declared]
end

"""
    partition_field(h::RemoteProject) -> Union{Nothing,String}

The field of `meta` that partitions the dataset, or `nothing` when it has none.
"""
function partition_field(h::RemoteProject)
    field = get(request(h.conn, "GET", dataset_path(h.id)), :partition_by, nothing)
    return field === nothing ? nothing : string(field)
end

"""
    partition_graph_min(h::RemoteProject) -> Union{Nothing,Int}

The number of members from which a partition of a dense dataset has its own graph, or `nothing`
for a dataset without dense partitions.
"""
function partition_graph_min(h::RemoteProject)
    n = get(request(h.conn, "GET", dataset_path(h.id)), :partition_graph_min, nothing)
    return n === nothing ? nothing : Int(n)
end

"""
    partitions(h::RemoteProject) -> Vector{@NamedTuple{name::String, items::Int, graph::Bool}}

The partitions of the dataset, sorted by name, as in the engine: the number of items of each
one, and whether it has its own graph. A dataset without a partition field raises
[`InvalidOption`](@ref).
"""
function partitions(h::RemoteProject)
    resp = request(h.conn, "GET", dataset_path(h.id, "/partitions"))
    return [(name=string(p[:name]), items=Int(p[:items]), graph=Bool(get(p, :graph, false)))
            for p in resp[:partitions]]
end

"""
    vocabulary_report(h::RemoteProject; scan=false, sample=nothing) -> Dict{String,Any}

The vocabulary report of a text dataset (`GET /api/v1/datasets/{id}/vocab`). `scan=true` adds
the fraction of tokens of the stored documents that the vocabulary does not have, measured over
at most `sample` documents. `sample=nothing` lets the server choose the number.
"""
function vocabulary_report(h::RemoteProject; scan::Bool=false, sample::Union{Nothing,Integer}=nothing)
    query = Dict{String,String}("scan" => string(scan))
    sample === nothing || (query["sample"] = string(sample))
    return _plain(request(h.conn, "GET", dataset_path(h.id, "/vocab"); query))
end

"A parsed JSON value as plain `Dict{String,Any}` and `Vector{Any}` values."
_plain(x::JSON3.Object) = Dict{String,Any}(string(k) => _plain(v) for (k, v) in x)
_plain(x::JSON3.Array) = Any[_plain(v) for v in x]
_plain(x) = x
