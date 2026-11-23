# Followers of a remote dataset (PLAN.md §5.12, issue #5, phase 5c). The functions have the
# names of the engine's functions. A follower is created with the wire names of the server, as
# `create_project` creates a dataset: this package does not declare the spec types of the engine.

"""
    RemoteFollower

A follower of a remote dataset, as returned by [`create_follower`](@ref) and
[`open_follower`](@ref). It is the remote counterpart of the engine's `Follower`. It holds the
[`RemoteProject`](@ref) of its leader and its name, and nothing else.
"""
struct RemoteFollower
    leader::RemoteProject
    name::String
end

"""
    RemoteFollowerView

One value of a remote partition follower, as returned by `open_follower(h, "name=value")`. It is
the remote counterpart of the engine's `FollowerView`: [`search`](@ref), [`ftsearch`](@ref) and
[`calibrate!`](@ref) on it are restricted to the items of that partition.
"""
struct RemoteFollowerView
    follower::RemoteFollower
    value::String
end

Base.show(io::IO, f::RemoteFollower) = print(io, "RemoteFollower(", repr(f.leader.conn.url), ", ", repr(_canonical(f)), ")")
Base.show(io::IO, v::RemoteFollowerView) =
    print(io, "RemoteFollowerView(", repr(v.follower.leader.conn.url), ", ", repr(_canonical(v)), ")")

_canonical(f::RemoteFollower) = string(f.leader.id, "/", f.name)
_canonical(v::RemoteFollowerView) = string(_canonical(v.follower), "=", v.value)

# The path of a route of a follower: `name`, or `name=value` for a view, escaped as one segment.
follower_path(f::RemoteFollower, rest::AbstractString="") =
    dataset_path(f.leader.id, "/followers/" * HTTP.escapeuri(f.name) * rest)
follower_path(v::RemoteFollowerView, rest::AbstractString="") =
    dataset_path(v.follower.leader.id, "/followers/" * HTTP.escapeuri(string(v.follower.name, "=", v.value)) * rest)

"""
    FollowerStatus

What [`follower_status`](@ref) and [`followers`](@ref) report about one follower. The fields are
those of the engine's `FollowerStatus`, and one more:

- `name`: the canonical name, `leader/follower`.
- `leader`: the name of the leader.
- `kind`, `source`, `policy`: the type of the spec, the key source and the policy, as the
  engine prints them: `PartitionFollower`, `MetaValue("repo")`, `OnIndex()`.
- `plugin`: the plugin that defines the spec or the key source, or `nothing`.
- `position`: the last `_id` of the leader that the follower processed.
- `leader_indexed`: the number of indexed items of the leader.
- `lag`: `leader_indexed - position`.
- `synchronized`: whether `lag` is zero and the follower is not stale.
- `stale`: whether the leader was rebuilt after the follower processed its items.
- `missing`: whether the leader lists the follower and its directory does not exist.
- `open`: whether the server has the follower open.
- `stats`: the statistics of its kind as a named tuple, or `nothing` when it is not open.
- `error`: the last error of the server when it opened or processed the follower, or `nothing`.
  The engine has no such field.
"""
struct FollowerStatus
    name::String
    leader::String
    kind::Union{Nothing,String}
    source::Union{Nothing,String}
    policy::Union{Nothing,String}
    plugin::Union{Nothing,String}
    position::Int
    leader_indexed::Int
    lag::Int
    synchronized::Bool
    stale::Bool
    missing::Bool
    open::Bool
    stats::Any
    error::Union{Nothing,String}
end

_string_or_nothing(x) = x === nothing ? nothing : string(x)

function _follower_status(r)
    stats = get(r, :stats, nothing)
    FollowerStatus(string(r[:name]), string(r[:leader]), _string_or_nothing(get(r, :kind, nothing)),
                   _string_or_nothing(get(r, :source, nothing)), _string_or_nothing(get(r, :policy, nothing)),
                   _string_or_nothing(get(r, :plugin, nothing)), Int(r[:position]), Int(r[:leader_indexed]),
                   Int(r[:lag]), Bool(r[:synchronized]), Bool(r[:stale]), Bool(r[:missing]), Bool(r[:open]),
                   stats === nothing ? nothing : NamedTuple(Symbol(k) => _plain(v) for (k, v) in stats),
                   _string_or_nothing(get(r, :error, nothing)))
end

"""
    create_follower(h::RemoteProject, name; kind, source, disjoint=nothing, graph_min=nothing,
                    distance=nothing, policy=nothing) -> RemoteFollower

Creates the follower `name` of the dataset (`POST /api/v1/datasets/{id}/followers`). The server
processes every indexed item of the dataset before it answers. The token needs the `admin`
permission.

The keywords are the wire names of the server, not the spec types of the engine:

- `kind`: `"partition"` or `"keyword"`.
- `source`: `"keywords"`, `"refs"`, `"doc_id"`, or `"meta:<field>"`.
- `disjoint`: a partition follower only; `false` when an item can have several keys. The server
  takes `true` when it is `nothing`.
- `graph_min`: a partition follower of a dense dataset only.
- `distance`: a keyword follower only: `"jaccard"`, `"dice"`, `"intersection"` or `"cosine_set"`.
- `policy`: `"on_index"`, `"on_write"`, `"manual"`, or `Dict("periodic" => Dict("seconds" => 60,
  "items" => 1000))`. The server takes `"on_index"` when it is `nothing`.
"""
function create_follower(h::RemoteProject, name::AbstractString; kind, source, disjoint=nothing,
                         graph_min=nothing, distance=nothing, policy=nothing)
    body = Dict{String,Any}("name" => String(name), "kind" => string(kind), "source" => string(source))
    for (key, value) in (("disjoint", disjoint), ("graph_min", graph_min), ("distance", distance))
        value === nothing || (body[key] = value isa Symbol ? string(value) : value)
    end
    policy === nothing || (body["policy"] = policy isa Symbol ? string(policy) : policy)
    request(h.conn, "POST", dataset_path(h.id, "/followers"); body)
    return RemoteFollower(h, String(name))
end

"""
    open_follower(h::RemoteProject, name) -> Union{RemoteFollower, RemoteFollowerView}

The follower `name` of the dataset. `name` is a follower name, or a canonical name with or
without the leader part: `"by_repo"`, `"repos/by_repo"`. With a value, `"by_repo=unicach"`, it
returns a [`RemoteFollowerView`](@ref) restricted to that value.

It asks the server for the status of the follower, so a name that the dataset does not list
raises [`UnknownFollower`](@ref). The value is not checked here: the first search with a value
that is not a partition raises [`UnknownPartition`](@ref).
"""
function open_follower(h::RemoteProject, name::AbstractString)
    name = String(name)
    i = findfirst('/', name)
    if i !== nothing
        leader = name[1:prevind(name, i)]
        leader == h.id || throw(ArgumentError("the canonical name $(repr(name)) names another leader; this leader is $(repr(h.id))"))
        name = name[nextind(name, i):end]
    end
    j = findfirst('=', name)
    base, value = j === nothing ? (name, nothing) : (name[1:prevind(name, j)], name[nextind(name, j):end])
    f = RemoteFollower(h, base)
    request(h.conn, "GET", follower_path(f))
    value === nothing && return f
    isempty(value) && throw(ArgumentError("the value after '=' in $(repr(name)) is empty"))
    return RemoteFollowerView(f, value)
end

"""
    close_follower!(f::RemoteFollower) -> nothing

Does nothing, because a `RemoteFollower` holds no resource. It exists so that a script written for
the engine runs unchanged. The server keeps the follower open.
"""
close_follower!(::RemoteFollower) = nothing

"""
    followers(h::RemoteProject) -> Vector{FollowerStatus}

The status of every follower of the dataset (`GET /api/v1/datasets/{id}/followers`).
"""
followers(h::RemoteProject) =
    FollowerStatus[_follower_status(r) for r in request(h.conn, "GET", dataset_path(h.id, "/followers"))[:followers]]

"""
    follower_status(f::RemoteFollower) -> FollowerStatus
    follower_status(v::RemoteFollowerView) -> FollowerStatus

The status of the follower (`GET /api/v1/datasets/{id}/followers/{name}`).
"""
follower_status(f::RemoteFollower) = _follower_status(request(f.leader.conn, "GET", follower_path(f)))
follower_status(v::RemoteFollowerView) = follower_status(v.follower)

"""
    sync!(f::RemoteFollower) -> Int

Processes the indexed items of the leader that the follower has not processed yet, and returns
their number (`POST /api/v1/datasets/{id}/followers/{name}/sync`).
"""
sync!(f::RemoteFollower) = Int(request(f.leader.conn, "POST", follower_path(f, "/sync"))[:processed])

"""
    rebuild_follower!(h::RemoteProject, name) -> RemoteFollower

Builds the follower again from every indexed item of the dataset, with the spec and the policy it
was created with (`POST /api/v1/datasets/{id}/followers/{name}/rebuild`). This is what a
[`StaleFollower`](@ref) needs after its leader was rebuilt. The token needs the `admin`
permission.
"""
function rebuild_follower!(h::RemoteProject, name::AbstractString)
    f = RemoteFollower(h, String(name))
    request(h.conn, "POST", follower_path(f, "/rebuild"))
    return f
end

"""
    delete_follower!(h::RemoteProject, name) -> nothing

Deletes the follower `name` and removes it from the list of the dataset
(`DELETE /api/v1/datasets/{id}/followers/{name}`). The token needs the `admin` permission.
"""
function delete_follower!(h::RemoteProject, name::AbstractString)
    request(h.conn, "DELETE", follower_path(RemoteFollower(h, String(name))))
    return nothing
end

"""
    migrate_partitions!(h::RemoteProject, name; disjoint::Bool) -> RemoteFollower

Moves the partitions of a dataset created with `partition_by` to a partition follower named
`name` (`POST /api/v1/datasets/{id}/migrate-partitions`). `disjoint` has no default: the
migration does not choose between disjoint and non-disjoint partitions. The token needs the
`admin` permission.
"""
function migrate_partitions!(h::RemoteProject, name::AbstractString; disjoint::Bool)
    request(h.conn, "POST", dataset_path(h.id, "/migrate-partitions"); body=Dict("name" => String(name), "disjoint" => disjoint))
    return RemoteFollower(h, String(name))
end

"""
    partitions(f::RemoteFollower) -> Vector{@NamedTuple{name::String, items::Int, graph::Bool}}

The partitions of a partition follower, sorted by name: the number of items of each one, and
whether it has its own graph.
"""
function partitions(f::RemoteFollower)
    resp = request(f.leader.conn, "GET", follower_path(f, "/partitions"))
    return [(name=string(p[:name]), items=Int(p[:items]), graph=Bool(get(p, :graph, false))) for p in resp[:partitions]]
end

"""
    search(v::RemoteFollowerView, vector, k=10; filter=nothing, minrecall=nothing) -> Vector{SearchResult}

The `k` nearest neighbors of `vector` among the items of one partition of a dense dataset
(`POST /api/v1/datasets/{id}/followers/{name=value}/search`). `filter` works as in the
[`search`](@ref) of a dataset. `minrecall` selects a calibrated level of the graph of the
partition, as in the engine.
"""
function search(v::RemoteFollowerView, vector::AbstractVector{<:Real}, k::Integer=10; filter=nothing, minrecall=nothing)
    _check_filter(filter)
    body = Dict{String,Any}("vector" => convert(Vector{Float32}, vector), "k" => Int(k))
    filter === nothing || (body["filter"] = filter)
    minrecall === nothing || (body["minrecall"] = minrecall)
    return _results(request(v.follower.leader.conn, "POST", follower_path(v, "/search"); body))
end

"""
    search(f::RemoteFollower, keys::AbstractVector{<:AbstractString}, k=10; t=1, filter=nothing)
        -> Vector{SearchResult}

Ranks the items of the dataset by the distance of a keyword follower between their keys and
`keys` (`POST /api/v1/datasets/{id}/followers/{name}/search`). `t` is the number of keys of the
query that an item must hold to be a candidate, as in the engine.
"""
function search(f::RemoteFollower, keys::AbstractVector{<:AbstractString}, k::Integer=10; t::Integer=1, filter=nothing)
    _check_filter(filter)
    body = Dict{String,Any}("keys" => String.(keys), "k" => Int(k), "t" => Int(t))
    filter === nothing || (body["filter"] = filter)
    return _results(request(f.leader.conn, "POST", follower_path(f, "/search"); body))
end

"""
    ftsearch(v::RemoteFollowerView, text, k=10; filter=nothing) -> Vector{SearchResult}

The `k` items of one partition of a text dataset with the best score for `text`
(`POST /api/v1/datasets/{id}/followers/{name=value}/ftsearch`).
"""
function ftsearch(v::RemoteFollowerView, text::AbstractString, k::Integer=10; filter=nothing)
    _check_filter(filter)
    body = Dict{String,Any}("text" => String(text), "k" => Int(k))
    filter === nothing || (body["filter"] = filter)
    return _results(request(v.follower.leader.conn, "POST", follower_path(v, "/ftsearch"); body))
end

"""
    calibrate!(v::RemoteFollowerView; levels=nothing, numqueries=nothing, ksearch=nothing)
        -> Vector{CalibrationLevel}

Tunes the beam search of the graph of one partition of a dense dataset for one recall level
(`POST /api/v1/datasets/{id}/followers/{name=value}/calibrate`). As in the [`calibrate!`](@ref)
of a dataset, `levels` is one number, and the server takes `0.9` when it is `nothing`.
"""
function calibrate!(v::RemoteFollowerView; levels::Union{Nothing,Real}=nothing, numqueries=nothing, ksearch=nothing)
    body = Dict{String,Any}()
    for (key, value) in (("minrecall", levels), ("numqueries", numqueries), ("ksearch", ksearch))
        value === nothing || (body[key] = value)
    end
    return [_calibration_level(request(v.follower.leader.conn, "POST", follower_path(v, "/calibrate"); body))]
end

"""
    posting_list(f::RemoteFollower, key; limit=nothing) -> Vector{Int32}
    posting_list(h::RemoteProject, key; limit=nothing) -> Vector{Int32}

The `_id`s of the items that hold `key`, in ascending order, as in the engine. `f` is a keyword
follower, and `key` one of its keys. `h` is a text dataset, with a token as the key, or a
sparse dataset, with a dimension as the key. With `limit=n`, it returns the first `n` ids.

A whole list of a frequent key in a large dataset can hold most of its items, and it travels in
one response. Pass `limit` when only some of them are needed.
"""
function posting_list(f::RemoteFollower, key::AbstractString; limit::Union{Nothing,Integer}=nothing)
    _posting_ids(request(f.leader.conn, "GET", follower_path(f, "/postings"); query=_postings_query(key, limit)))
end

function posting_list(h::RemoteProject, key::Union{AbstractString,Integer}; limit::Union{Nothing,Integer}=nothing)
    _posting_ids(request(h.conn, "GET", dataset_path(h.id, "/postings"); query=_postings_query(key, limit)))
end

function _postings_query(key, limit)
    query = Dict{String,String}("key" => string(key))
    limit === nothing || (query["limit"] = string(limit))
    return query
end

_posting_ids(resp) = Int32[Int32(i) for i in resp[:ids]]
