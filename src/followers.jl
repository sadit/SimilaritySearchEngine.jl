# Leader and follower datasets (CHANGELOG.md §5.11, issue #5).
#
# A follower organizes the items of a leader without copying them. It reads the records, the
# `meta` and, when its key source asks for it, the payload of each item of the leader, and it
# stores its own structure, expressed with `_id`s of the leader, in its own RocksDB database.
# This file holds the framework: key sources, the interface of a kind of follower, policies,
# plugins, the handles, and the functions that create, open, process, rebuild and delete a
# follower. The kinds themselves (partitions, keywords) are separate.

"The pattern of a follower name: the pattern of a dataset id of the server."
const FOLLOWER_NAME_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"

"""
    FOLLOWER_SYNC_BLOCK

The number of items of the leader that [`sync!`](@ref) reads, processes and saves at a time.
`sync!` saves the position of the follower after each block, so an interruption repeats at
most one block.
"""
const FOLLOWER_SYNC_BLOCK = 16_384

# --- Key sources ---------------------------------------------------------------------

"""
    KeySource

The part of an item of the leader that a follower uses (CHANGELOG.md §5.11). A key source is a type
with a method of [`keys_of`](@ref). The engine includes [`Keywords`](@ref), [`Refs`](@ref),
[`DocId`](@ref) and [`MetaValue`](@ref). A key source defined outside the engine belongs to a
module registered with [`register_plugin!`](@ref).
"""
abstract type KeySource end

"The keywords of an item (`record.keywords`)."
struct Keywords <: KeySource end

"The refs of an item (`record.refs`)."
struct Refs <: KeySource end

"The `doc_id` of an item. An item without one has no key."
struct DocId <: KeySource end

"""
    MetaValue(name)

The value of the field `name` of the `meta` of an item: a string, a list of strings, or absent.
"""
struct MetaValue <: KeySource
    name::String
end

Base.show(io::IO, ::Keywords) = print(io, "Keywords()")
Base.show(io::IO, ::Refs) = print(io, "Refs()")
Base.show(io::IO, ::DocId) = print(io, "DocId()")
Base.show(io::IO, s::MetaValue) = print(io, "MetaValue(", repr(s.name), ")")

"""
    keys_of(source::KeySource, item::StoredItem)

The keys of `item` for `source`: `nothing`, a `String`, or a vector of strings. A key is a
non-empty string. `item` is the item of the leader as [`fetch_items`](@ref) returns it. Its
`payload` is `nothing` unless [`needs_payload`](@ref) of `source` is `true`.
"""
keys_of(::Keywords, item::Schema.StoredItem) = item.keywords
keys_of(::Refs, item::Schema.StoredItem) = item.refs
keys_of(::DocId, item::Schema.StoredItem) = item.doc_id
keys_of(s::MetaValue, item::Schema.StoredItem) = get(item.meta, s.name, nothing)

"""
    needs_payload(source::KeySource) -> Bool

Whether [`keys_of`](@ref) reads the payload of the item. It is `false` for the key sources of
the engine. A copy of each vector or text costs time and memory, so `sync!` reads the payload
only for a key source that returns `true`.
"""
needs_payload(::KeySource) = false

# The keys of one item as a vector of distinct non-empty strings, whatever `keys_of` returned.
function _key_list(source::KeySource, keys, id::Int)
    keys === nothing && return String[]
    keys isa AbstractString && return [_one_key(source, keys, id)]
    keys isa AbstractVector || payload_mismatch(
        "the key source $(repr(source)) returned a $(typeof(keys)) for item $id; a key source ",
        "returns nothing, a string, or a vector of strings")
    return unique!(String[_one_key(source, k, id) for k in keys])
end

function _one_key(source::KeySource, key, id::Int)
    key isa AbstractString && !isempty(key) && return String(key)
    payload_mismatch("the key source $(repr(source)) returned $(repr(key))::$(typeof(key)) as a key of ",
                     "item $id; a key is a non-empty string")
end

# --- Kinds of follower ---------------------------------------------------------------

"""
    FollowerSpec

What a follower holds and how it is built from the items of its leader (CHANGELOG.md §5.11). A kind
of follower is a subtype with a field `source::KeySource`, or a method of `key_source`, and
methods of these functions, all qualified with `SimilaritySearchEngine.`:

| function | what it does |
|---|---|
| `follower_column_families(spec)` | the names of the column families the kind uses; none by default |
| `check_follower_spec(spec, leader)` | raises when the leader cannot have this follower; accepts every leader by default |
| `create_follower_state(spec, leader, store, db)` | the empty state of a new follower |
| `open_follower_state(spec, leader, store, db)` | the state of an existing follower, read from its storage |
| `process_items!(state, leader, ids, keys)` | adds the items `ids` of the leader, with their keys, to the structure, and saves it |
| `follower_stats(state)` | the statistics of the kind for `describe`; `nothing` by default |
| `check_follower_value(state, value)` | raises when `value` does not name a group of the follower; accepts every value by default |
| `close_follower_state!(state)` | releases what the state holds; nothing by default |

`store` is the `Persistence.EngineStore` of the follower and `db` its `RocksDB.DB`. `ids` is a
range of `_id`s of the leader, and `keys[j]` holds the keys of `ids[j]`. `process_items!` must
give the same result when it runs twice on the same items, because [`sync!`](@ref) repeats the
block that an interruption left unsaved.
"""
abstract type FollowerSpec end

key_source(spec::FollowerSpec) = spec.source
follower_column_families(::FollowerSpec) = String[]
check_follower_spec(::FollowerSpec, leader) = nothing
function create_follower_state end
function open_follower_state end
function process_items! end
follower_stats(state) = nothing
check_follower_value(state, value::String) = nothing
close_follower_state!(state) = nothing

# --- Policies ------------------------------------------------------------------------

"""
    SyncPolicy

When a follower processes the new items of its leader (CHANGELOG.md §5.11): [`OnIndex`](@ref),
[`OnWrite`](@ref), [`Periodic`](@ref) or [`Manual`](@ref). The leader reports its new items to
its open followers with an [`ItemsIndexed`](@ref) event, and the policy says what the follower
does with it (CHANGELOG.md §5.15).
"""
abstract type SyncPolicy end

"""
    OnIndex()

The follower processes the new items when they become searchable: after each `index!` of its
leader, and after each `append_items!` of a leader that indexes on insertion (an exact dense or
a sparse project). The default policy.
"""
struct OnIndex <: SyncPolicy end

"""
    OnWrite()

Another name of [`OnIndex`](@ref), since 2026-11-23 (CHANGELOG.md §5.15). It processed the new items
after each `append_items!` of the leader. A follower processes indexed items only, so a leader
that stages its items had nothing new for it until `index!`, and in the server, whose `append`
indexes, it did the same work as `OnIndex`. A follower created with it keeps it.
"""
struct OnWrite <: SyncPolicy end

"""
    Manual()

The follower processes new items only when [`sync!`](@ref) is called.
"""
struct Manual <: SyncPolicy end

"""
    Periodic(; seconds=nothing, items=nothing)

The follower processes the new items when it is `items` or more items behind its leader, or
every `seconds`. At least one of the two is given. The follower checks `items` at each event of
its leader, and processes in [`on_event`](@ref) (CHANGELOG.md §5.15). The task of `serve` processes it
every `seconds`, and when its lag reaches `items` (CHANGELOG.md §5.12). The engine alone has no task,
so a policy with only `seconds` is treated as [`Manual`](@ref).
"""
struct Periodic <: SyncPolicy
    seconds::Union{Nothing,Float64}
    items::Union{Nothing,Int}
    function Periodic(seconds::Union{Nothing,Real}, items::Union{Nothing,Integer})
        seconds === nothing && items === nothing &&
            invalid_option(:policy, "Periodic needs `seconds`, `items`, or both")
        seconds === nothing || seconds > 0 || invalid_option(:policy, "Periodic needs `seconds` > 0; got $seconds")
        items === nothing || items > 0 || invalid_option(:policy, "Periodic needs `items` > 0; got $items")
        new(seconds === nothing ? nothing : Float64(seconds), items === nothing ? nothing : Int(items))
    end
end
Periodic(; seconds=nothing, items=nothing) = Periodic(seconds, items)

Base.show(io::IO, ::OnIndex) = print(io, "OnIndex()")
Base.show(io::IO, ::OnWrite) = print(io, "OnWrite()")
Base.show(io::IO, ::Manual) = print(io, "Manual()")
Base.show(io::IO, p::Periodic) = print(io, "Periodic(seconds=", something(p.seconds, "nothing"),
                                       ", items=", something(p.items, "nothing"), ")")

# --- Plugins -------------------------------------------------------------------------

const _PLUGINS = Dict{String,Module}()
const _ENGINE_MODULE = @__MODULE__

"""
    register_plugin!(mod::Module) -> Module

Registers `mod` as a plugin: its types can be the spec or the key source of a follower
(CHANGELOG.md §5.11). The name of the plugin is `nameof(mod)`. A follower saves that name, and
`open_follower` raises [`PluginNotLoaded`](@ref) in a process that has not registered it,
because the engine saves the spec with `Serialization`, and a value of a type from a module
that is not loaded cannot be read back.
"""
function register_plugin!(mod::Module)
    _PLUGINS[String(nameof(mod))] = mod
    return mod
end

# The module of `m` or one of its parents that is the engine or a registered plugin.
function _owner_module(m::Module)
    while true
        (m === _ENGINE_MODULE || any(p -> p === m, values(_PLUGINS))) && return m
        p = parentmodule(m)
        p === m && return nothing
        m = p
    end
end

# The name of the plugin that defines the types of `spec`, or `nothing` for the engine's own.
function _plugin_name(spec::FollowerSpec)
    plugin = nothing
    for x in (spec, key_source(spec))
        m = parentmodule(typeof(x))
        owner = _owner_module(m)
        owner === nothing && invalid_option(:spec,
            "$(nameof(typeof(x))) is defined in the module $m, which is not a registered plugin. Call ",
            "register_plugin!($m) before create_follower: the follower saves the type, and only a ",
            "registered module can be required when the follower is opened again")
        owner === _ENGINE_MODULE && continue
        name = String(nameof(owner))
        plugin === nothing || plugin == name || invalid_option(:spec,
            "the spec and the key source come from two plugins, $plugin and $name; a follower names one plugin")
        plugin = name
    end
    return plugin
end

function _check_plugin(plugin::Union{Nothing,String}, canonical::String)
    plugin === nothing || haskey(_PLUGINS, plugin) || plugin_not_loaded(plugin,
        "the follower '$canonical' uses types of the plugin $plugin, which is not registered in this process. ",
        "Load its module and call register_plugin! on it before open_follower")
    return nothing
end

# --- Handles -------------------------------------------------------------------------

"""
    Follower

An open follower (CHANGELOG.md §5.11), from [`create_follower`](@ref) or [`open_follower`](@ref).
It holds its leader handle, and it is registered in it: [`close_project!`](@ref) of the leader
closes it.
"""
mutable struct Follower <: ProjectObserver
    leader::EmbeddedEngine
    name::String
    dir::String
    db::RocksDB.DB
    store::Persistence.EngineStore
    spec::FollowerSpec
    policy::SyncPolicy
    plugin::Union{Nothing,String}
    state::Any
    position::Int
    read_only::Bool
    lock::ReentrantLock
    closed::Bool
    last_error::Union{Nothing,String}   # the error of the last sync! that failed, until one succeeds
end

"""
    FollowerView

A follower restricted to one value, from `open_follower(leader, "name=value")`: the canonical
name is `leader/name=value`. It shares the open [`Follower`](@ref), which [`close_follower!`](@ref)
closes.
"""
struct FollowerView
    follower::Follower
    value::String
end

_canonical_name(leader::String, name::String) = string(leader, "/", name)
_canonical_name(f::Follower) = _canonical_name(f.leader.dataset, f.name)
_canonical_name(v::FollowerView) = string(_canonical_name(v.follower), "=", v.value)

Base.show(io::IO, f::Follower) = print(io, "Follower(", repr(_canonical_name(f)), f.closed ? ", closed" : "", ")")
Base.show(io::IO, v::FollowerView) = print(io, "FollowerView(", repr(_canonical_name(v)), ")")

"""
    follower_dir(workdir, leader, name) -> String

The directory of the follower `name` of `leader`: `<workdir>/<leader>.followers/<name>`. It is
not inside the directory of the leader, which is the RocksDB database of the leader. A dataset
id cannot contain `.`, so this directory is never taken for a dataset.
"""
follower_dir(workdir::AbstractString, leader::AbstractString, name::AbstractString) =
    joinpath(workdir, string(leader, ".followers"), name)

_follower_dir(leader::EmbeddedEngine, name::String) = follower_dir(leader.workdir, leader.dataset, name)
_listed_followers(leader::EmbeddedEngine) = Persistence.load_field(leader.store, :followers, String[])
_leader_generation(leader::EmbeddedEngine) = Persistence.load_field(leader.store, :generation, nothing)

function _check_follower_open(f::Follower)
    f.closed && project_closed(_canonical_name(f), "the follower '", _canonical_name(f),
        "' was closed by close_follower! or by close_project! of its leader; open it again with open_follower")
    return nothing
end

function _check_writable_leader(leader::EmbeddedEngine, operation::Symbol)
    _check_open(leader)
    leader.read_only && unsupported_operation(operation, "$operation writes to the leader '", leader.dataset,
        "', and this handle was opened read-only; open the leader for writing")
    return nothing
end

function _check_follower_name(name::String)
    occursin(FOLLOWER_NAME_PATTERN, name) || invalid_option(:name,
        "a follower name has 1 to 128 letters, digits, '_' or '-'; got $(repr(name))")
    return nothing
end

# Opens the database of a follower with every column family it has, or creates it with `cfs`.
function _open_follower_db(dir::String, cfs::Vector{String}, read_only::Bool, create::Bool)
    if create
        names = vcat(["default", Persistence.ENGINE_CF], cfs)
        return RocksDB.opendb(dir, column_families=names, create_if_missing=true, create_missing_column_families=true)
    end
    names = Project.existing_column_families(dir)
    isempty(names) && corrupted_storage("the directory $dir holds no follower database")
    return read_only ? RocksDB.opendb(dir, column_families=names, read_only=true) :
                       RocksDB.opendb(dir, column_families=names)
end

# The fields of a follower that need no plugin: the name of its leader, its plugin, the
# generation of the leader it was built from, its position, and a description for `describe`.
function _plain_fields(store::Persistence.EngineStore)
    (leader = Persistence.load_field(store, :leader, nothing),
     plugin = Persistence.load_field(store, :follows_plugin, nothing),
     generation = Persistence.load_field(store, :leader_generation, nothing),
     position = Persistence.load_field(store, :position, 0),
     summary = Persistence.load_field(store, :follows_summary, nothing))
end

# Creates the database and the fields of a new follower in `dir`, and its empty state.
function _build_follower(leader::EmbeddedEngine, name::String, dir::String, spec::FollowerSpec,
                         policy::SyncPolicy, plugin::Union{Nothing,String}, generation::String)
    mkpath(dir)
    db = _open_follower_db(dir, follower_column_families(spec), false, true)
    try
        store = Persistence.open_engine_store(db)
        # The plain fields first: they are what `followers(leader)` reads without the plugin.
        Persistence.save_fields!(store, (leader=leader.dataset, follows_plugin=plugin,
            leader_generation=generation, position=0,
            follows_summary=(kind=string(nameof(typeof(spec))), source=repr(key_source(spec)),
                             policy=repr(policy))))
        Persistence.save_field!(store, :follows, (spec=spec, policy=policy))
        state = create_follower_state(spec, leader, store, db)
        return Follower(leader, name, dir, db, store, spec, policy, plugin, state, 0, false, ReentrantLock(), false,
                        nothing)
    catch
        close(db)
        rethrow()
    end
end

"""
    create_follower(leader::EmbeddedEngine, name, spec::FollowerSpec; policy=OnIndex()) -> Follower

Creates the follower `name` of `leader` (CHANGELOG.md §5.11), processes every indexed item of the
leader, and returns it open. `leader` must be writable: its list of followers is saved in it.

The follower is stored in [`follower_dir`](@ref)`(workdir, leader, name)`. Its canonical name
is `leader/name`. `policy` says when it processes new items (see [`SyncPolicy`](@ref)). A spec or
a key source whose type is defined outside the engine must belong to a module registered with
[`register_plugin!`](@ref).

Raises `InvalidOption` when the leader already lists `name`, when the directory exists without
being listed (an earlier `create_follower` stopped before it finished; [`delete_follower!`](@ref)
removes it), or when a type of `spec` belongs to a module that is not a registered plugin. When
processing the items of the leader raises, the follower is deleted and the error is raised.
"""
function create_follower(leader::EmbeddedEngine, name::AbstractString, spec::FollowerSpec;
                         policy::SyncPolicy=OnIndex())
    _check_writable_leader(leader, :create_follower)
    name = String(name)
    _check_follower_name(name)
    listed = _listed_followers(leader)
    name in listed && invalid_option(:name, "the leader '", leader.dataset, "' already has a follower named '",
        name, "'; open it with open_follower, or delete it with delete_follower!")
    dir = _follower_dir(leader, name)
    ispath(dir) && invalid_option(:name, "the directory $dir exists, and the leader '", leader.dataset,
        "' does not list a follower '", name, "': an earlier create_follower stopped before it finished. ",
        "Remove it with delete_follower!(leader, \"", name, "\")")
    plugin = _plugin_name(spec)
    check_follower_spec(spec, leader)
    generation = _leader_generation(leader)
    if generation === nothing
        # A leader created before generations existed (CHANGELOG.md §5.11).
        generation = Project.generate_id()
        Persistence.save_field!(leader.store, :generation, generation)
    end
    f = _build_follower(leader, name, dir, spec, policy, plugin, generation)
    # Listed last: a crash before this line leaves a directory that the leader does not list,
    # which the check above reports.
    Persistence.save_field!(leader.store, :followers, vcat(listed, name))
    leader.wrote[] = true
    _register_follower!(leader, name, f)
    try
        sync!(f)
    catch
        # A follower that cannot process the items of its leader is not left behind.
        close_follower!(f)
        delete_follower!(leader, name)
        rethrow()
    end
    return f
end

# Splits "name", "name=value", "leader/name" or "leader/name=value".
function _parse_follower_name(leader::EmbeddedEngine, name::String)
    i = findfirst('/', name)
    if i !== nothing
        name[1:prevind(name, i)] == leader.dataset || invalid_option(:name,
            "the canonical name $(repr(name)) names another leader; this leader is '", leader.dataset, "'")
        name = name[nextind(name, i):end]
    end
    j = findfirst('=', name)
    j === nothing && (_check_follower_name(name); return name, nothing)
    base, value = name[1:prevind(name, j)], name[nextind(name, j):end]
    _check_follower_name(base)
    isempty(value) && invalid_option(:name, "the value after '=' in $(repr(name)) is empty")
    return base, value
end

function _open_existing_follower(leader::EmbeddedEngine, name::String)
    listed = _listed_followers(leader)
    canonical = _canonical_name(leader.dataset, name)
    if !(name in listed)
        nearest = _closest_names(listed, name, 3)
        unknown_follower(name, "the leader '", leader.dataset, "' has no follower named '", name, "'",
            isempty(nearest) ? "; it has no followers" : "; the closest names are " * join(repr.(nearest), ", "))
    end
    dir = _follower_dir(leader, name)
    isdir(dir) || corrupted_storage("the leader '", leader.dataset, "' lists the follower '", name,
        "', and its directory $dir does not exist. Remove the name with delete_follower!(leader, \"", name, "\")")
    db = _open_follower_db(dir, String[], leader.read_only, false)
    try
        store = Persistence.open_engine_store(db)
        fields = _plain_fields(store)
        _check_plugin(fields.plugin, canonical)
        fields.generation == _leader_generation(leader) || stale_follower(canonical,
            "the follower '", canonical, "' was built from an earlier numbering of the items of '", leader.dataset,
            "': the leader was rebuilt after the follower processed its items. Build it again with ",
            "rebuild_follower!(leader, \"", name, "\")")
        follows = Persistence.load_field(store, :follows, nothing)
        follows === nothing && corrupted_storage("the follower '", canonical, "' has no saved spec")
        state = open_follower_state(follows.spec, leader, store, db)
        return Follower(leader, name, dir, db, store, follows.spec, follows.policy, fields.plugin, state,
                        fields.position, leader.read_only, ReentrantLock(), false, nothing)
    catch
        close(db)
        rethrow()
    end
end

"""
    open_follower(leader::EmbeddedEngine, name) -> Union{Follower, FollowerView}

Opens the follower `name` of `leader` (CHANGELOG.md §5.11). `name` is a follower name, or a
canonical name with or without the leader part: `"by_repo"`, `"repos/by_repo"`. With a value,
`"by_repo=unicach"`, it returns a [`FollowerView`](@ref) restricted to that value.

A follower that is already open on `leader` is returned again, not opened twice. A follower
opened from a read-only handle of its leader is read-only. Opening processes nothing: a follower
that lags behind its leader stays behind until the next event of the leader, as its policy says,
or until [`sync!`](@ref). The open follower is an observer of its leader (CHANGELOG.md §5.15).

Raises [`UnknownFollower`](@ref) for a name that the leader does not list,
[`PluginNotLoaded`](@ref) when the follower needs a plugin that is not registered, and
[`StaleFollower`](@ref) when the leader was rebuilt after the follower processed its items.
"""
function open_follower(leader::EmbeddedEngine, name::AbstractString)
    _check_open(leader)
    base, value = _parse_follower_name(leader, String(name))
    f = get(leader.followers, base, nothing)
    if f === nothing
        f = _open_existing_follower(leader, base)
        _register_follower!(leader, base, f)
    end
    value === nothing && return f
    check_follower_value(f.state, value)
    return FollowerView(f, value)
end

"""
    close_follower!(f::Follower)

Closes the follower and removes it from its leader handle. A second call does nothing. After it,
every function that takes `f` raises [`ProjectClosed`](@ref).
"""
function close_follower!(f::Follower)
    f.closed && return nothing
    lock(f.lock) do
        try
            close_follower_state!(f.state)
        finally
            close(f.db)
            f.closed = true
        end
    end
    get(f.leader.followers, f.name, nothing) === f && delete!(f.leader.followers, f.name)
    unobserve!(f.leader, f)
    return nothing
end

"""
    close_follower!(leader::EmbeddedEngine, name)

Closes the follower `name` if it is open on `leader`; does nothing otherwise. For a caller that
holds the leader and a name, such as the server before it deletes a follower (CHANGELOG.md §5.12).
"""
function close_follower!(leader::EmbeddedEngine, name::AbstractString)
    f = get(leader.followers, String(name), nothing)
    f === nothing || close_follower!(f)
    return nothing
end

"""
    open_followers(leader::EmbeddedEngine) -> Vector{Follower}

The followers open on `leader`, sorted by name. The server walks them to process the followers
with a [`Periodic`](@ref) policy (CHANGELOG.md §5.12).
"""
open_followers(leader::EmbeddedEngine) = sort!(Follower[f for f in values(leader.followers)]; by=f -> f.name)

"""
    sync!(f::Follower; block=FOLLOWER_SYNC_BLOCK) -> Int

Processes the items of the leader from the position of the follower plus one to
`IndexEngine.indexed_length` of the leader, and returns the number of items processed. A staged
item that the leader has not indexed yet is not processed.

It works in blocks of `block` items. After each block it saves the position, so an interruption
repeats at most one block. Deletes are not processed: the leader marks a deleted item in each
result. Raises `UnsupportedOperation` on a follower opened from a read-only leader handle.

A follower that cannot process an item stops before the block of that item, and it does not skip
it: the next `sync!` tries the same item again. Each failure is written to the log as an error,
with the name of the follower, its position and the error, and `follower_status(f).error` holds
its message until a `sync!` succeeds.
"""
function sync!(f::Follower; block::Integer=FOLLOWER_SYNC_BLOCK)
    _check_follower_open(f)
    _check_open(f.leader)
    f.read_only && unsupported_operation(:sync!, "the follower '", _canonical_name(f),
        "' was opened from a read-only handle of its leader; open the leader for writing to process new items")
    block > 0 || invalid_option(:block, "sync! needs `block` > 0; got $block")
    return lock(f.lock) do
        try
            n = _sync_blocks!(f, key_source(f.spec), Int(block))
            f.last_error = nothing
            n
        catch e
            f.last_error = sprint(showerror, e)
            @error "the follower $(_canonical_name(f)) stopped at position $(f.position) of its leader; it stays there until the next item can be processed" exception=(e, catch_backtrace())
            rethrow()
        end
    end
end

# Specialized on the type of the key source: it is resolved once per call, not once per item.
function _sync_blocks!(f::Follower, source::S, block::Int) where {S<:KeySource}
    leader = f.leader
    target = IndexEngine.indexed_length(leader.engine)
    payload = needs_payload(source)
    done = 0
    while f.position < target
        ids = (f.position + 1):min(target, f.position + block)
        keys = Vector{Vector{String}}(undef, length(ids))
        for (j, id) in enumerate(ids)
            keys[j] = _key_list(source, keys_of(source, _leader_item(leader, id, payload)), id)
        end
        process_items!(f.state, leader, ids, keys)
        f.position = last(ids)
        Persistence.save_field!(f.store, :position, f.position)
        done += length(ids)
    end
    return done
end

# The item `id` of the leader, as `fetch_items` returns it, with its payload only when asked.
function _leader_item(leader::EmbeddedEngine, id::Int, payload::Bool)
    record = get_metadata(leader.project, id)
    record === nothing && corrupted_storage("the leader '", leader.dataset, "' has no record for its indexed item $id")
    meta = something(get_meta(leader.project, id; lazy=false), Dict{String,Any}())
    return Schema.StoredItem(record, payload ? _stored_payload(leader, id) : nothing, meta)
end

# An open follower is registered in its leader handle by name, and it observes the events of
# the leader (CHANGELOG.md §5.15).
function _register_follower!(leader::EmbeddedEngine, name::String, f::Follower)
    leader.followers[name] = f
    observe!(leader, f)
    return f
end

"""
    on_event(f::Follower, leader::EmbeddedEngine, e::ItemsIndexed)

What a follower does with new items of its leader, as its policy says (CHANGELOG.md §5.15):
[`OnIndex`](@ref) and [`OnWrite`](@ref) process them with [`sync!`](@ref);
[`Periodic`](@ref) with `items` processes them when its lag reaches `items`; [`Manual`](@ref)
and `Periodic` with only `seconds` do nothing. A follower opened from a read-only handle does
nothing. An error of `sync!` is in the log and in `follower_status(f).error`, and the call of
the leader returns.
"""
function on_event(f::Follower, leader::EmbeddedEngine, ::ItemsIndexed)
    (f.closed || f.read_only) && return nothing
    _processes_now(f.policy, f) || return nothing
    try
        sync!(f)
    catch err
        # `sync!` wrote the error to the log and keeps it for the status of the follower, which
        # stays at the item it could not process. The change of the leader is saved.
        err isa InterruptException && rethrow()
    end
    return nothing
end

_processes_now(::Union{OnIndex,OnWrite}, ::Follower) = true
_processes_now(::Manual, ::Follower) = false
_processes_now(p::Periodic, f::Follower) =
    p.items !== nothing && IndexEngine.indexed_length(f.leader.engine) - f.position >= p.items

"""
    rebuild_follower!(leader::EmbeddedEngine, name) -> Follower

Builds the follower `name` again from every indexed item of `leader`, with the spec and the
policy it was created with, and returns it open. This is what a [`StaleFollower`](@ref) needs
after its leader was rebuilt. A handle of the follower that was open is closed.

The new follower is built in a temporary directory and replaces the old one when it is complete.
"""
function rebuild_follower!(leader::EmbeddedEngine, name::AbstractString)
    _check_writable_leader(leader, :rebuild_follower!)
    name = String(name)
    _check_follower_name(name)
    name in _listed_followers(leader) || unknown_follower(name, "the leader '", leader.dataset,
        "' has no follower named '", name, "'")
    old = get(leader.followers, name, nothing)
    old === nothing || close_follower!(old)
    dir = _follower_dir(leader, name)
    canonical = _canonical_name(leader.dataset, name)
    # The spec and the policy, read from the follower itself, which may be stale.
    db = _open_follower_db(dir, String[], true, false)
    spec, policy, plugin = try
        store = Persistence.open_engine_store(db)
        fields = _plain_fields(store)
        _check_plugin(fields.plugin, canonical)
        follows = Persistence.load_field(store, :follows, nothing)
        follows === nothing && corrupted_storage("the follower '", canonical, "' has no saved spec")
        follows.spec, follows.policy, fields.plugin
    finally
        close(db)
    end
    generation = something(_leader_generation(leader), Project.generate_id())
    Persistence.save_field!(leader.store, :generation, generation)
    tmp = dir * ".rebuild"
    rm(tmp; recursive=true, force=true)
    f = _build_follower(leader, name, tmp, spec, policy, plugin, generation)
    try
        sync!(f)
    finally
        close_follower!(f)
    end
    rm(dir; recursive=true, force=true)
    mv(tmp, dir)
    f = _open_existing_follower(leader, name)
    _register_follower!(leader, name, f)
    return f
end

"""
    attach_follower!(leader::EmbeddedEngine, name)

Lists in `leader` a follower whose directory already exists in
[`follower_dir`](@ref)`(workdir, leader, name)`, and writes the name of the leader in it
(CHANGELOG.md §5.12). It processes nothing. A rebuilt leader has a new generation, so a follower
attached to it stays stale until [`rebuild_follower!`](@ref). A leader loaded from a copy has the
generation of the copy, so a follower copied with it is ready.

The follower must not be open. A name that the leader already lists only gets the name of the
leader written again.
"""
function attach_follower!(leader::EmbeddedEngine, name::AbstractString)
    _check_writable_leader(leader, :attach_follower!)
    name = String(name)
    _check_follower_name(name)
    haskey(leader.followers, name) && invalid_option(:name, "the follower '", _canonical_name(leader.dataset, name),
        "' is open; close it with close_follower! first")
    dir = _follower_dir(leader, name)
    isdir(dir) || unknown_follower(name, "there is no follower directory $dir to attach to '", leader.dataset, "'")
    db = _open_follower_db(dir, String[], false, false)
    try
        Persistence.save_field!(Persistence.open_engine_store(db), :leader, leader.dataset)
    finally
        close(db)
    end
    listed = _listed_followers(leader)
    if !(name in listed)
        Persistence.save_field!(leader.store, :followers, vcat(listed, name))
        leader.wrote[] = true
    end
    return nothing
end

"""
    delete_follower!(leader::EmbeddedEngine, name)

Removes the follower `name`: its directory and its name in the list of the leader. The follower
must not be open. Also removes a directory that the leader does not list, which an interrupted
[`create_follower`](@ref) leaves.
"""
function delete_follower!(leader::EmbeddedEngine, name::AbstractString)
    _check_writable_leader(leader, :delete_follower!)
    name = String(name)
    _check_follower_name(name)
    haskey(leader.followers, name) && invalid_option(:name, "the follower '", _canonical_name(leader.dataset, name),
        "' is open; close it with close_follower! first")
    listed = _listed_followers(leader)
    dir = _follower_dir(leader, name)
    (name in listed || ispath(dir)) || unknown_follower(name, "the leader '", leader.dataset,
        "' has no follower named '", name, "'")
    rm(dir; recursive=true, force=true)
    rm(dir * ".rebuild"; recursive=true, force=true)
    if name in listed
        Persistence.save_field!(leader.store, :followers, filter(!=(name), listed))
        leader.wrote[] = true
    end
    return nothing
end

# --- Status --------------------------------------------------------------------------

"""
    FollowerStatus

What [`follower_status`](@ref) and [`followers`](@ref) report about one follower:

- `name`: the canonical name, `leader/follower`.
- `leader`: the name of the leader.
- `kind`, `source`, `policy`: the type of the spec, the key source and the policy, as text.
- `plugin`: the plugin that defines the spec or the key source, or `nothing`.
- `position`: the last `_id` of the leader that the follower processed.
- `leader_indexed`: the number of indexed items of the leader.
- `lag`: `leader_indexed - position`.
- `synchronized`: whether `lag` is zero and the follower is not stale.
- `stale`: whether the leader was rebuilt after the follower processed its items.
- `missing`: whether the leader lists the follower and its directory does not exist.
- `open`: whether the follower is open on the leader handle.
- `stats`: the statistics of its kind, or `nothing` when the follower is not open.
- `error`: the message of the last `sync!` that failed, until one succeeds, or `nothing`. A
  follower that is not open has `nothing`.
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

"""
    follower_status(f::Follower) -> FollowerStatus

The status of an open follower: its position against the indexed items of its leader, and the
statistics of its kind.
"""
function follower_status(f::Follower)
    _check_follower_open(f)
    indexed = IndexEngine.indexed_length(f.leader.engine)
    FollowerStatus(_canonical_name(f), f.leader.dataset, string(nameof(typeof(f.spec))),
                   repr(key_source(f.spec)), repr(f.policy), f.plugin, f.position, indexed,
                   indexed - f.position, indexed == f.position, false, false, true, follower_stats(f.state),
                   f.last_error)
end
follower_status(v::FollowerView) = follower_status(v.follower)

"""
    followers(leader::EmbeddedEngine) -> Vector{FollowerStatus}

The status of every follower that `leader` lists. A follower that is not open is read from its
saved fields, with a read-only open of its database, and its `stats` is `nothing`. A follower
whose directory is missing is reported with `missing = true`.
"""
function followers(leader::EmbeddedEngine)
    _check_open(leader)
    indexed = IndexEngine.indexed_length(leader.engine)
    generation = _leader_generation(leader)
    out = FollowerStatus[]
    for name in _listed_followers(leader)
        f = get(leader.followers, name, nothing)
        if f !== nothing
            push!(out, follower_status(f))
            continue
        end
        canonical = _canonical_name(leader.dataset, name)
        dir = _follower_dir(leader, name)
        if !isdir(dir)
            push!(out, FollowerStatus(canonical, leader.dataset, nothing, nothing, nothing, nothing,
                                      0, indexed, indexed, false, false, true, false, nothing, nothing))
            continue
        end
        db = _open_follower_db(dir, String[], true, false)
        fields = try
            _plain_fields(Persistence.open_engine_store(db))
        finally
            close(db)
        end
        s = fields.summary
        stale = fields.generation != generation
        push!(out, FollowerStatus(canonical, leader.dataset,
            s === nothing ? nothing : s.kind, s === nothing ? nothing : s.source,
            s === nothing ? nothing : s.policy, fields.plugin, fields.position, indexed,
            indexed - fields.position, !stale && indexed == fields.position, stale, false, false, nothing, nothing))
    end
    return out
end

# --- Writes to a follower ------------------------------------------------------------

function _follower_write(f::Follower, operation::Symbol)
    unsupported_operation(operation, "a follower accepts no direct writes: '", _canonical_name(f),
        "' organizes the items of its leader '", f.leader.dataset, "'. Write to the leader; the follower ",
        "processes its new items with its policy, ", repr(f.policy), ", or with sync!")
end

append_items!(f::Follower, items) = _follower_write(f, :append_items!)
append_items!(v::FollowerView, items) = _follower_write(v.follower, :append_items!)
delete_item!(f::Follower, _id::Integer) = _follower_write(f, :delete_item!)
delete_item!(v::FollowerView, _id::Integer) = _follower_write(v.follower, :delete_item!)
index!(f::Follower) = _follower_write(f, :index!)
index!(v::FollowerView) = _follower_write(v.follower, :index!)
