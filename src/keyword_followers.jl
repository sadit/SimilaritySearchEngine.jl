# Keyword followers and posting lists (CHANGELOG.md §5.11, issue #5, phase 4).
#
# A keyword follower is an inverted file of SimilaritySearch over the keys of its key source,
# `Keywords()` by default. Its posting lists are keyed by the key itself, a `String`
# (`DictInvertedFile`), so it needs no vocabulary of its own. It indexes one object per item of
# its leader, the set of keys of the item, empty when the item has none. The numbers of the
# inverted file are therefore the `_id`s of the leader, and its posting lists hold them.

const _InvertedFiles = SimilaritySearch.InvertedFiles

"""
    KEYWORD_ITEMS_CF

The column family of a keyword follower that holds the keys of each item of the leader: key the
`_id` (big-endian), value the keys. The follower rebuilds its inverted file from it when it is
opened.
"""
const KEYWORD_ITEMS_CF = "keyword_items"

"""
    KeywordFollower(source=Keywords(); distance=Dist.Sets.Jaccard())

A follower that indexes the keys of each item of its leader in an inverted file of
SimilaritySearch (CHANGELOG.md §5.11). A query is a set of keys, and [`search`](@ref) ranks the items
of the leader by `distance` between their keys and the query.

`distance` is a distance between sets. `Dist.Sets.Jaccard`, `Dist.Sets.Dice`,
`Dist.Sets.Intersection` and `Dist.Sets.CosineSet` are computed from the sizes of the two sets
and of their intersection while the posting lists are merged. Another distance is evaluated on
the stored keys of each candidate. [`posting_list`](@ref) returns the items that hold one key.

The leader can be a project of any kind: keys are metadata of every item.
"""
struct KeywordFollower <: FollowerSpec
    source::KeySource
    distance::SimilaritySearch.PreMetric
end
KeywordFollower(source::KeySource=Keywords(); distance::SimilaritySearch.PreMetric=Dist.Sets.Jaccard()) =
    KeywordFollower(source, distance)

Base.show(io::IO, s::KeywordFollower) = print(io, "KeywordFollower(", repr(s.source), "; distance=", s.distance, ")")

follower_column_families(::KeywordFollower) = [KEYWORD_ITEMS_CF]

"""
    KeywordState

The state of an open [`KeywordFollower`](@ref): the inverted file in memory, its insertion
context, a pool of search contexts, the lock that separates insertions from searches, and the
column family of the keys of each item.
"""
mutable struct KeywordState
    spec::KeywordFollower
    db::RocksDB.DB
    items::RocksDB.ColumnFamily
    index::Any
    ctx::Any
    pool::IndexEngine.ContextPool
    lock::IndexEngine.ReadWriteLock
end

function _keyword_index(spec::KeywordFollower)
    _InvertedFiles.DictInvertedFile(String, spec.distance; db=VectorDatabase(Vector{String}[]))
end

_keyword_context() = _InvertedFiles.InvertedFileContext(; reporters=[], observers=[], scheduler=IndexEngine.BATCH_SCHEDULER)

function _new_keyword_state(spec::KeywordFollower, db::RocksDB.DB)
    KeywordState(spec, db, db.column_families[KEYWORD_ITEMS_CF], _keyword_index(spec), _keyword_context(),
                 IndexEngine.ContextPool(IndexEngine._search_context(_InvertedFiles.InvertedFileContext)),
                 IndexEngine.ReadWriteLock())
end

# The keys of one item as stored: their number, then the length and the bytes of each one.
function _encode_keys(keys::Vector{String})
    io = IOBuffer()
    write(io, UInt32(length(keys)))
    for k in keys
        write(io, UInt32(ncodeunits(k)))
        write(io, k)
    end
    take!(io)
end

function _decode_keys(bytes::AbstractVector{UInt8})
    io = IOBuffer(bytes)
    n = read(io, UInt32)
    [String(read(io, read(io, UInt32))) for _ in 1:n]
end

create_follower_state(spec::KeywordFollower, leader::EmbeddedEngine, store, db) = _new_keyword_state(spec, db)

function open_follower_state(spec::KeywordFollower, leader::EmbeddedEngine, store, db)
    st = _new_keyword_state(spec, db)
    # The keys of the processed items only: a block that an interruption left unsaved is
    # processed again by the next `sync!`.
    position = Persistence.load_field(store, :position, 0)
    objs = Vector{String}[]
    for (k, v) in RocksDB.DBIterator(db; cf=st.items)
        id = Persistence._decode_be_key(k)
        id > position && break
        id == length(objs) + 1 ||
            corrupted_storage("the keys of item $(length(objs) + 1) are missing from a keyword follower")
        push!(objs, _decode_keys(v))
    end
    length(objs) == position ||
        corrupted_storage("a keyword follower at position $position holds the keys of $(length(objs)) items")
    isempty(objs) || SimilaritySearch.append_items!(st.index, st.ctx, VectorDatabase(objs))
    return st
end

function process_items!(st::KeywordState, leader::EmbeddedEngine, ids::UnitRange{Int}, keys::Vector{Vector{String}})
    # An item that the inverted file already holds, from a block repeated in this process, is
    # skipped: the inverted file numbers its objects in order.
    n = length(st.index)
    first(ids) <= n + 1 || corrupted_storage("a keyword follower holds $n items and was given items from $(first(ids))")
    b = RocksDB.WriteBatch()
    for (j, id) in enumerate(ids)
        RocksDB.put!(b, Persistence._be_key(id), _encode_keys(keys[j]); cf=st.items)
    end
    RocksDB.write!(st.db, b)
    fresh = [keys[j] for (j, id) in enumerate(ids) if id > n]
    isempty(fresh) && return nothing
    IndexEngine.write_lock(st.lock) do
        SimilaritySearch.append_items!(st.index, st.ctx, VectorDatabase(fresh))
    end
    return nothing
end

function follower_stats(st::KeywordState)
    IndexEngine.read_lock(st.lock) do
        lengths = sort!([length(v) for v in values(st.index.adj.end_point)])
        (keys=length(lengths), postings=sum(lengths; init=0),
         items_without_keys=count(iszero, st.index.sizes),
         shortest=isempty(lengths) ? 0 : lengths[1], median=isempty(lengths) ? 0 : lengths[cld(length(lengths), 2)],
         longest=isempty(lengths) ? 0 : lengths[end])
    end
end

check_follower_value(st::KeywordState, value::String) = unsupported_operation(:open_follower,
    "a keyword follower has no restriction to one value; search it with a set of keys, or read the items ",
    "of one key with posting_list")

function _keyword_state(f::Follower, operation::Symbol)
    _check_follower_open(f)
    _check_open(f.leader)
    f.state isa KeywordState || unsupported_operation(operation,
        "$operation with keys needs a keyword follower; '", _canonical_name(f), "' is a ", nameof(typeof(f.spec)))
    return f.state
end

"""
    search(f::Follower, keys::AbstractVector{<:AbstractString}, k=10; t=1, filter=nothing, candidates=2k,
           stats=nothing) -> Vector{SearchResult}

Ranks the items of the leader of a keyword follower by the distance of the follower between
their keys and `keys`. The results are items of the leader.

`t` is the number of keys of the query that an item must hold to be a candidate: `t=1` takes
every item with at least one of them, and `t = length(keys)` only the items with all of them. A
key that no item holds is ignored, so it does not count toward `length(keys)`. `filter`,
`candidates` and `stats` work as in the search of the leader.
"""
function search(f::Follower, keys::AbstractVector{<:AbstractString}, k::Int=10; t::Integer=1, filter=nothing,
                candidates::Int=2k, stats::Union{Nothing,SearchStats}=nothing)
    st = _keyword_state(f, :search)
    t >= 1 || invalid_option(:t, "t must be at least 1; got $t")
    query = unique!(String[String(key) for key in keys])
    leader = f.leader
    n = filter === nothing ? k : max(candidates, k)
    raw = IndexEngine.read_lock(st.lock) do
        ctx = IndexEngine.checkout!(st.pool)
        try
            snap = copy(ctx.costdists)
            res = SimilaritySearch.knnqueue(SimilaritySearch.KnnSorted, max(n, 1))
            SimilaritySearch.search(st.index, ctx, query, res; t)
            ids = UInt32[res.ids[i] for i in res.sp:res.ep]
            (id=ids, dist=Float32[res.dists[i] for i in res.sp:res.ep],
             deleted=[id in leader.engine.deleted_ids for id in ids],
             distance_evaluations=SimilaritySearch.distance_evaluations(ctx, snap))
        finally
            IndexEngine.checkin!(st.pool, ctx)
        end
    end
    res = filter === nothing ? raw : _filter_hits(leader.project, raw, k, filter)
    _record_stats!(stats, res)
    return _hydrate_results(leader.project, res)
end

# A follower is searched with keys (a keyword follower) or through a restricted view.
search(f::Follower, query, k::Int=10; kwargs...) = unsupported_operation(:search,
    "a follower is searched with a set of keys when it is a keyword follower, or through a restricted view, ",
    "open_follower(leader, \"", f.name, "=value\"), when it is a partition follower; '", _canonical_name(f),
    "' is a ", nameof(typeof(f.spec)), " and got a ", typeof(query))

"""
    posting_list(h, key; limit=nothing) -> Vector{Int32}

The `_id`s of the items that hold `key`, in ascending order (CHANGELOG.md §5.11). `h` is a project or
a follower with an inverted index:

- a keyword follower: `key` is a key of its source, a `String`;
- a text project: `key` is a token of its vocabulary, as `TextSearch.query_tokens` gives it. The
  list holds the indexed documents that contain it;
- a sparse project: `key` is a dimension, an integer from 1 to the dimension of the project.

With `limit=n`, it returns the first `n` ids. There is no cursor. A key that no item holds
returns an empty list. Deleted items are in the list: [`exists`](@ref) reports which ones are
deleted.

Reading a whole list copies it. In a large index the list of a frequent key holds a large
fraction of the items, so a caller that needs only some of them passes `limit`.

A dense project has no inverted index and raises `UnsupportedOperation`.
"""
function posting_list(f::Follower, key::AbstractString; limit::Union{Nothing,Integer}=nothing)
    st = _keyword_state(f, :posting_list)
    _check_limit(limit)
    IndexEngine.read_lock(st.lock) do
        list = SimilaritySearch.neighbors(st.index.adj, String(key))
        _limited(list, limit)
    end
end

function posting_list(h::EmbeddedEngine, key; limit::Union{Nothing,Integer}=nothing)
    _check_open(h)
    _check_limit(limit)
    return _posting_list(h.engine, key, limit)
end

function _check_limit(limit::Union{Nothing,Integer})
    limit === nothing || limit >= 0 || invalid_option(:limit, "limit must be at least 0; got $limit")
    return nothing
end

_limited(list::Nothing, limit) = Int32[]
_limited(list::AbstractVector, limit) =
    Int32[list[i] for i in 1:(limit === nothing ? length(list) : min(Int(limit), length(list)))]

function _posting_list(engine::IndexEngine.FullTextEngine, key, limit)
    key isa AbstractString || invalid_option(:key, "a text project takes a token as the key; got a $(typeof(key))")
    IndexEngine.read_lock(engine.lock) do
        index = engine.backend.index
        index === nothing && return Int32[]
        voc, adj = index isa BM25InvertedFile ? (index.voc, index.adj) : (index.model.voc, index.invfile.adj)
        token = TextSearch.token2id(voc, String(key))
        token == 0 && return Int32[]
        out = _limited(SimilaritySearch.neighbors(adj, token), limit)
        IndexEngine.maybe_trim_cache!(adj)
        out
    end
end

function _posting_list(engine::IndexEngine.SparseEngine, key, limit)
    key isa Integer || invalid_option(:key, "a sparse project takes a dimension as the key; got a $(typeof(key))")
    dim = engine.backend.dimension
    1 <= key <= dim || invalid_option(:key, "a dimension of this project is from 1 to $dim; got $key")
    IndexEngine.read_lock(engine.lock) do
        _limited(SimilaritySearch.neighbors(engine.backend.index.adj, Int(key)), limit)
    end
end

_posting_list(engine::IndexEngine.AbstractSearchEngine, key, limit) = unsupported_operation(:posting_list,
    "posting_list needs an inverted index; this project holds $(IndexEngine.payload_kind(engine)) vectors in a ",
    "graph or an exact index. A keyword follower indexes the keywords of any project")
