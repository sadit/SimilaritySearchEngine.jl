# Partition followers (PLAN.md §5.11, issue #5, phase 3).
#
# A partition follower holds the structures that the partitions of §5.9 and §5.10 hold inside a
# project: the partition table, the members of each partition, and either the posting lists of
# each partition (text leader) or the members and graphs of each partition (dense leader). It
# reads everything else from its leader: the vocabulary, the scorer, the document vectors, the
# vectors and the deleted items. A search in a partition therefore gives each item the score or
# the distance that the global search gives it.

"""
    PartitionFollower(source::KeySource; disjoint=true, graph_min=nothing)

A follower that divides the items of its leader into partitions, one per key of `source`
(PLAN.md §5.11). For example `PartitionFollower(MetaValue("repo"))` makes one partition per value
of `meta["repo"]`. An item without a key belongs to no partition.

- `disjoint=true`: an item has at most one key, and an item with several keys raises
  `PayloadMismatch`. `disjoint=false` puts an item in each partition it names: an author in each
  institution where they have documents.
- `graph_min`: on a dense leader with a `SearchGraph`, the number of members from which a
  partition has its own graph, by default [`DEFAULT_PARTITION_GRAPH_MIN`](@ref). A smaller
  partition, and every partition of an exact leader, is searched with an exact scan of its
  members. It is an error on a text leader.

The leader is a text or a dense project. `open_follower(leader, "name=value")` returns the
partition `value`, which [`search`](@ref), [`ftsearch`](@ref) and [`calibrate!`](@ref) accept.
"""
struct PartitionFollower <: FollowerSpec
    source::KeySource
    disjoint::Bool
    graph_min::Union{Nothing,Int}
end
PartitionFollower(source::KeySource; disjoint::Bool=true, graph_min::Union{Nothing,Integer}=nothing) =
    PartitionFollower(source, disjoint, graph_min === nothing ? nothing : Int(graph_min))

function Base.show(io::IO, s::PartitionFollower)
    print(io, "PartitionFollower(", repr(s.source), "; disjoint=", s.disjoint)
    s.graph_min === nothing || print(io, ", graph_min=", s.graph_min)
    print(io, ")")
end

follower_column_families(::PartitionFollower) =
    [Persistence.PARTITION_TABLE_CF, Persistence.PARTITION_MEMBERS_CF,
     Persistence.PARTITION_POSTINGS_CF, Persistence.PARTITION_GRAPH_CF]

function check_follower_spec(spec::PartitionFollower, leader::EmbeddedEngine)
    kind = IndexEngine.payload_kind(leader.engine)
    kind === :sparse && invalid_option(:spec,
        "a partition follower needs a text or a dense leader; '", leader.dataset, "' holds sparse vectors, ",
        "which cannot be partitioned yet")
    if spec.graph_min !== nothing
        kind === :dense || invalid_option(:spec,
            "`graph_min` applies to a dense leader; '", leader.dataset, "' holds text, whose partitions are posting lists")
        spec.graph_min >= 1 || invalid_option(:spec, "`graph_min` must be at least 1; got $(spec.graph_min)")
    end
    return nothing
end

"""
    PartitionState

The state of an open [`PartitionFollower`](@ref): its engine store, its `PartitionStore` (the
table, the members, the lists of a text leader, the graphs of a dense leader), and, on a dense
leader, the partitions in memory with the number of members each graph holds.
"""
mutable struct PartitionState
    spec::PartitionFollower
    store::Persistence.EngineStore
    parts::Persistence.PartitionStore
    dense::Union{Nothing,IndexEngine.DensePartitions}
    graph_len::Dict{UInt32,Int}
end

# Saves each range of a partition graph in the follower: its direct neighbors, then the number of
# members the graph holds. The partitions inside a project use the same scheme (`embedded.jl`).
_partition_graph_observer(parts::Persistence.PartitionStore, store::Persistence.EngineStore,
                          graph_len::Dict{UInt32,Int}) =
    number -> (index, sp, ep) -> begin
        Persistence.save_partition_neighbors!(parts, number, sp, [IndexEngine.direct_neighbors(index, i) for i in sp:ep])
        graph_len[number] = ep
        Persistence.save_field!(store, :partition_graph_len, graph_len)
    end

_partition_label(spec::PartitionFollower) = repr(spec.source)

function create_follower_state(spec::PartitionFollower, leader::EmbeddedEngine, store, db)
    parts = Persistence.open_partition_store(db, _partition_label(spec), leader.postings_cache...)
    graph_len = Dict{UInt32,Int}()
    IndexEngine.payload_kind(leader.engine) === :dense || return PartitionState(spec, store, parts, nothing, graph_len)
    graph_min = something(spec.graph_min, DEFAULT_PARTITION_GRAPH_MIN)
    Persistence.save_field!(store, :partition_graph_min, graph_min)
    dense = IndexEngine.new_partitions(leader.engine, graph_min, _partition_graph_observer(parts, store, graph_len))
    return PartitionState(spec, store, parts, dense, graph_len)
end

function open_follower_state(spec::PartitionFollower, leader::EmbeddedEngine, store, db)
    parts = Persistence.open_partition_store(db, _partition_label(spec), leader.postings_cache...)
    graph_len = Persistence.load_field(store, :partition_graph_len, Dict{UInt32,Int}())
    IndexEngine.payload_kind(leader.engine) === :dense || return PartitionState(spec, store, parts, nothing, graph_len)
    engine = leader.engine
    graph_min = Persistence.load_field(store, :partition_graph_min, DEFAULT_PARTITION_GRAPH_MIN)
    dense = IndexEngine.new_partitions(engine, graph_min, _partition_graph_observer(parts, store, graph_len))
    search_state = Persistence.load_field(store, :partition_search_state, Dict{UInt32,NamedTuple}())
    opt = Persistence.load_field(store, :partition_opt_beamsearch, Dict{UInt32,IndexEngine.OptBeamSearch}())
    # The members of the processed items only. A block that an interruption left unsaved is
    # processed again by the next `sync!`; a graph may hold some of its members, which are dropped
    # here. A graph keeps only direct links to earlier members, so its first members stand alone.
    position = Persistence.load_field(store, :position, 0)
    membership = Persistence.load_partition_membership(parts, position)
    for number in sort!(collect(keys(membership)))
        ids = membership[number]
        st = get(search_state, number, nothing)
        IndexEngine.restore_partition!(engine, dense, number, ids;
            load_neighbors=(i -> Persistence.load_partition_neighbors(parts, number, i)),
            graph_len=min(get(graph_len, number, 0), length(ids)),
            beamsearch=st === nothing ? nothing : st.beamsearch,
            hints=st === nothing ? nothing : st.hints,
            opt_beamsearch=get(opt, number, nothing))
    end
    return PartitionState(spec, store, parts, dense, graph_len)
end

function process_items!(st::PartitionState, leader::EmbeddedEngine, ids::UnitRange{Int}, keys::Vector{Vector{String}})
    if st.spec.disjoint
        for (j, ks) in enumerate(keys)
            length(ks) <= 1 || payload_mismatch("item $(ids[j]) has $(length(ks)) values for a disjoint partition ",
                "follower over ", repr(st.spec.source), ": ", join(repr.(ks), ", "), ". A disjoint partition takes ",
                "one value per item; create the follower with disjoint=false to put an item in several partitions")
        end
    end
    members = Persistence.assign_new_partitions!(st.parts, first(ids), keys)
    if st.dense === nothing
        # The lists of the block, by partition and token, each one ascending because the ids are.
        tokens = IndexEngine.document_tokens(leader.engine, ids)
        lists = Dict{UInt64,Vector{UInt32}}()
        for (j, nums) in enumerate(members), num in nums, t in tokens[j]
            push!(get!(Vector{UInt32}, lists, Persistence._partition_cache_key(num, t)), UInt32(ids[j]))
        end
        Persistence.append_partition_postings!(st.parts, lists)
    else
        changed = IndexEngine.add_new_partition_members!(leader.engine, st.dense, first(ids), members)
        isempty(changed) ||
            Persistence.save_field!(st.store, :partition_search_state, IndexEngine.partition_search_state(leader.engine, st.dense))
    end
    return nothing
end

function follower_stats(st::PartitionState)
    counts = Persistence.partition_counts(st.parts)
    sizes = sort(counts)
    graphs = st.dense === nothing ? 0 : count(p -> p.graph !== nothing, values(st.dense.parts))
    (partitions=length(sizes), memberships=sum(sizes; init=0),
     smallest=isempty(sizes) ? 0 : sizes[1], median=isempty(sizes) ? 0 : sizes[cld(length(sizes), 2)],
     largest=isempty(sizes) ? 0 : sizes[end], graphs=graphs)
end

function _partition_number(st::PartitionState, value::AbstractString)
    number = Persistence.partition_number(st.parts, value)
    if number === nothing
        names = Persistence.partition_names(st.parts)
        nearest = _closest_names(names, String(value), 5)
        unknown_partition(value, "this follower has no partition $(repr(String(value))) ($(length(names)) partitions)",
            isempty(nearest) ? "" : "; the closest names are $(join(repr.(nearest), ", "))")
    end
    return number
end

check_follower_value(st::PartitionState, value::String) = (_partition_number(st, value); nothing)

# The state and the partition number of a view, after the checks every operation makes.
function _partition_view(v::FollowerView, operation::Symbol)
    f = v.follower
    _check_follower_open(f)
    _check_open(f.leader)
    st = f.state
    st isa PartitionState || unsupported_operation(operation,
        "$operation on a restricted follower needs a partition follower; '", _canonical_name(f), "' is a ",
        nameof(typeof(f.spec)))
    return st, _partition_number(st, v.value)
end

"""
    search(v::FollowerView, vector, k=10; filter=nothing, minrecall=nothing, candidates=2k, stats=nothing)
        -> Vector{SearchResult}

Searches the partition `v.value` of a partition follower of a dense leader (PLAN.md §5.11). The
results are items of the leader, with the distances of the global search. A partition with fewer
than `graph_min` members, and every partition of an exact leader, is scanned. A larger one uses
its own graph, and `minrecall` uses its calibrated levels (see [`calibrate!`](@ref)). `filter`,
`candidates` and `stats` work as in the search of the leader, over the items of the partition.
"""
function search(v::FollowerView, vector, k::Int=10; filter=nothing, minrecall=nothing, candidates::Int=2k,
                stats::Union{Nothing,SearchStats}=nothing)
    st, number = _partition_view(v, :search)
    leader = v.follower.leader
    engine = leader.engine
    engine isa IndexEngine.DenseEngine || invalid_option(:query,
        "search takes a vector; the leader '", leader.dataset, "' holds text, so search its partitions with ftsearch")
    query = _search_query(engine, vector)
    n = filter === nothing ? k : max(candidates, k)
    raw = IndexEngine.search_partition_live(engine, st.dense, number, query, n, minrecall)
    res = filter === nothing ? raw : _filter_hits(leader.project, raw, k, filter)
    _record_stats!(stats, res)
    return _hydrate_results(leader.project, res)
end

"""
    ftsearch(v::FollowerView, query, k=10; policy=QueryPolicy(), filter=nothing, candidates=2k, stats=nothing,
             minrecall=nothing) -> Vector{SearchResult}

Searches the partition `v.value` of a partition follower of a text leader (PLAN.md §5.11). Only
the documents of the partition are evaluated, and each gets the score that the global search
gives it, because the search uses the vocabulary, the scorer and the document vectors of the
leader. The arguments work as in the `ftsearch` of the leader.
"""
function ftsearch(v::FollowerView, query::Union{AbstractString,AbstractVector{<:AbstractString}}, k::Int=10;
                  minrecall=nothing, policy::QueryPolicy=QueryPolicy(), stats::Union{Nothing,SearchStats}=nothing,
                  filter=nothing, candidates::Int=2k)
    st, number = _partition_view(v, :ftsearch)
    leader = v.follower.leader
    engine = leader.engine
    engine isa IndexEngine.FullTextEngine || invalid_option(:query,
        "ftsearch takes text; the leader '", leader.dataset, "' holds ", IndexEngine.payload_kind(engine),
        " items, so search its partitions with search")
    n = filter === nothing ? k : max(candidates, k)
    parts = st.parts
    raw = IndexEngine.search_partition_live(engine, vocsize -> Persistence.PartitionPostings(parts, number, vocsize),
                                            query, n; policy)
    res = filter === nothing ? raw : _filter_hits(leader.project, raw, k, filter)
    _count_oov!(leader, query)
    _record_stats!(stats, res)
    return _hydrate_results(leader.project, res)
end

"""
    calibrate!(v::FollowerView; levels, numqueries=64, ksearch=10, queries=nothing, tradeoff=1.5,
               transition=(-1, 1), nboot=1000, confidence=0.95, rng=Random.default_rng())
        -> Vector{CalibrationLevel}

Calibrates the graph of the partition `v.value` of a partition follower of a dense `SearchGraph`
leader, as `calibrate!` of the leader does for its global graph, and saves the levels in the
follower. A search with `minrecall` in the partition uses them. Raises `InvalidOption` for a
partition without a graph: an exact scan has nothing to calibrate.
"""
function calibrate!(v::FollowerView; levels=IndexEngine.DEFAULT_MINRECALL_LEVELS, numqueries::Int=64,
                    ksearch::Int=10, queries=nothing, tradeoff::Real=1.5, transition=(-1, 1),
                    nboot::Int=1000, confidence::Real=0.95, rng::AbstractRNG=Random.default_rng())
    st, number = _partition_view(v, :calibrate!)
    f = v.follower
    f.read_only && unsupported_operation(:calibrate!, "the follower '", _canonical_name(f),
        "' was opened from a read-only handle of its leader; open the leader for writing")
    engine = f.leader.engine
    engine isa IndexEngine.DenseEngine{IndexEngine.GraphBackend} || invalid_option(:partition,
        "calibrate! with a partition needs a dense leader with a SearchGraph; the partitions of '",
        f.leader.dataset, "' are scanned, which has nothing to calibrate")
    if queries !== nothing && IndexEngine.isquantized(engine)
        queries = [_search_query(engine, q) for q in queries]
    end
    report = IndexEngine.calibrate!(engine; levels, numqueries, ksearch, queries, tradeoff, transition, nboot,
                                    confidence, rng, partition=number, partitions=st.dense)
    Persistence.save_fields!(st.store, (partition_search_state=IndexEngine.partition_search_state(engine, st.dense),
                                        partition_opt_beamsearch=IndexEngine.partition_opt_beamsearch(engine, st.dense)))
    return report
end

"""
    partitions(f::Follower) -> Vector{@NamedTuple{name::String, items::Int, graph::Bool}}

The partitions of a partition follower, sorted by name, each with the number of items it holds
(deleted items of the leader included) and whether it has its own graph.
"""
function partitions(f::Follower)
    _check_follower_open(f)
    st = f.state
    st isa PartitionState || unsupported_operation(:partitions,
        "'", _canonical_name(f), "' is a ", nameof(typeof(f.spec)), ", not a partition follower")
    names = Persistence.partition_names(st.parts)
    counts = Persistence.partition_counts(st.parts)
    graphs = st.dense === nothing ? Dict{UInt32,Bool}() : IndexEngine.partition_graphs(f.leader.engine, st.dense)
    sort!([(name=n, items=c, graph=get(graphs, UInt32(i), false)) for (i, (n, c)) in enumerate(zip(names, counts))];
          by=p -> p.name)
end

"""
    migrate_partitions!(leader::EmbeddedEngine, name; disjoint::Bool) -> Follower

Moves the partitions inside a project, declared with `PartitionBy` or `declare_partition!`, to a
[`PartitionFollower`](@ref) named `name` (PLAN.md §5.11). It is an explicit step: no other
function migrates a project.

The follower is `PartitionFollower(MetaValue(field); disjoint, graph_min)`, with the field and,
on a dense leader, the `graph_min` of the project. The caller chooses `disjoint`. With
`disjoint=true`, an item whose field holds a list raises `PayloadMismatch`, the follower is not
created, and the project keeps its partitions.

The follower is built from the items of the leader. Then the partitions inside the project are
removed: their column families are emptied, and the fields `:partition_by`,
`:partition_graph_min`, `:partition_graph_len`, `:partition_search_state` and
`:partition_opt_beamsearch` are deleted. The calibrated levels of the partition graphs are not
moved; calibrate the partitions of the follower again. A call interrupted after the follower was
created finishes the removal when it is called again with the same name.
"""
function migrate_partitions!(leader::EmbeddedEngine, name::AbstractString; disjoint::Bool)
    _check_writable_leader(leader, :migrate_partitions!)
    field = partition_field(leader)
    field === nothing && invalid_option(:partitions,
        "the project '", leader.dataset, "' has no partitions inside it, so there is nothing to migrate")
    graph_min = partition_graph_min(leader)
    spec = PartitionFollower(MetaValue(field); disjoint, graph_min)
    name = String(name)
    f = if name in _listed_followers(leader)
        existing = open_follower(leader, name)
        existing.spec isa PartitionFollower && existing.spec.source == MetaValue(field) || invalid_option(:name,
            "the leader '", leader.dataset, "' already has a follower named '", name, "', and it is not a partition ",
            "follower over ", repr(MetaValue(field)), "; choose another name")
        existing
    else
        create_follower(leader, name, spec)
    end
    _drop_project_partitions!(leader)
    return f
end

# Removes the partitions inside a project, after `migrate_partitions!` built their follower.
function _drop_project_partitions!(leader::EmbeddedEngine)
    parts = leader.partitions[]
    parts === nothing && return nothing
    engine = leader.engine
    IndexEngine.write_lock(engine.lock) do
        Persistence.clear_partitions!(parts)
        for field in (:partition_by, :partition_graph_min, :partition_graph_len, :partition_search_state,
                      :partition_opt_beamsearch)
            Persistence.delete_field!(leader.store, field)
        end
        leader.partitions[] = nothing
        engine isa IndexEngine.DenseEngine && (engine.partitions = nothing)
    end
    leader.wrote[] = true
    return nothing
end
