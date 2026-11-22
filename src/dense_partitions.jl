# Partitions of a dense project (PLAN.md §5.10). Part of `IndexEngine`: the structures live in
# memory, and `embedded.jl` reads and writes them through `Persistence`.
#
# A partition is a `SubDatabase` of the vectors of the global index: its `map` is the sorted
# list of the `_id`s of its members. A partition with fewer members than `graph_min` is searched
# with an exact scan of that `SubDatabase`. A larger one has its own `SearchGraph` over it.

export DensePartition, DensePartitions, enable_partitions!, new_partitions, restore_partition!, add_partition_members!,
       add_new_partition_members!, index_partition_graphs!,
       search_partition_live, partition_graphs, partition_search_state, partition_opt_beamsearch

"""
    enable_partitions!(engine::DenseEngine, graph_min::Integer, on_change) -> DensePartitions

Gives `engine` an empty set of partitions (PLAN.md §5.10). `graph_min` is the number of members
from which a partition of a `SearchGraph` project has its own graph; an exact project never
builds one. `on_change` is `number -> (index, sp, ep) -> nothing`, the observer that saves a
range of the graph of partition `number`, or `nothing`.
"""
function enable_partitions!(engine::DenseEngine, graph_min::Integer, on_change::Union{Nothing,Function})
    parts = new_partitions(engine, graph_min, on_change)
    write_lock(() -> (engine.partitions = parts), engine.lock)
    return parts
end

"""
    new_partitions(engine::DenseEngine, graph_min::Integer, on_change) -> DensePartitions

An empty set of partitions over the vectors of `engine`, not attached to it. A partition
follower (PLAN.md §5.11) holds its own set; [`enable_partitions!`](@ref) attaches one to the
engine for the partitions inside a project. The arguments are those of `enable_partitions!`.
"""
function new_partitions(engine::DenseEngine, graph_min::Integer, on_change::Union{Nothing,Function})
    graph_min >= 1 || invalid_option(:partition_graph_min, "partition_graph_min must be at least 1; got $graph_min")
    is_graph = engine isa DenseEngine{GraphBackend}
    DensePartitions(is_graph ? Int(graph_min) : typemax(Int), Dict{UInt32,DensePartition}(), on_change,
                    is_graph ? engine.backend.minrecall : nothing,
                    is_graph ? Int(engine.backend.ctx.maxbatches) : 1,
                    ContextPool(_search_context(GenericContext)))
end

# The database the partitions are views of: every staged vector of a graph project, every
# indexed vector of an exact one.
_parent_database(engine::DenseEngine) = database(engine.backend.index)
_distance(engine::DenseEngine) = SimilaritySearch.distance(engine.backend.index)

function _partition!(parts::DensePartitions, number::UInt32)
    get!(parts.parts, number) do
        DensePartition(UInt32[], nothing, nothing, OptBeamSearch())
    end
end

"""
    _add_members!(parts::DensePartitions, first_id::Integer, members::AbstractVector) -> Nothing

Appends the ids `first_id`, `first_id + 1`, ... to their partitions: `members[i]` holds the
partition numbers of id `first_id + i - 1`. Ids grow, so each partition's list stays sorted.
"""
function _add_members!(parts::DensePartitions, first_id::Integer, members::AbstractVector)
    for (i, nums) in enumerate(members)
        id = UInt32(first_id + i - 1)
        for num in nums
            push!(_partition!(parts, UInt32(num)).ids, id)
        end
    end
    return nothing
end

"""
    add_partition_members!(engine::DenseEngine, first_id::Integer, members::AbstractVector) -> Nothing

[`_add_members!`](@ref) under the write lock, for an exact project, whose items are indexed when
they are appended. A `SearchGraph` project adds members in
[`index!`](@ref index!(::DenseEngine{GraphBackend}, ::Function)) instead.
"""
function add_partition_members!(engine::DenseEngine, first_id::Integer, members::AbstractVector)
    parts = engine.partitions
    parts === nothing && return nothing
    write_lock(() -> _add_members!(parts, first_id, members), engine.lock)
end

"""
    add_new_partition_members!(engine::DenseEngine, parts::DensePartitions, first_id::Integer, members::AbstractVector)
        -> Vector{UInt32}

Appends the ids `first_id`, `first_id + 1`, ... to their partitions in `parts`, as
[`_add_members!`](@ref) does, skipping an id that its partition already ends with or exceeds.
Adding the same ids twice therefore gives the same partitions, which a follower needs when it
repeats a block (PLAN.md §5.11). Then indexes the partition graphs of a `SearchGraph` engine. All
of it runs under the write lock of `engine`, whose vectors the partitions are views of. Returns
the numbers of the partitions whose graph changed.
"""
function add_new_partition_members!(engine::DenseEngine, parts::DensePartitions, first_id::Integer,
                                    members::AbstractVector)
    write_lock(engine.lock) do
        for (i, nums) in enumerate(members)
            id = UInt32(first_id + i - 1)
            for num in nums
                ids = _partition!(parts, UInt32(num)).ids
                (isempty(ids) || id > ids[end]) && push!(ids, id)
            end
        end
        engine isa DenseEngine{GraphBackend} ? _index_partition_graphs!(engine, parts) : UInt32[]
    end
end

function _partition_context(parts::DensePartitions, number::UInt32)
    observers = parts.on_change === nothing ? [] : [CallbackLog(parts.on_change(number))]
    # No reporter: a project with many partitions would print one status line per range of
    # each of them. The global graph keeps its own.
    _searchgraph_context(parts.minrecall, (; reporters=[], observers), parts.maxbatches)
end

"""
    _index_partition_graphs!(engine::DenseEngine{GraphBackend}) -> Vector{UInt32}

Indexes the members each partition graph does not hold yet, and builds the graph of each
partition that reached `graph_min`. Called under the write lock. Returns the numbers of the
partitions whose graph changed.
"""
_index_partition_graphs!(engine::DenseEngine{GraphBackend}) = _index_partition_graphs!(engine, engine.partitions)

function _index_partition_graphs!(engine::DenseEngine{GraphBackend}, parts::DensePartitions)
    changed = UInt32[]
    for number in sort!(collect(keys(parts.parts)))
        p = parts.parts[number]
        if p.graph === nothing
            length(p.ids) >= parts.graph_min || continue
            p.graph = SearchGraph(_distance(engine), SubDatabase(_parent_database(engine), p.ids))
            p.ctx = _partition_context(parts, number)
        end
        length(p.graph) < length(p.ids) || continue
        SimilaritySearch.index!(p.graph, p.ctx)
        push!(changed, number)
    end
    return changed
end

"""
    index_partition_graphs!(engine::DenseEngine{GraphBackend}) -> Vector{UInt32}

[`_index_partition_graphs!`](@ref) under the write lock, for `declare_partition!`, which adds the
members of an indexed project without indexing anything else.
"""
index_partition_graphs!(engine::DenseEngine{GraphBackend}) = write_lock(() -> _index_partition_graphs!(engine), engine.lock)

"""
    restore_partition!(engine::DenseEngine, number, ids; load_neighbors, graph_len, beamsearch, hints,
                       opt_beamsearch) -> DensePartition

Rebuilds partition `number` when a project is opened: its members are `ids`, sorted. A graph is
rebuilt when `graph_len > 0`: the direct neighbors of local ids `1:graph_len` come from
`load_neighbors(i)`, and `beamsearch` and `hints` are set on it as on the global graph (PLAN.md
§0, item 13). A graph that holds fewer members than `ids` gets the rest from the next
[`index!`](@ref index!(::DenseEngine{GraphBackend}, ::Function)).
"""
function restore_partition!(engine::DenseEngine, number::Integer, ids::Vector{UInt32}; load_neighbors::Function,
                            graph_len::Integer, beamsearch, hints, opt_beamsearch)
    parts = engine.partitions
    parts === nothing && invalid_option(:partition_by, "restore_partition! needs enable_partitions! first")
    restore_partition!(engine, parts, number, ids; load_neighbors, graph_len, beamsearch, hints, opt_beamsearch)
end

function restore_partition!(engine::DenseEngine, parts::DensePartitions, number::Integer, ids::Vector{UInt32};
                            load_neighbors::Function, graph_len::Integer, beamsearch, hints, opt_beamsearch)
    p = DensePartition(ids, nothing, nothing, opt_beamsearch === nothing ? OptBeamSearch() : opt_beamsearch)
    if graph_len > 0 && engine isa DenseEngine{GraphBackend}
        graph_len <= length(ids) ||
            corrupted_storage("the graph of partition $number holds $graph_len members; the partition has $(length(ids))")
        graph = SearchGraph(_distance(engine), SubDatabase(_parent_database(engine), ids))
        restore_adjacency!(graph, load_neighbors, graph_len)
        beamsearch === nothing || (graph.algo[] = beamsearch)
        hints === nothing || append!(graph.hints, hints)
        p.graph = graph
        p.ctx = _partition_context(parts, UInt32(number))
    end
    parts.parts[UInt32(number)] = p
    return p
end

"""
    _partition_graph_parts(engine::DenseEngine{GraphBackend}, number::UInt32) -> (graph, ctx, table)

The graph of partition `number`, its insertion context and its table of calibrated levels, for
[`calibrate!`](@ref). Raises `InvalidOption` when the partition has no graph: an exact scan has
nothing to calibrate.
"""
function _partition_graph_parts(engine::DenseEngine{GraphBackend}, number::UInt32)
    parts = engine.partitions
    parts === nothing && invalid_option(:partition, "this project has no partition field")
    _partition_graph_parts(engine, parts, number)
end

function _partition_graph_parts(engine::DenseEngine{GraphBackend}, parts::DensePartitions, number::UInt32)
    p = get(parts.parts, number, nothing)
    (p === nothing || p.graph === nothing) &&
        invalid_option(:partition, "this partition has fewer than $(parts.graph_min) indexed members, so it is " *
              "searched with an exact scan, which has a recall of 1 and nothing to calibrate")
    return (p.graph, p.ctx, p.opt_beamsearch)
end

"""
    search_partition_live(engine::DenseEngine, number::UInt32, query, k::Int, minrecall)
        -> (id=..., dist=..., deleted=..., distance_evaluations=...)

[`search_live`](@ref) over partition `number` of a dense project (PLAN.md §5.10). The ids
returned are `_id`s. A partition without a graph is scanned, and `minrecall` has no effect on
it. A partition with a graph is searched with its `BeamSearch`, or with its level for
`minrecall`. A partition that has a graph and no calibrated level raises `NotTrained` when
`minrecall` is given: a search does not calibrate a partition.

A partition that is in the table and has no indexed member returns no result.
"""
function search_partition_live(engine::DenseEngine, number::UInt32, query, k::Int, minrecall::Union{Nothing,Real})
    parts = engine.partitions
    parts === nothing && invalid_option(:partition, "this project has no partition field")
    search_partition_live(engine, parts, number, query, k, minrecall)
end

function search_partition_live(engine::DenseEngine, parts::DensePartitions, number::UInt32, query, k::Int,
                               minrecall::Union{Nothing,Real})
    read_lock(engine.lock) do
        p = get(parts.parts, number, nothing)
        p === nothing && return (id=UInt32[], dist=Float32[], deleted=Bool[], distance_evaluations=0)
        if p.graph !== nothing
            bs = if minrecall === nothing
                nothing
            elseif isempty(p.opt_beamsearch)
                not_trained("minrecall needs a calibrated partition, and this one was never calibrated. Run " *
                            "calibrate!(handle; partition=...) once; a search does not calibrate a partition")
            else
                _nearest_level(p.opt_beamsearch, minrecall)
            end
            return _search_partition_graph(engine, p, bs, query, k)
        end
        return _scan_partition(engine, parts, p, query, k)
    end
end

function _search_partition_graph(engine::DenseEngine, p::DensePartition, bs, query, k::Int)
    ctx = checkout!(engine.search_ctx_pool)
    try
        snap = copy(ctx.costdists)
        res = knnqueue(KnnSorted, max(k, 1))
        if bs === nothing
            search(p.graph, ctx, query, res)
        else
            vstate = SimilaritySearch.getvstate(length(p.graph), ctx)
            search(bs, p.graph, ctx, query, res, p.graph.hints, vstate)
        end
        return _collect_partition(engine, p, res, SimilaritySearch.distance_evaluations(ctx, snap))
    finally
        checkin!(engine.search_ctx_pool, ctx)
    end
end

function _scan_partition(engine::DenseEngine, parts::DensePartitions, p::DensePartition, query, k::Int)
    ctx = checkout!(parts.scan_pool)
    try
        snap = copy(ctx.costdists)
        res = knnqueue(KnnSorted, max(k, 1))
        scan = ExhaustiveSearch(_distance(engine), SubDatabase(_parent_database(engine), p.ids))
        search(scan, ctx, query, res)
        return _collect_partition(engine, p, res, SimilaritySearch.distance_evaluations(ctx, snap))
    finally
        checkin!(parts.scan_pool, ctx)
    end
end

# The result of a partition search, with the local ids replaced by `_id`s.
function _collect_partition(engine::DenseEngine, p::DensePartition, res, evals)
    ids = UInt32[p.ids[i] for i in view(res.ids, res.sp:res.ep)]
    dists = collect(view(res.dists, res.sp:res.ep))
    deleted = [id in engine.deleted_ids for id in ids]
    return (id=ids, dist=dists, deleted=deleted, distance_evaluations=evals)
end

"""
    partition_graphs(engine::DenseEngine) -> Dict{UInt32,Bool}

Whether each partition with an indexed member has a graph.
"""
partition_graphs(engine::DenseEngine) =
    engine.partitions === nothing ? Dict{UInt32,Bool}() : partition_graphs(engine, engine.partitions)

function partition_graphs(engine::DenseEngine, parts::DensePartitions)
    read_lock(() -> Dict(n => p.graph !== nothing for (n, p) in parts.parts), engine.lock)
end

"""
    partition_search_state(engine::DenseEngine) -> Dict{UInt32,NamedTuple}

The `BeamSearch` and the hints of each partition graph, by partition number: what
`embedded.jl` saves so that a reopened partition graph searches as it did (PLAN.md §0, item 13).
"""
partition_search_state(engine::DenseEngine) =
    engine.partitions === nothing ? Dict{UInt32,NamedTuple}() : partition_search_state(engine, engine.partitions)

function partition_search_state(engine::DenseEngine, parts::DensePartitions)
    read_lock(engine.lock) do
        Dict{UInt32,NamedTuple}(n => (beamsearch=p.graph.algo[], hints=collect(UInt32, p.graph.hints))
                                for (n, p) in parts.parts if p.graph !== nothing)
    end
end

"""
    partition_opt_beamsearch(engine::DenseEngine) -> Dict{UInt32,OptBeamSearch}

The calibrated levels of each partition that has any, by partition number.
"""
partition_opt_beamsearch(engine::DenseEngine) =
    engine.partitions === nothing ? Dict{UInt32,OptBeamSearch}() : partition_opt_beamsearch(engine, engine.partitions)

function partition_opt_beamsearch(engine::DenseEngine, parts::DensePartitions)
    read_lock(engine.lock) do
        Dict{UInt32,OptBeamSearch}(n => copy(p.opt_beamsearch) for (n, p) in parts.parts if !isempty(p.opt_beamsearch))
    end
end
