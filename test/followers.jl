# Leader and follower datasets, phase 2: the framework (PLAN.md §5.11, issue #5). The engine
# defines no kind of follower in this phase, so the kind is defined here, in a module registered
# as a plugin. That also checks that a kind can be defined outside the engine.

module FollowerTestKinds
using SimilaritySearchEngine
const SSE = SimilaritySearchEngine

# The `_id`s of the leader by key, saved whole after each block. A set per key, so processing
# an item twice gives the same groups.
struct GroupsSpec <: FollowerSpec
    source::KeySource
end

mutable struct GroupsState
    store::Any
    groups::Dict{String,Set{UInt32}}
    fail_at::Union{Nothing,Int}     # raises when it reaches this `_id`, to test an interruption
end

SSE.create_follower_state(::GroupsSpec, leader, store, db) = GroupsState(store, Dict{String,Set{UInt32}}(), nothing)
SSE.open_follower_state(::GroupsSpec, leader, store, db) =
    GroupsState(store, SSE.Persistence.load_field(store, :groups, Dict{String,Set{UInt32}}()), nothing)

function SSE.process_items!(st::GroupsState, leader, ids, keys)
    for (id, ks) in zip(ids, keys)
        st.fail_at == id && error("simulated failure at item $id")
        for k in ks
            push!(get!(Set{UInt32}, st.groups, k), UInt32(id))
        end
    end
    SSE.Persistence.save_field!(st.store, :groups, st.groups)
end

SSE.follower_stats(st::GroupsState) = (groups=length(st.groups),)
SSE.check_follower_value(st::GroupsState, value::String) =
    haskey(st.groups, value) || SSE.Errors.unknown_partition(value, "no group named $value")

# A key source that reads the payload: the sign of the first coordinate of a vector.
struct FirstSign <: KeySource end
SSE.keys_of(::FirstSign, item::StoredItem) = item.payload[1] >= 0 ? "pos" : "neg"
SSE.needs_payload(::FirstSign) = true
end

module UnregisteredKinds
using SimilaritySearchEngine
struct Spec <: FollowerSpec
    source::KeySource
end
end

register_plugin!(FollowerTestKinds)
const FTK = FollowerTestKinds

follower_item(i) = DenseItem(Float32[isodd(i) ? i : -i, 1, 2]; doc_id="d$i",
    keywords=i % 3 == 0 ? ["three", isodd(i) ? "odd" : "even"] : [isodd(i) ? "odd" : "even"],
    meta=Dict{String,Any}("repo" => i % 5 == 0 ? ["a", "b"] : "a"))
groups(f) = Dict(k => sort!(collect(v)) for (k, v) in f.state.groups)
expected_groups(n, key) = begin
    d = Dict{String,Vector{UInt32}}()
    for i in 1:n, k in key(i)
        push!(get!(Vector{UInt32}, d, k), UInt32(i))
    end
    d
end
keywords_of(i) = (i % 3 == 0 ? ["three"] : String[]) ∪ [isodd(i) ? "odd" : "even"]

@testset "followers: creation processes the indexed items, key sources, names" begin
    mktempworkdir() do workdir
        h = create_project(workdir, "lead", DenseVectors(; index=Exhaustive()))
        append_items!(h, [follower_item(i) for i in 1:30])

        f = create_follower(h, "kw", FTK.GroupsSpec(Keywords()))
        @test f isa Follower
        @test isdir(follower_dir(workdir, "lead", "kw"))
        @test follower_status(f).position == 30
        @test follower_status(f).synchronized
        @test groups(f) == expected_groups(30, keywords_of)

        repo = create_follower(h, "repo", FTK.GroupsSpec(MetaValue("repo")))
        @test groups(repo) == expected_groups(30, i -> i % 5 == 0 ? ["a", "b"] : ["a"])
        ids = create_follower(h, "ids", FTK.GroupsSpec(DocId()))
        @test length(ids.state.groups) == 30 && ids.state.groups["d7"] == Set(UInt32[7])
        sign = create_follower(h, "sign", FTK.GroupsSpec(FTK.FirstSign()))
        @test groups(sign) == expected_groups(30, i -> [isodd(i) ? "pos" : "neg"])
        @test follower_status(sign).stats == (groups=2,)

        # Names: an open follower is returned again, and a value gives a view.
        @test open_follower(h, "kw") === f
        @test open_follower(h, "lead/kw") === f
        v = open_follower(h, "lead/kw=three")
        @test v isa FollowerView && v.follower === f && v.value == "three"
        @test follower_status(v).name == "lead/kw"
        @test_throws UnknownPartition open_follower(h, "kw=four")
        @test_throws InvalidOption open_follower(h, "other/kw")
        @test_throws InvalidOption open_follower(h, "kw=")
        e = try open_follower(h, "kv"); nothing catch e; e end
        @test e isa UnknownFollower && occursin("\"kw\"", e.msg)
        @test_throws InvalidOption create_follower(h, "kw", FTK.GroupsSpec(Keywords()))
        @test_throws InvalidOption create_follower(h, "bad name", FTK.GroupsSpec(Keywords()))

        # A key that is not a string is refused with the item, and the creation is undone.
        h2 = create_project(workdir, "lead2", DenseVectors(; index=Exhaustive()))
        append_items!(h2, [DenseItem(Float32[1, 2, 3]; meta=Dict{String,Any}("repo" => 7))])
        @test_throws PayloadMismatch create_follower(h2, "repo", FTK.GroupsSpec(MetaValue("repo")))
        @test isempty(followers(h2)) && !isdir(follower_dir(workdir, "lead2", "repo"))
        close_project!(h2)

        # No direct write.
        for op in (() -> append_items!(f, [follower_item(1)]), () -> delete_item!(f, 1), () -> index!(f),
                   () -> append_items!(v, [follower_item(1)]))
            err = try op(); nothing catch err; err end
            @test err isa UnsupportedOperation && occursin("'lead'", err.msg)
        end
        close_project!(h)
        @test f.closed && sign.closed
        @test_throws ProjectClosed follower_status(f)
    end
end

@testset "followers: policies" begin
    mktempworkdir() do workdir
        # An exact leader indexes on append.
        h = create_project(workdir, "exact", DenseVectors(; index=Exhaustive()))
        append_items!(h, [follower_item(i) for i in 1:10])
        on_write = create_follower(h, "w", FTK.GroupsSpec(Keywords()); policy=OnWrite())
        on_index = create_follower(h, "i", FTK.GroupsSpec(Keywords()))
        manual = create_follower(h, "m", FTK.GroupsSpec(Keywords()); policy=Manual())
        periodic = create_follower(h, "p", FTK.GroupsSpec(Keywords()); policy=Periodic(seconds=60))
        append_items!(h, [follower_item(i) for i in 11:20])
        @test on_write.position == 20
        @test on_index.position == 10 && manual.position == 10 && periodic.position == 10
        index!(h)
        @test on_index.position == 20 && manual.position == 10 && periodic.position == 10
        @test sync!(manual) == 10 && manual.position == 20
        @test sync!(manual) == 0
        @test groups(manual) == expected_groups(20, keywords_of)
        @test_throws InvalidOption Periodic()
        close_project!(h)

        # A graph leader stages its items: a follower sees them after `index!`.
        g = create_project(workdir, "graph", DenseVectors())
        append_items!(g, [follower_item(i) for i in 1:40])
        f = create_follower(g, "kw", FTK.GroupsSpec(Keywords()); policy=OnWrite())
        @test f.position == 0
        index!(g)
        @test f.position == 0                   # OnWrite: nothing new was indexed by a write
        fi = create_follower(g, "kwi", FTK.GroupsSpec(Keywords()))
        @test fi.position == 40                 # created after index!: it processes everything
        append_items!(g, [follower_item(i) for i in 41:50])
        @test fi.position == 40 && follower_status(fi).lag == 0
        index!(g)
        @test fi.position == 50 && groups(fi) == expected_groups(50, keywords_of)
        close_project!(g)
    end
end

@testset "followers: an interrupted sync! repeats one block" begin
    mktempworkdir() do workdir
        h = create_project(workdir, "lead", DenseVectors(; index=Exhaustive()))
        append_items!(h, [follower_item(i) for i in 1:5])
        f = create_follower(h, "kw", FTK.GroupsSpec(Keywords()); policy=Manual())
        ref = create_follower(h, "ref", FTK.GroupsSpec(Keywords()); policy=Manual())
        append_items!(h, [follower_item(i) for i in 6:35])
        f.state.fail_at = 18
        @test_throws ErrorException sync!(f; block=10)
        @test f.position == 15                  # the block 6:15 was saved, 16:25 was not
        close_follower!(f)
        f = open_follower(h, "kw")
        @test f.position == 15
        @test sync!(f; block=10) == 20
        sync!(ref)
        @test groups(f) == groups(ref) == expected_groups(35, keywords_of)

        # Through the leader: the leader keeps its work, and the error is raised afterwards.
        f2 = create_follower(h, "w", FTK.GroupsSpec(Keywords()); policy=OnWrite())
        f2.state.fail_at = 36
        @test_throws ErrorException append_items!(h, [follower_item(36)])
        @test IndexEngine.indexed_length(h.engine) == 36 && f2.position == 35
        f2.state.fail_at = nothing
        @test sync!(f2) == 1
        close_project!(h)
    end
end

@testset "followers: generation, plugins, status, deletion, read-only leader" begin
    mktempworkdir() do workdir
        P = SimilaritySearchEngine.Persistence
        h = create_project(workdir, "lead", DenseVectors(; index=Exhaustive()))
        append_items!(h, [follower_item(i) for i in 1:12])
        a = create_follower(h, "a", FTK.GroupsSpec(Keywords()))
        b = create_follower(h, "b", FTK.GroupsSpec(Refs()); policy=Manual())
        @test_throws InvalidOption create_follower(h, "u", UnregisteredKinds.Spec(Keywords()))
        @test !isdir(follower_dir(workdir, "lead", "u"))

        close_follower!(b)
        append_items!(h, [follower_item(i) for i in 13:15])
        index!(h)
        st = Dict(s.name => s for s in followers(h))
        @test st["lead/a"].open && st["lead/a"].synchronized
        @test !st["lead/b"].open && st["lead/b"].lag == 3 && st["lead/b"].stats === nothing
        @test st["lead/b"].kind == "GroupsSpec" && st["lead/b"].source == "Refs()" && st["lead/b"].policy == "Manual()"
        @test st["lead/b"].plugin == "FollowerTestKinds"

        # A rebuilt leader has a new generation: its followers are stale until rebuilt.
        close_follower!(a)
        P.save_field!(h.store, :generation, "another numbering")
        @test only(s for s in followers(h) if s.name == "lead/a").stale
        @test_throws StaleFollower open_follower(h, "a")
        a = rebuild_follower!(h, "a")
        @test a.position == 15 && groups(a) == expected_groups(15, keywords_of)
        @test !isdir(follower_dir(workdir, "lead", "a") * ".rebuild")
        close_follower!(a)
        a = open_follower(h, "a")
        @test a.position == 15
        close_follower!(a)
        @test_throws StaleFollower open_follower(h, "b")     # every follower of the leader
        close_follower!(rebuild_follower!(h, "b"))

        # A plugin that is not registered in this process.
        delete!(SimilaritySearchEngine._PLUGINS, "FollowerTestKinds")
        try
            @test_throws PluginNotLoaded open_follower(h, "b")
        finally
            register_plugin!(FollowerTestKinds)
        end

        # Deletion: an open follower first has to be closed.
        bb = open_follower(h, "b")
        @test_throws InvalidOption delete_follower!(h, "b")
        close_follower!(bb)
        delete_follower!(h, "b")
        @test !isdir(follower_dir(workdir, "lead", "b"))
        @test [s.name for s in followers(h)] == ["lead/a"]
        @test_throws UnknownFollower open_follower(h, "b")
        @test_throws UnknownFollower delete_follower!(h, "b")

        # A missing directory is reported, and delete_follower! removes the name.
        rm(follower_dir(workdir, "lead", "a"); recursive=true)
        @test only(followers(h)).missing
        @test_throws CorruptedStorage open_follower(h, "a")
        delete_follower!(h, "a")
        @test isempty(followers(h))

        c = create_follower(h, "c", FTK.GroupsSpec(Keywords()))
        close_project!(h)

        # A follower opened from a read-only leader handle is read-only.
        hr = open_project(workdir, "lead"; read_only=true)
        cr = open_follower(hr, "c")
        @test cr.position == 15
        @test_throws UnsupportedOperation sync!(cr)
        @test_throws UnsupportedOperation create_follower(hr, "d", FTK.GroupsSpec(Keywords()))
        close_project!(hr)

        # The leader's generation and list survive a reopen.
        h = open_project(workdir, "lead")
        c = open_follower(h, "c")
        @test groups(c) == expected_groups(15, keywords_of)
        close_project!(h)
    end
end

@testset "followers: attach_follower!, closing by name, open_followers" begin
    mktempworkdir() do workdir
        h = create_project(workdir, "lead", DenseVectors(; index=Exhaustive()))
        append_items!(h, [follower_item(i) for i in 1:10])
        a = create_follower(h, "a", FTK.GroupsSpec(Keywords()))
        b = create_follower(h, "b", FTK.GroupsSpec(Keywords()); policy=Manual())
        @test [f.name for f in open_followers(h)] == ["a", "b"]
        close_follower!(h, "b")
        close_follower!(h, "absent")                   # not open: nothing to do
        @test [f.name for f in open_followers(h)] == ["a"] && b.closed
        @test_throws InvalidOption attach_follower!(h, "a")   # open
        close_project!(h)

        # A copy of the leader with one of its followers, as `load` makes it: same generation.
        cp(joinpath(workdir, "lead"), joinpath(workdir, "copy"))
        mkpath(joinpath(workdir, "copy.followers"))
        cp(follower_dir(workdir, "lead", "a"), follower_dir(workdir, "copy", "a"))
        c = open_project(workdir, "copy")
        st = Dict(s.name => s for s in followers(c))
        @test st["copy/b"].missing                       # the list came with the copy
        delete_follower!(c, "b")
        attach_follower!(c, "a")
        @test open_follower(c, "a").position == 10
        @test follower_status(open_follower(c, "a")).leader == "copy"
        close_project!(c)

        # A rebuilt leader is a new project with a new generation: the attached follower is stale.
        fresh = create_project(workdir, "fresh", DenseVectors(; index=Exhaustive()))
        append_items!(fresh, [follower_item(i) for i in 1:12])
        @test_throws UnknownFollower attach_follower!(fresh, "a")     # no directory yet
        mkpath(joinpath(workdir, "fresh.followers"))
        cp(follower_dir(workdir, "lead", "a"), follower_dir(workdir, "fresh", "a"))
        attach_follower!(fresh, "a")
        @test only(followers(fresh)).stale
        @test_throws StaleFollower open_follower(fresh, "a")
        @test rebuild_follower!(fresh, "a").position == 12
        close_project!(fresh)
    end
end
