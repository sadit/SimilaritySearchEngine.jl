# Events of a project (CHANGELOG.md §5.15): one `ItemsIndexed` per call that indexes items, with the
# range of their `_id`s, to the observers that `observe!` added. The followers are observers.

struct EventRecorder <: ProjectObserver
    events::Vector{Tuple{Int,Int}}
end
EventRecorder() = EventRecorder(Tuple{Int,Int}[])
SimilaritySearchEngine.on_event(r::EventRecorder, ::EmbeddedEngine, e::ItemsIndexed) = push!(r.events, (e.sp, e.ep))

struct FailingObserver <: ProjectObserver end
SimilaritySearchEngine.on_event(::FailingObserver, ::EmbeddedEngine, ::ItemsIndexed) = error("the observer failed")

ev_item(i) = DenseItem(Float32[i, 1, 2]; doc_id="d$i", keywords=["k$(i % 3)"])

@testset "events: one ItemsIndexed per call, with the range of its _ids" begin
    mktempworkdir() do workdir
        # A graph stages its items, and index! reports them.
        g = create_project(workdir, "g", DenseVectors())
        r = observe!(g, EventRecorder())
        @test observers(g) == ProjectObserver[r]
        append_items!(g, [ev_item(i) for i in 1:10])
        @test isempty(r.events)
        index!(g)
        @test r.events == [(1, 10)]
        index!(g)                                       # nothing new: no event
        append_items!(g, [ev_item(i) for i in 11:15])
        index!(g)
        @test r.events == [(1, 10), (11, 15)]
        close_project!(g)
        @test_throws ProjectClosed observe!(g, EventRecorder())

        # A reopened handle reports what it indexes from then on.
        g = open_project(workdir, "g")
        r2 = observe!(g, EventRecorder())
        @test observe!(g, r2) === r2 && length(observers(g)) == 1     # added once
        append_items!(g, [ev_item(i) for i in 16:18])
        index!(g)
        @test r2.events == [(16, 18)]
        unobserve!(g, r2)
        append_items!(g, [ev_item(19)])
        index!(g)
        @test r2.events == [(16, 18)] && isempty(observers(g))
        close_project!(g)

        # An exact project indexes on append, and the append reports. Items indexed before the
        # observer was added are not reported to it.
        x = create_project(workdir, "x", DenseVectors(; index=Exhaustive()))
        append_items!(x, [ev_item(i) for i in 1:4])
        rx = observe!(x, EventRecorder())
        index!(x)
        append_items!(x, [ev_item(i) for i in 5:6])
        append_items!(x, [ev_item(7)])
        @test rx.events == [(5, 6), (7, 7)]
        close_project!(x)

        # A quantized project stages its vectors, and index! encodes and reports them.
        q = create_project(workdir, "q", DenseVectors(; index=Exhaustive(), storage=Quantized()))
        rq = observe!(q, EventRecorder())
        append_items!(q, [ev_item(i) for i in 1:8])
        @test isempty(rq.events)
        index!(q)
        @test rq.events == [(1, 8)]
        close_project!(q)

        # A text project reports at index!, also the items appended before the observer.
        t = create_project(workdir, "t", TextDocuments(FitFromCorpus()))
        append_items!(t, [TextItem("the text number $i"; doc_id="t$i") for i in 1:5])
        rt = observe!(t, EventRecorder())
        index!(t)
        @test rt.events == [(1, 5)]
        close_project!(t)
    end
end

@testset "events: an observer that fails does not fail the call of the project" begin
    mktempworkdir() do workdir
        h = create_project(workdir, "h", DenseVectors(; index=Exhaustive()))
        observe!(h, FailingObserver())
        r = observe!(h, EventRecorder())
        n = @test_logs (:error, r"failed on ItemsIndexed\(1, 3\)") match_mode=:any append_items!(h, [ev_item(i) for i in 1:3])
        # The items are saved, and the next observer receives the event.
        @test n == 3 && r.events == [(1, 3)]
        @test length(search(h, Float32[1, 1, 2], 5)) == 3
        close_project!(h)
    end
end

@testset "events: an open follower is an observer of its leader" begin
    mktempworkdir() do workdir
        h = create_project(workdir, "h", DenseVectors(; index=Exhaustive()))
        append_items!(h, [ev_item(i) for i in 1:3])
        f = create_follower(h, "kw", KeywordFollower())
        @test any(o -> o === f, observers(h))
        close_follower!(f)
        @test isempty(observers(h))
        # A closed follower lags, and opening it processes nothing.
        append_items!(h, [ev_item(i) for i in 4:6])
        f = open_follower(h, "kw")
        @test any(o -> o === f, observers(h)) && f.position == 3
        # The next event processes what it missed.
        append_items!(h, [ev_item(7)])
        @test f.position == 7 && posting_list(f, "k1") == [1, 4, 7]
        close_project!(h)
        @test isempty(h.observers)
    end
end
