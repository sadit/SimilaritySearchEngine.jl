@heavy "whole-dataset operations: fft, dnet, neardup, closestpairs, bichromatic_kclosestpairs" begin
    mktempworkdir() do workdir
        items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:60]]
        h = create_project(workdir, "dense_ops")
        append_items!(h, dense_items(items))
        index!(h)

        # fft: Farthest-First Traversal
        r_fft = fft(h, 6; verbose=false)
        @test r_fft isa CenterSelectionResult
        @test r_fft isa FFTResult
        @test length(r_fft.centers) == 6
        @test length(r_fft.assign) == 60
        @test all(a -> 1 <= a <= 6, r_fft.assign)
        @test all(d -> d >= 0f0, r_fft.assigndist)
        @test r_fft.covering >= 0f0
        @test r_fft.separation >= 0f0

        # dnet: density net
        r_dnet = dnet(h, 6; verbose=false)
        @test r_dnet isa CenterSelectionResult
        @test length(r_dnet.centers) >= 5
        @test length(r_dnet.assign) == 60
        @test all(d -> d >= 0f0, r_dnet.assigndist)

        # neardup: near-duplicate extraction by epsilon
        r_nd = neardup(h, 0.3; verbose=false)
        @test r_nd isa NearDupResult
        @test length(r_nd.centers) <= 60
        @test length(r_nd.assign) == 60
        @test r_nd.epsilon == 0.3f0
        @test all(d -> d <= r_nd.epsilon + 1e-5, r_nd.assigndist)

        # closestpairs
        cp = closestpairs(h; k=5)
        @test length(cp) == 5
        @test all(p -> p[1] != p[2], cp)
        @test issorted(cp; by=p -> p[3])

        # bichromatic_kclosestpairs against another database
        other_vectors = [Float32.(it["vector"]) for it in items[1:10]]
        other_db = MatrixDatabase(hcat(other_vectors...))
        bcp = bichromatic_kclosestpairs(h, other_db; k=3)
        @test length(bcp) == 3
        @test issorted(bcp; by=p -> p[3])

        close_project!(h)
    end
end

@heavy "whole-dataset operations run safely concurrently with append_items!/index!" begin
    mktempworkdir() do workdir
        items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:60]]
        more_items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[61:80]]
        h = create_project(workdir, "dense_concurrent")
        append_items!(h, dense_items(items))
        index!(h)

        # Found live 2026-09-09: allknn/fft/dnet/neardup/closestpairs/bichromatic_kclosestpairs
        # used to read engine.backend.ctx directly with no lock at all -- a scan racing a
        # mutating add_item!/index!. Fixing that with a read_lock (matching search_live) was
        # NOT enough: this very test, run against that fix, failed immediately with "`@threads
        # :static` cannot be used concurrently or nested" -- these six all parallelize
        # internally via @BATCHES, and read_lock lets several of them run at once, which
        # collides with itself regardless of locking. See allknn_live's docstring in
        # index_engine.jl for why they take write_lock instead (full exclusivity, including
        # from each other, not just from writes). This drives real concurrent traffic through
        # both sides at once instead of only ever calling these serially like every other test
        # in this file does -- it's what caught the :static problem in the first place, and
        # then a second, unrelated bug (see `_require_no_backlog`'s docstring): a reader can
        # legitimately observe a nonzero append_items!/index! backlog here (the writer task
        # below deliberately interleaves the two), which is an EXPECTED refusal
        # (`ErrorException`, "requires no pending backlog"), not a crash -- only some OTHER
        # exception type indicates the fix actually failed.
        had_error = Threads.Atomic{Bool}(false)
        errlock = ReentrantLock()
        first_error = Ref{Any}(nothing)
        record_error!(e) = (had_error[] = true; lock(() -> (first_error[] === nothing && (first_error[] = e)), errlock))
        # The point of typing the errors: this used to match on the message text.
        _is_expected_backlog_refusal(e) = e isa PendingBacklog

        @sync begin
            for _ in 1:8
                Threads.@spawn begin
                    try
                        for _ in 1:20
                            allknn(h; k=5)
                            fft(h, 5)
                            dnet(h, 5)
                            closestpairs(h; k=3)
                        end
                    catch e
                        _is_expected_backlog_refusal(e) || record_error!(e)
                    end
                end
            end
            Threads.@spawn begin
                try
                    for it in more_items
                        append_items!(h, dense_items([it]))
                        index!(h)
                    end
                catch e
                    record_error!(e)
                end
            end
        end
        had_error[] && @error "concurrent whole-dataset op failed" exception=first_error[]
        @test !had_error[]

        close_project!(h)
    end
end

# The testset above drives concurrent traffic through ONE project, where `write_lock`
# serializes everything. `write_lock` is a field of the engine, so it says nothing about
# two projects, and a process that holds several -- a server -- has nothing serializing
# them. Found live 2026-09-15, when `:static` was still `SimilaritySearch`'s default
# schedule: this fails on the first try under it, because Julia refuses to enter a
# `@threads :static` region while another is running anywhere in the process, whatever
# data each one touches:
#
#     index!(a) => `@threads :static` cannot be used concurrently or nested
#     index!(b) => ok
#
# Both calls below are reachable from a server request handler (`server.jl`'s append and
# calibrate paths call them in-process), so this is the shape of a real 500. See
# `IndexEngine.BATCH_SCHEDULER` for the schedule the engine's contexts carry instead.
@testset "two projects at once: batch operations do not collide across engines" begin
    mktempworkdir() do workdir
        items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:400]]

        function project(name)
            h = create_project(workdir, name)
            append_items!(h, dense_items(items))
            h
        end
        a, b = project("pair_a"), project("pair_b")

        # Each of these parallelizes internally through @BATCHES; only the schedule keeps
        # them from colliding, since neither lock knows about the other engine.
        function both(f)
            ta, tb = Threads.@spawn(f(a)), Threads.@spawn(f(b))
            results = map((ta, tb)) do t
                try (fetch(t); nothing) catch e
                    e isa TaskFailedException ? e.task.exception : e
                end
            end
            failed = filter(!isnothing, collect(results))
            isempty(failed) || @error "concurrent batch op across two projects failed" exception=first(failed)
            @test isempty(failed)
        end

        both(index!)
        both(h -> allknn(h; k=5))
        both(h -> calibrate!(h; numqueries=20))

        close_project!(a)
        close_project!(b)
    end
end
