# Sketched dense projects (PLAN.md §5.14): a dense project that stores the sketches of an
# AnchoredDistantHyperplanes model, encoded by a QuantSketch at 1, 2, 4 or 8 bits per hyperplane.

import SimilaritySearch

sketch_items(n) = dense_items([JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:n]])
_sketch_ids(res) = [r._id for r in res]
# Small fitting options: the Frankenstein vectors are few, and the defaults sample 4096 vectors.
small_sketch(; kw...) = Sketched(128; henc=256, samplesize=400, kw...)

@testset "sketched project: the options are checked" begin
    @test_throws InvalidOption Sketched(100)
    @test_throws InvalidOption Sketched(128; bits=3)
    @test_throws InvalidOption Sketched(128; rerank=4)               # rerank needs the Float32 vectors
    @test Sketched(128; rerank=4, keep_vectors=true).rerank == 4
    @test_throws InvalidOption Sketched(128; henc=100)
    @test_throws InvalidOption Sketched(128; hsel=64)
    # At entquantile=0.5, hsel=200 keeps about 100 candidate pairs, fewer than 128 hyperplanes.
    @test_throws InvalidOption Sketched(128; hsel=200)
    @test_throws InvalidOption Sketched(128; entquantile=0.95)
    @test Sketched(128; hsel=1280, entquantile=0.9).hsel == 1280
    @test_throws InvalidOption Sketched(128; anchorpolicy=:central)
    # The anchors are selected with a SemiMetric; RussellRao is only a PreMetric.
    @test_throws InvalidOption DenseVectors(SimilaritySearch.Dist.Bits.RussellRao(); storage=Sketched())
end

@testset "sketched project: round trip and growth at every number of bits" begin
    items = sketch_items(500)
    for bits in (1, 2, 4, 8), index in (Exhaustive(), Graph())
        @testset "$bits bit(s), $(nameof(typeof(index)))" begin
        mktempworkdir() do workdir
            spec = DenseVectors(Dist.L2(); index, storage=small_sketch(; bits))
            h = create_project(workdir, "s", spec)
            @test !sketch_info(h).fitted && quantizer_info(h) === nothing
            append_items!(h, items[1:400])
            index!(h)
            info = sketch_info(h)
            @test info.fitted && info.dimension == 128 && info.bytes_per_vector == 128 * bits ÷ 8
            @test info.bytes_on_disk_per_vector == info.bytes_per_vector
            @test (info.min === nothing) == (bits == 1)
            # keep_vectors=false: the Float32 vectors are gone once they are encoded.
            @test !isfile(joinpath(workdir, "s", "dense_vectors.mmapdb"))
            @test fetch_items(h, ["frankenstein_5"])[1].payload === nothing
            # Items appended after the fitting are encoded with the saved encoder and found.
            append_items!(h, items[401:500])
            index!(h)
            res = search(h, items[450].vector, 10)
            @test length(res) == 10
            index isa Exhaustive && @test any(r -> r._id == 450 && r.distance == 0, res)
            # Distances between sketches tie often at 1 and 2 bits. A graph reopened from disk
            # visits the neighbors of a vertex in another order, so among tied results its order
            # can change; the distances do not. An exact index keeps the order too.
            answer(h) = [index isa Exhaustive ? _sketch_ids(search(h, items[i].vector, 10)) :
                         [r.distance for r in search(h, items[i].vector, 10)] for i in 1:25:500]
            before = answer(h)
            close_project!(h)

            h = open_project(workdir, "s")
            @test project_spec(h).storage == spec.storage
            @test sketch_info(h) == info
            @test answer(h) == before
            close_project!(h)
        end
        end
    end
end

@testset "sketched exact project: results equal the library's search over the same encoder" begin
    items = sketch_items(300)
    for bits in (1, 8)
        mktempworkdir() do workdir
            h = create_project(workdir, "s", DenseVectors(Dist.L2(); index=Exhaustive(), storage=small_sketch(; bits)))
            append_items!(h, items)
            index!(h)
            enc = h.engine.quant.encoder
            codes = [IndexEngine.sketch_vector(enc, it.vector) for it in items]
            ref = ExhaustiveSearch(SimilaritySearch.distance(enc.qs), SimilaritySearch.VectorDatabase(codes))
            for i in 1:30:300
                res = SimilaritySearch.search(ref, SimilaritySearch.GenericContext(), IndexEngine.sketch_vector(enc, items[i].vector),
                                              SimilaritySearch.knnqueue(SimilaritySearch.KnnSorted, 10))
                want = sort!([p.dist for p in res])
                @test sort!([r.distance for r in search(h, items[i].vector, 10)]) ≈ want
            end
            # searchbatch answers with the same distances.
            ids, dists = searchbatch(h, [items[i].vector for i in 1:30:300], 10)
            @test size(ids) == (10, 10) && all(>(0), ids)
            close_project!(h)
        end
    end
end

@testset "sketched project: the second evaluation, and the overlap with the exact answer" begin
    items = sketch_items(600)
    X = [it.vector for it in items]
    exact(q) = sortperm([SimilaritySearch.evaluate(Dist.L2(), q, x) for x in X])[1:10]
    queries = 1:30:600
    overlap(h) = sum(length(intersect(_sketch_ids(search(h, X[i], 10)), exact(X[i]))) for i in queries) / (10 * length(queries))
    report = Dict{String,Float64}()
    for (name, storage) in (("1 bit", small_sketch(; bits=1)), ("8 bits", small_sketch(; bits=8)),
                            ("8 bits, rerank 4", small_sketch(; bits=8, rerank=4, keep_vectors=true)))
        mktempworkdir() do workdir
            h = create_project(workdir, "s", DenseVectors(Dist.L2(); index=Exhaustive(), storage))
            append_items!(h, items)
            index!(h)
            report[name] = overlap(h)
            if storage.rerank > 0
                # The results have the distance of the project, against the kept vectors.
                res = search(h, X[7], 10)
                @test res[1]._id == 7 && res[1].distance == 0
                @test all(r -> r.distance ≈ SimilaritySearch.evaluate(Dist.L2(), X[7], X[r._id]), res)
                @test issorted([r.distance for r in res])
                @test sketch_info(h).bytes_on_disk_per_vector == 128 + 4 * 128
                @test fetch_items(h, ["frankenstein_7"])[1].payload == X[7]
                ids, dists = searchbatch(h, [X[7], X[31]], 10)
                @test ids[1, 1] == 7 && ids[1, 2] == 31 && dists[1, 1] == 0
                # A partition search of a sketched project is evaluated again too: there is none
                # here, and a filter keeps the second evaluation.
                fres = search(h, X[7], 5; filter=(r, m) -> isodd(r._id))
                @test all(r -> isodd(r._id), fres) && issorted([r.distance for r in fres])
            end
            close_project!(h)
        end
    end
    @info "overlap with the exact top 10 over $(length(queries)) queries, 128 hyperplanes, Frankenstein (128 dimensions)" report
    @test report["8 bits, rerank 4"] >= report["8 bits"]
    @test report["8 bits"] > 0.3
end

@testset "sketched project: partitions and partition followers" begin
    items = sketch_items(300)
    for (i, it) in enumerate(items)
        it.meta["half"] = isodd(i) ? "odd" : "even"
    end
    mktempworkdir() do workdir
        storage = small_sketch(; bits=8, rerank=4, keep_vectors=true)
        h = create_project(workdir, "s", DenseVectors(Dist.L2(); index=Exhaustive(), storage); partitions=PartitionBy("half"))
        append_items!(h, items)
        index!(h)
        res = search(h, items[9].vector, 5; partition="odd")
        @test res[1]._id == 9 && all(r -> isodd(r._id), res) && issorted([r.distance for r in res])
        @test res[2].distance ≈ SimilaritySearch.evaluate(Dist.L2(), items[9].vector, items[res[2]._id].vector)
        f = create_follower(h, "half", PartitionFollower(MetaValue("half")))
        vres = search(open_follower(h, "half=even"), items[10].vector, 5)
        @test vres[1]._id == 10 && all(r -> iseven(r._id), vres)
        @test vres[2].distance ≈ SimilaritySearch.evaluate(Dist.L2(), items[10].vector, items[vres[2]._id].vector)
        close_project!(h)
    end
end

@testset "sketched project: the encoder needs more than 64 staged vectors" begin
    items = sketch_items(100)
    mktempworkdir() do workdir
        h = create_project(workdir, "s", DenseVectors(Dist.L2(); index=Exhaustive(), storage=small_sketch()))
        append_items!(h, items[1:50])
        index!(h)
        @test !sketch_info(h).fitted
        @test_throws NotTrained search(h, items[1].vector, 5)
        @test_throws NotTrained fit_quantizer!(h)
        append_items!(h, items[51:100])
        index!(h)
        @test sketch_info(h).fitted
        @test length(search(h, items[1].vector, 5)) == 5
        close_project!(h)
    end
end

@testset "sketched project: centering subtracts the mean of the sample" begin
    items = sketch_items(200)
    mktempworkdir() do workdir
        h = create_project(workdir, "s", DenseVectors(Dist.Cosine(); index=Exhaustive(), storage=small_sketch(; center=true)))
        append_items!(h, items)
        index!(h)
        enc = h.engine.quant.encoder
        @test enc.mean isa Vector{Float32} && length(enc.mean) == 128
        @test any(r -> r._id == 3 && r.distance == 0, search(h, items[3].vector, 5))
        close_project!(h)
    end
end
