# Quantized dense projects (issue #1): a dense project that stores the codes of a scalar quantizer
# of SimilaritySearch.ScalarQuant instead of Float32 vectors, on every dense backend.

import SimilaritySearch

const QUANT_INDEXES = (Exhaustive(), Exhaustive(parallel=true), Graph())

quant_rows(n) = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:n]]
_ids(res) = [r._id for r in res]

@testset "quantized project: round trip on every dense backend and quantizer" begin
    rows = quant_rows(400)
    items = dense_items(rows)
    for index in QUANT_INDEXES, storage in (Quantized(SQ(8)), Quantized(SQ(4; range=PerVectorRange())),
                                            Quantized(SQ(4); rotation=QRRotation()))
        mktempworkdir() do workdir
            h = create_project(workdir, "q", DenseVectors(Dist.NormCosine(); index, storage))
            @test !quantizer_info(h).fitted
            append_items!(h, items[1:300])
            # Staged, not encoded: nothing to search yet, and the ids are already assigned.
            @test isempty(search(h, items[1].vector, 5))
            @test fetch_items(h, ["frankenstein_300"])[1]._id == 300
            index!(h)
            append_items!(h, items[301:400])
            index!(h)
            info = quantizer_info(h)
            @test info.fitted && info.storage == storage
            @test info.dimension == 128
            perbyte = 8 ÷ storage.quantizer.bits
            @test info.bytes_per_vector == 128 ÷ perbyte + (storage.quantizer.range isa PerVectorRange ? 8 : 0)
            # keep_vectors=false: the Float32 vectors are gone once encoded.
            @test !isfile(joinpath(workdir, "q", "dense_vectors.mmapdb"))
            @test isfile(joinpath(workdir, "q", "dense_codes.mmapdb"))
            @test fetch_items(h, ["frankenstein_5"])[1].payload === nothing
            before = [_ids(search(h, items[i].vector, 10)) for i in 1:20:400]
            @test all(b -> length(b) == 10, before)
            close_project!(h)

            h = open_project(workdir, "q")
            @test quantizer_info(h) == info
            @test [_ids(search(h, items[i].vector, 10)) for i in 1:20:400] == before
            close_project!(h)
        end
    end
end

@testset "quantized exact project: results equal the library's search over the same encoder" begin
    rows = quant_rows(300)
    items = dense_items(rows)
    X = reduce(hcat, [it.vector for it in items])
    for spec in (Quantized(SQ(8)), Quantized(SQ(8; range=PerVectorRange())), Quantized(SQ(4); rotation=QRRotation()))
        mktempworkdir() do workdir
            h = create_project(workdir, "q", DenseVectors(Dist.SqL2(); index=Exhaustive(), storage=spec))
            append_items!(h, items)
            index!(h)
            enc = h.engine.quant.encoder
            ref = ExhaustiveSearch(enc.dist, SimilaritySearch.ScalarQuant.sqcodes(enc, X))
            qs = [SimilaritySearch.encodequery(enc, items[i].vector) for i in 1:25:300]
            ids, _ = SimilaritySearch.searchbatch(ref, SimilaritySearch.GenericContext(), SimilaritySearch.VectorDatabase(qs), 10)
            for (j, i) in enumerate(1:25:300)
                @test _ids(search(h, items[i].vector, 10)) == Int.(ids[:, j])
            end
            close_project!(h)
        end
    end
end

@testset "quantized project: overlap with the exact Float32 top-10" begin
    rows = quant_rows(700)
    items = dense_items(rows)
    overlap = Dict{Symbol,Float64}()
    storages = Dict(:sqgu8 => Quantized(SQ(8)), :sqgu4 => Quantized(SQ(4)), :squ8 => Quantized(SQ(8; range=PerVectorRange())))
    mktempworkdir() do workdir
        exact = create_project(workdir, "exact", DenseVectors(Dist.NormCosine(); index=Exhaustive()))
        append_items!(exact, items)
        for kind in (:sqgu8, :sqgu4, :squ8)
            h = create_project(workdir, "q_$kind", DenseVectors(Dist.NormCosine(); index=Exhaustive(), storage=storages[kind]))
            append_items!(h, items)
            index!(h)
            o = 0
            for i in 1:7:700
                gold = Set(_ids(search(exact, items[i].vector, 10)))
                o += length(intersect(gold, _ids(search(h, items[i].vector, 10))))
            end
            overlap[kind] = o / (100 * 10)
            close_project!(h)
        end
        close_project!(exact)
    end
    @info "quantized project: overlap with the exact Float32 top-10, 100 queries over 700 vectors of dimension 128" overlap
    @test overlap[:sqgu8] >= 0.9
    @test overlap[:squ8] >= 0.9
    @test overlap[:sqgu4] >= 0.6
end

@testset "quantized project: kept vectors, later appends and pending items" begin
    rows = quant_rows(200)
    items = dense_items(rows)
    mktempworkdir() do workdir
        h = create_project(workdir, "keep", DenseVectors(; storage=Quantized(SQ(8); keep_vectors=true)))
        append_items!(h, items[1:100])
        index!(h)
        append_items!(h, items[101:200])
        @test fetch_items(h, ["frankenstein_150"])[1].payload == items[150].vector
        @test fetch_items(h, ["frankenstein_5"])[1].payload == items[5].vector
        @test_throws PendingBacklog allknn(h; k=3)
        close_project!(h)
        # The 100 staged vectors are still pending after the reopen.
        h = open_project(workdir, "keep")
        @test all(r -> r._id <= 100, search(h, items[150].vector, 30))
        index!(h)
        @test first(search(h, items[150].vector, 1)).doc_id == "frankenstein_150"
        @test isfile(joinpath(workdir, "keep", "dense_vectors.mmapdb"))
        knn = allknn(h; k=3)
        @test length(knn) == 200
        close_project!(h)
    end

    mktempworkdir() do workdir
        # keep_vectors=false: an append after index! starts a new file of staged vectors, whose
        # first row is item 101.
        h = create_project(workdir, "nokeep", DenseVectors(; index=Exhaustive(), storage=Quantized(SQ(8; range=PerVectorRange()))))
        append_items!(h, items[1:100])
        index!(h)
        # The file of staged vectors is removed, and the encoder gives the dimension of an append.
        @test !isfile(joinpath(workdir, "nokeep", "dense_vectors.mmapdb"))
        @test_throws WrongDimension append_items!(h, [DenseItem(rand(Float32, 3))])
        append_items!(h, items[101:200])
        @test fetch_items(h, ["frankenstein_150"])[1].payload == items[150].vector
        @test fetch_items(h, ["frankenstein_50"])[1].payload === nothing
        close_project!(h)
        h = open_project(workdir, "nokeep")
        @test fetch_items(h, ["frankenstein_150"])[1].payload == items[150].vector
        index!(h)
        @test first(search(h, items[150].vector, 1)).doc_id == "frankenstein_150"
        @test !isfile(joinpath(workdir, "nokeep", "dense_vectors.mmapdb"))
        delete_item!(h, 150)
        @test all(r -> r.doc_id != "frankenstein_150", search(h, items[150].vector, 10))
        close_project!(h)
        h = open_project(workdir, "nokeep"; read_only=true)
        @test all(r -> r.doc_id != "frankenstein_150", search(h, items[150].vector, 10))
        @test close_project!(h) === nothing
    end
end

@testset "quantized project: filters, batches and partitions" begin
    rows = quant_rows(300)
    items = [DenseItem(Float32.(r["vector"]); doc_id=r["doc_id"], meta=Dict{String,Any}("part" => isodd(i) ? "odd" : "even"))
             for (i, r) in enumerate(rows)]
    for index in (Exhaustive(), Graph())
        mktempworkdir() do workdir
            h = create_project(workdir, "p", DenseVectors(; index, storage=Quantized(SQ(8)));
                               partitions=PartitionBy("part"; graph_min=index isa Graph ? 50 : nothing))
            append_items!(h, items)
            index!(h)
            res = search(h, items[2].vector, 10; partition="even")
            @test length(res) == 10 && all(r -> iseven(r._id), res)
            res = search(h, items[3].vector, 10; filter=(rec, m) -> m !== nothing && m["part"] == "odd")
            @test all(r -> isodd(r._id), res)
            batch = search(h, [items[i].vector for i in 1:5], 3)
            @test all(b -> length(b) == 3, batch)
            # An exact index answers a batch as it answers each query. A graph search is
            # approximate, and a search with another `k` can reach other neighbors.
            index isa Exhaustive &&
                @test [_ids(b) for b in batch] == [_ids(search(h, items[i].vector, 3)) for i in 1:5]
            close_project!(h)
            h = open_project(workdir, "p")
            @test all(r -> iseven(r._id), search(h, items[2].vector, 10; partition="even"))
            close_project!(h)
        end
    end
end

@testset "quantized project: what it refuses" begin
    mktempworkdir() do workdir
        @test_throws InvalidOption SQ(16)
        @test_throws InvalidOption GlobalRange(minmax=(1, -1))
        @test_throws InvalidOption GlobalRange(samplesize=0)
        @test_throws InvalidOption PerVectorRange(samplesize=0)
        @test_throws InvalidOption DenseVectors(Dist.Angle(); storage=Quantized(SQ(8)))

        h = create_project(workdir, "c", DenseVectors(; storage=Quantized(SQ(8))))
        @test_throws NothingStaged index!(h)
        @test_throws NothingStaged fit_quantizer!(h)
        close_project!(h)

        # 2 bits pack 4 coordinates per byte, so a dimension of 6 cannot be stored.
        h = create_project(workdir, "d", DenseVectors(; index=Exhaustive(), storage=Quantized(SQ(2))))
        append_items!(h, [DenseItem(rand(Float32, 6)) for _ in 1:10])
        @test_throws InvalidOption index!(h)
        close_project!(h)

        h = create_project(workdir, "e", DenseVectors(; index=Exhaustive()))
        @test quantizer_info(h) === nothing
        @test_throws InvalidOption fit_quantizer!(h)
        close_project!(h)

        # A fixed range: nothing to fit, and the range reported is the one given.
        h = create_project(workdir, "f", DenseVectors(; index=Exhaustive(), storage=Quantized(SQ(8; range=GlobalRange(minmax=(-1, 1))))))
        append_items!(h, dense_items(quant_rows(20)))
        fit_quantizer!(h)
        info = quantizer_info(h)
        @test info.fitted && info.min == -1f0 && isapprox(info.max, 1f0; atol=1e-5)
        @test_throws WrongDimension search(h, rand(Float32, 64), 3)
        close_project!(h)
    end
end

@testset "quantized project: calibrate! and the operations over the whole dataset" begin
    items = dense_items(quant_rows(300))
    mktempworkdir() do workdir
        h = create_project(workdir, "w", DenseVectors(; storage=Quantized(SQ(8))))
        append_items!(h, items)
        index!(h)
        report = calibrate!(h; levels=[0.9], numqueries=16)
        @test length(report) == 1
        queries = [items[i].vector for i in 1:10]
        @test length(calibrate!(h; levels=[0.8], queries)) == 1
        @test length(fft(h, 5).centers) == 5
        @test length(neardup(h, 0.01f0).assign) == 300
        @test length(closestpairs(h; k=5)) == 5
        close_project!(h)
    end
end

# Resident memory of an opened project, measured in a new process for each project: the process
# compiles a search on a small project first, then reads its resident memory (`/proc/self/statm`),
# opens the project, runs one search, and reads it again. The difference is the memory of the
# opened project: 200,000 × 256 × 4 bytes = 205 MB of Float32 vectors, or 51 MB of 8-bit codes.
# `Sys.maxrss` does not measure this: it is the peak of the process, which package loading and
# compilation set, and it was 1,638 MB for both projects on 2026-11-20.
@heavy "quantized project: resident memory follows the size of the codes" begin
    # It reads `/proc/self/statm`, which only Linux has.
    if Sys.islinux()
        n, dim = 200_000, 256
        rng = Xoshiro(1)
        vecs = [(v = randn(rng, Float32, dim); v ./ sqrt(sum(abs2, v))) for _ in 1:n]
        mktempworkdir() do workdir
            # A graph project, not indexed: opening it copies every Float32 vector into memory, as
            # opening an exact project does. An exact project is not used because its append saves
            # the whole index after each item, which takes hours at this size.
            f = create_project(workdir, "float", DenseVectors(Dist.NormCosine()))
            append_items!(f, [DenseItem(v) for v in vecs])
            close_project!(f)
            q = create_project(workdir, "codes", DenseVectors(Dist.NormCosine(); index=Exhaustive(), storage=Quantized(SQ(8))))
            append_items!(q, [DenseItem(v) for v in vecs])
            index!(q)
            close_project!(q)
            query = vecs[1]
            rss(name) = begin
                script = """
                    using SimilaritySearchEngine
                    rss() = parse(Int, split(read("/proc/self/statm", String))[2]) * Int(ccall(:getpagesize, Cint, ()))
                    for (n, spec) in (("warm_float", DenseVectors()), ("warm_codes", DenseVectors(; index=Exhaustive(), storage=Quantized(SQ(8)))))
                        w = mktempdir()
                        h = create_project(w, n, spec)
                        append_items!(h, [DenseItem(rand(Float32, $dim)) for _ in 1:64]); index!(h)
                        search(h, rand(Float32, $dim), 10); close_project!(h)
                    end
                    GC.gc(); GC.gc()
                    before = rss()
                    h = open_project($(repr(workdir)), $(repr(name)); read_only=true)
                    search(h, $(repr(query)), 10)
                    GC.gc(); GC.gc()
                    println(rss() - before)
                    close_project!(h)
                    """
                cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $script`
                parse(Int, last(split(strip(read(cmd, String)), '\n')))
            end
            rss_float = rss("float")
            rss_codes = rss("codes")
            @info "quantized project: resident memory added by opening the project and one search, 200,000 vectors of dimension 256" float_MB=rss_float ÷ 2^20 codes_MB=rss_codes ÷ 2^20
            # The vectors take 154 MB more in Float32; at least half of that must show.
            @test rss_float - rss_codes > (n * dim * 3) ÷ 2
        end
    end
end
