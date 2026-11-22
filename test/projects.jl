@testset "project creation: the spec says what the project holds and what indexes it" begin
    mktempworkdir() do workdir
        # defaults: a dense project on a SearchGraph, which is what `create_project(w, ds)` means
        h = create_project(workdir, "defaults")
        @test payload_kind(h.engine) === :dense
        @test h.engine.backend isa SimilaritySearchEngine.IndexEngine.GraphBackend
        @test project_spec(h) == DenseVectors()
        @test project_spec(h) == DenseVectors(Dist.SqL2(); index=Graph(minrecall=0.9), storage=Float32Vectors())
        close_project!(h)

        # an exact dense backend is the same engine with a different index under it
        he = create_project(workdir, "exact", DenseVectors(; index=Exhaustive()))
        @test payload_kind(he.engine) === :dense
        @test he.engine.backend.index isa ExhaustiveSearch
        close_project!(he)
        hp = create_project(workdir, "pexact", DenseVectors(; index=Exhaustive(parallel=true)))
        @test hp.engine.backend.index isa SimilaritySearchEngine.ParallelExhaustiveSearch
        close_project!(hp)

        # what a spec still refuses: values out of range, and pairs no type can exclude
        @test_throws InvalidOption SparseVectors(0)
        @test_throws InvalidOption Graph(minrecall=1.5)
        @test_throws InvalidOption Graph(minrecall=0)
        @test_throws InvalidOption SQ(3)
        @test_throws InvalidOption GlobalRange(minmax=(1, -1))
        @test_throws InvalidOption PartitionBy("")
        @test_throws InvalidOption PartitionBy("f"; graph_min=0)
        @test_throws InvalidOption Runtime(maxbatches=0)
        @test_throws InvalidOption DenseVectors(Dist.Angle(); storage=Quantized())

        # a sparse project cannot be partitioned yet, and the refusal leaves nothing behind
        @test_throws InvalidOption create_project(workdir, "sparse_part", SparseVectors(16); partitions=PartitionBy("f"))
        @test !isdir(joinpath(workdir, "sparse_part"))

        # `graph_min` is the size from which a dense partition has its own graph; a text partition
        # is posting lists
        @test_throws InvalidOption create_project(workdir, "text_gmin", TextDocuments(FitFromCorpus());
                                                  partitions=PartitionBy("f"; graph_min=10))
        @test !isdir(joinpath(workdir, "text_gmin"))

        # edit correction applies to the tokens of a text project only
        @test_throws InvalidOption create_project(workdir, "dense_edits"; runtime=Runtime(edit_correction=true))
        @test !isdir(joinpath(workdir, "dense_edits"))
    end
end

@testset "project spec: saved, returned and read from the legacy fields" begin
    mktempworkdir() do workdir
        specs = [
            "graph" => DenseVectors(Dist.NormCosine(); index=Graph(minrecall=0.95)),
            "graph_nomr" => DenseVectors(; index=Graph(minrecall=nothing)),
            "exact" => DenseVectors(Dist.SqL2(); index=Exhaustive(parallel=true)),
            "quant" => DenseVectors(Dist.NormCosine(); index=Exhaustive(),
                                    storage=Quantized(SQ(4; range=PerVectorRange(samplesize=100));
                                                      rotation=QRRotation(), keep_vectors=true)),
            "quant_g" => DenseVectors(Dist.SqL2(); storage=Quantized(SQ(8; range=GlobalRange(minmax=(-1, 1))))),
            "sparse" => SparseVectors(64, Dist.NormCosine()),
            "bm25" => TextDocuments(FitFromCorpus()),
            "vs" => TextDocuments(FitFromCorpus(min_ndocs=2); index=VectorSpace(distance=Dist.NormCosine())),
            "default_profile" => TextDocuments(DefaultProfile(:es; nickname="whatever", refit=false, max_documents=10)),
        ]
        for (name, spec) in specs
            h = create_project(workdir, name, spec)
            @test project_spec(h) == spec
            close_project!(h)
            h = open_project(workdir, name; read_only=true)
            got = project_spec(h)
            if spec isa TextDocuments && spec.model isa FitFromCorpus
                # a FitFromCorpus is read back from `:fitspec`, a value without `==`
                @test got isa TextDocuments && got.index == spec.index && got.model isa FitFromCorpus
                @test got.model.min_ndocs == spec.model.min_ndocs
            else
                @test got == spec
            end
            close_project!(h)
        end

        # A project saved before issue #2 has no `:spec`: it is built from the old fields, and a
        # writable open saves it.
        legacy = [
            "lg" => (DenseVectors(Dist.NormCosine(); index=Graph(minrecall=0.8)),
                     (kind=:dense, backend=:graph, distance=Dist.NormCosine(), minrecall=0.8f0)),
            "lx" => (DenseVectors(Dist.SqL2(); index=Exhaustive(parallel=true)),
                     (kind=:dense, backend=:parallel_exhaustive)),
            "ls" => (SparseVectors(32, Dist.NormCosine()),
                     (kind=:sparse, backend=:inverted_file, distance=Dist.NormCosine(), dimension=32)),
            "lt" => (TextDocuments(FitFromCorpus(); index=VectorSpace(distance=Dist.NormCosine())),
                     (kind=:text, backend=:text_inverted_file, distance=Dist.NormCosine())),
        ]
        for (name, (spec, fields)) in legacy
            h = create_project(workdir, name, spec)
            delete!(h.store.dict, "spec")
            SimilaritySearchEngine.Persistence.save_fields!(h.store, fields)
            # An exact project of that time saved its whole index in `:index` (issue #3).
            spec isa DenseVectors{Exhaustive} &&
                SimilaritySearchEngine.Persistence.save_field!(h.store, :index, h.engine.backend.index)
            close_project!(h)
            h = open_project(workdir, name)
            got = project_spec(h)
            if spec isa TextDocuments
                @test got isa TextDocuments && got.index == spec.index && got.model isa FitFromCorpus
            else
                @test got == spec
            end
            @test SimilaritySearchEngine.Persistence.has_field(h.store, :spec)
            close_project!(h)
        end
    end
end

@testset "project creation: the text model is stated explicitly" begin
    mktempworkdir() do workdir
        # a text spec has no default model: the default it would have to pick freezes the
        # vocabulary at the first index! call, so the constructor requires one
        @test_throws MethodError TextDocuments()
        h = create_project(workdir, "with_model", TextDocuments(FitFromCorpus()))
        @test payload_kind(h.engine) === :text
        close_project!(h)
    end
end

@testset "a closed handle raises ProjectClosed (issue #4)" begin
    mktempworkdir() do workdir
        q = rand(Float32, 8)
        h = create_project(workdir, "closed_dense")
        append_items!(h, [DenseItem(rand(Float32, 8); doc_id="d$i") for i in 1:50])
        index!(h)
        @test length(search(h, q, 3)) == 3
        close_project!(h)
        @test close_project!(h) === nothing          # a second close does nothing
        err = try searchbatch(h, [q], 3); nothing catch e; e end
        @test err isa ProjectClosed && err isa ConflictingState && err.dataset == "closed_dense"
        @test_throws ProjectClosed search(h, q, 3)
        @test_throws ProjectClosed search(h, [q, q], 3)
        @test_throws ProjectClosed append_items!(h, DenseItem(q; doc_id="late"))
        @test_throws ProjectClosed index!(h)
        @test_throws ProjectClosed fetch_items(h, ["d1"])
        @test_throws ProjectClosed exists(h, [1])
        @test_throws ProjectClosed delete_item!(h, 1)
        @test_throws ProjectClosed calibrate!(h)
        @test_throws ProjectClosed allknn(h; k=2)
        @test_throws ProjectClosed compact_project!(h)
        @test_throws ProjectClosed project_spec(h)
        @test_throws ProjectClosed meta_schema(h)
        @test_throws ProjectClosed partitions(h)
        @test_throws ProjectClosed quantizer_info(h)

        # the project itself is intact: a new handle serves it
        h2 = open_project(workdir, "closed_dense")
        @test length(search(h2, q, 3)) == 3
        close_project!(h2)

        ht = create_project(workdir, "closed_text", TextDocuments(FitFromCorpus()))
        append_items!(ht, [TextItem("a closed handle raises an error $i"; doc_id="t$i") for i in 1:20])
        index!(ht)
        close_project!(ht)
        @test_throws ProjectClosed ftsearch(ht, "closed handle", 3)
        @test_throws ProjectClosed ftexplain(ht, "closed handle")
        @test_throws ProjectClosed vocabulary_report(ht)

        hs = create_project(workdir, "closed_sparse", SparseVectors(16))
        close_project!(hs)
        @test_throws ProjectClosed search(hs, sparsevec([1], [1f0], 16), 3)
    end
end
