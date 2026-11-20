@testset "project creation: the engine says what it holds, the backend says what holds it" begin
    mktempworkdir() do workdir
        # defaults: a dense project on a SearchGraph, which is what `create_project(w, ds)` means
        @test default_backend(DenseEngine) === SearchGraph
        @test default_backend(SparseEngine) === InvertedFile
        @test default_backend(FullTextEngine) === BM25InvertedFile
        h = create_project(workdir, "defaults")
        @test payload_kind(h.engine) === :dense
        @test h.engine.backend isa SimilaritySearchEngine.IndexEngine.GraphBackend
        close_project!(h)

        # an exact dense backend is the same engine with a different index under it
        he = create_project(workdir, "exact"; engine=DenseEngine, backend=ExhaustiveSearch)
        @test payload_kind(he.engine) === :dense
        @test he.engine.backend.index isa ExhaustiveSearch
        close_project!(he)

        # every legal pairing is in BACKENDS, and only those
        @test Set(keys(BACKENDS)) == Set([DenseEngine, SparseEngine, FullTextEngine])
        for (engine, backends) in BACKENDS, b in backends
            @test SimilaritySearchEngine.IndexEngine.validate_backend(engine, b) === b
        end
        @test_throws UnknownBackend SimilaritySearchEngine.IndexEngine.validate_backend(SparseEngine, SearchGraph)
        @test_throws UnknownBackend SimilaritySearchEngine.IndexEngine.validate_backend(DenseEngine, BM25InvertedFile)

        # a sparse project has to be sized; nothing else may be
        @test_throws InvalidOption create_project(workdir, "nodim"; engine=SparseEngine)
        @test_throws InvalidOption create_project(workdir, "dim_on_dense"; dimension=16)
        @test_throws InvalidOption create_project(workdir, "dim_on_text"; engine=FullTextEngine,
                                                   textmodel=FitFromCorpus(), dimension=16)
        @test_throws ArgumentError create_project(workdir, "zerodim"; engine=SparseEngine, dimension=0)

        # a text model belongs to a text project only -- and the check is the engine's, not
        # the backend's, because InvertedFile is a legal backend for both kinds
        @test_throws InvalidOption create_project(workdir, "sparse_model"; engine=SparseEngine,
                                                   dimension=16, textmodel=FitFromCorpus())
        @test !isdir(joinpath(workdir, "sparse_model"))

        # the pre-restructuring keyword names its replacement instead of being ignored
        err = try; create_project(workdir, "old"; index_type=SearchGraph); catch e; e end
        @test err isa InvalidOption
        @test err.option === :index_type
        @test occursin("engine=", err.msg) && occursin("backend=", err.msg)
        @test !isdir(joinpath(workdir, "old"))

        @test_throws UnknownBackend create_project(workdir, "bad_engine"; engine=Int)
    end
end

@testset "project creation: the text model is stated explicitly, or not at all" begin
    mktempworkdir() do workdir
        # omitting it on a text project is an error rather than a silent default, because
        # the default it would have to pick freezes the vocabulary at the first index! call
        @test_throws InvalidOption create_project(workdir, "no_model"; engine=FullTextEngine, backend=BM25InvertedFile)

        msg = try
            create_project(workdir, "no_model"; engine=FullTextEngine, backend=BM25InvertedFile)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("BaseProfile", msg) && occursin("FitFromCorpus", msg)

        # ... and the failed call left nothing behind: no directory, no held write lock,
        # so the same name is still free to create properly
        @test !isdir(joinpath(workdir, "no_model"))
        h = create_project(workdir, "no_model"; engine=FullTextEngine, backend=BM25InvertedFile, textmodel=FitFromCorpus())
        close_project!(h)

        # handing a text model to a dense project is an error too, not an ignored keyword
        @test_throws InvalidOption create_project(workdir, "dense_model"; engine=DenseEngine, backend=SearchGraph,
                                                   textmodel=FitFromCorpus())
        @test !isdir(joinpath(workdir, "dense_model"))
    end
end
