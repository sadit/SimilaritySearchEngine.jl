@testset "sparse dataset: caller-encoded vectors, no vocabulary anywhere" begin
    mktempworkdir() do workdir
        items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:100]]
        sitems = sparse_items(items)

        h = create_project(workdir, "sparse_ds"; engine=SparseEngine, dimension=SPARSE_DIM,
                           distance=Dist.NormCosine())
        @test payload_kind(h.engine) === :sparse
        @test h.engine.backend.dimension == SPARSE_DIM
        @test append_items!(h, sitems) == 100

        # An InvertedFile indexes on insertion: unlike a SearchGraph there is no backlog, so
        # the item is searchable before index! and index! is a no-op kept for uniformity.
        res = search(h, payload(sitems[1]), 5)
        @test length(res) == 5
        @test res[1].doc_id == "frankenstein_1"
        @test res[1].distance ≈ 0.0 atol=1e-5
        index!(h)
        @test [r._id for r in search(h, payload(sitems[1]), 5)] == [r._id for r in res]

        # the payload comes back as a sparse vector, not densified on the way out
        fetched = fetch_items(h, ["frankenstein_3"])
        @test length(fetched) == 1
        @test fetched[1].payload isa SparseVector{Float32,Int32}
        @test nonzeroinds(fetched[1].payload) == nonzeroinds(payload(sitems[3]))
        @test fetched[1].meta["word_count"] == items[3]["word_count"]

        delete_item!(h, 2)
        @test exists(h, ["frankenstein_2"])[1].deleted
        hit = findfirst(r -> r._id == 2, search(h, payload(sitems[1]), 100))
        @test hit === nothing || search(h, payload(sitems[1]), 100)[hit].deleted

        filtered = search(h, payload(sitems[1]), 5;
                          filter=(record, meta) -> record.doc_id == "frankenstein_1")
        @test length(filtered) == 1 && filtered[1].doc_id == "frankenstein_1"

        # a project of sparse vectors takes SparseItems and nothing else
        @test_throws PayloadMismatch append_items!(h, dense_items(items[1:1]))
        @test_throws PayloadMismatch append_items!(h, text_items(items[1:1]))
        close_project!(h)

        h2 = open_project(workdir, "sparse_ds")
        @test payload_kind(h2.engine) === :sparse
        @test h2.engine.backend.dimension == SPARSE_DIM
        reopened = search(h2, payload(sitems[1]), 5)
        @test [r._id for r in reopened] == [r._id for r in res]
        @test exists(h2, ["frankenstein_2"])[1].deleted   # the tombstone survived
        close_project!(h2)
    end
end

@testset "sparse project: the query is a sparse vector of the project's own dimension" begin
    mktempworkdir() do workdir
        items = sparse_items([JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:20]])
        h = create_project(workdir, "sq_ds"; engine=SparseEngine, dimension=SPARSE_DIM,
                           distance=Dist.NormCosine())
        append_items!(h, items)
        # A dense query would `convert` cleanly and then die inside the library on a Float32
        # used as a posting-list index, so it is refused here where the message can say why.
        @test_throws PayloadMismatch search(h, rand(Float32, SPARSE_DIM), 5)
        @test_throws WrongDimension search(h, sparsevec(Int32[1, 2], Float32[1, 0], 8), 5)
        @test length(search(h, payload(items[1]), 3)) == 3
        close_project!(h)
    end
end
