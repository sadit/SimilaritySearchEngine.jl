# Keyword followers and posting lists (PLAN.md §5.11, issue #5, phase 4).

kw_keys(i) = vcat(i % 2 == 0 ? ["even"] : ["odd"], i % 3 == 0 ? ["three"] : String[], i % 10 == 0 ? ["ten"] : String[],
                  i == 7 ? ["seven"] : String[])
kw_item(i) = DenseItem(Float32[i, 1, 2]; doc_id="d$i", keywords=(i % 13 == 0 ? String[] : kw_keys(i)))
kw_has(i, key) = i % 13 != 0 && key in kw_keys(i)

@testset "keyword follower: posting lists, search by keys, reopen" begin
    mktempworkdir() do workdir
        h = create_project(workdir, "lead", DenseVectors(; index=Exhaustive()))
        append_items!(h, [kw_item(i) for i in 1:60])
        f = create_follower(h, "kw", KeywordFollower())
        @test f.position == 60
        @test posting_list(f, "three") == Int32[i for i in 1:60 if kw_has(i, "three")]
        @test posting_list(f, "three"; limit=3) == Int32[3, 6, 9]
        @test posting_list(f, "three"; limit=0) == Int32[]
        @test posting_list(f, "absent") == Int32[]
        @test_throws InvalidOption posting_list(f, "three"; limit=-1)
        st = follower_status(f).stats
        @test st.keys == 5 && st.items_without_keys == count(i -> i % 13 == 0, 1:60) && st.longest == 30 - 2

        # t=1: an item with any key of the query; t=2: an item with both.
        both = [r._id for r in search(f, ["three", "ten"], 60; t=2)]
        @test sort(both) == [i for i in 1:60 if kw_has(i, "three") && kw_has(i, "ten")]
        any_ = [r._id for r in search(f, ["three", "ten"], 60)]
        @test sort(any_) == [i for i in 1:60 if kw_has(i, "three") || kw_has(i, "ten")]
        # Jaccard: an item whose keys equal the query comes first, at distance 0.
        top = first(search(f, ["odd", "seven"], 3))
        @test top._id == 7 && top.distance == 0 && top.doc_id == "d7"
        @test_throws InvalidOption search(f, ["odd"], 3; t=0)
        # A filter and a deleted item work as in the search of the leader.
        delete_item!(h, 30)
        r30 = [r for r in search(f, ["ten"], 10) if r._id == 30]
        @test only(r30).deleted
        @test all(r -> r._id > 30, search(f, ["ten"], 10; filter=(rec, meta) -> rec._id > 30, candidates=10))
        @test_throws UnsupportedOperation open_follower(h, "kw=three")

        # New items reach the follower with the policy of the leader (OnIndex).
        append_items!(h, [kw_item(i) for i in 61:70])
        @test f.position == 60
        index!(h)
        @test f.position == 70 && posting_list(f, "ten") == Int32[10, 20, 30, 40, 50, 60, 70]

        # Processing the same items again leaves the inverted file as it was.
        f.position = 55
        @test sync!(f; block=4) == 15
        @test posting_list(f, "ten") == Int32[10, 20, 30, 40, 50, 60, 70]
        @test length(f.state.index) == 70
        close_project!(h)

        h = open_project(workdir, "lead")
        f = open_follower(h, "kw")
        @test posting_list(f, "three") == Int32[i for i in 1:70 if kw_has(i, "three")]
        @test first(search(f, ["odd", "seven"], 3))._id == 7
        close_project!(h)
        hr = open_project(workdir, "lead"; read_only=true)
        @test posting_list(open_follower(hr, "kw"), "ten"; limit=2) == Int32[10, 20]
        close_project!(hr)
    end
end

@testset "keyword follower: other key sources and distances" begin
    mktempworkdir() do workdir
        rows = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:50]]
        h = create_project(workdir, "txt", TextDocuments(FitFromCorpus()))
        append_items!(h, [TextItem(r["text"]; doc_id=r["doc_id"], keywords=r["keywords"],
                                   meta=Dict{String,Any}("tags" => ["t$(i % 4)", "u$(i % 5)"])) for (i, r) in enumerate(rows)])
        index!(h)
        k = create_follower(h, "kw", KeywordFollower())
        names = unique(vcat([r["keywords"] for r in rows]...))
        name = first(n for n in names if count(r -> n in r["keywords"], rows) >= 2)
        @test posting_list(k, name) == Int32[i for (i, r) in enumerate(rows) if name in r["keywords"]]
        tags = create_follower(h, "tags", KeywordFollower(MetaValue("tags"); distance=Dist.Sets.Dice()))
        @test posting_list(tags, "t1") == Int32[i for i in 1:50 if i % 4 == 1]
        @test sort([r._id for r in search(tags, ["t1", "u2"], 50; t=2)]) == [i for i in 1:50 if i % 4 == 1 && i % 5 == 2]
        @test_throws UnsupportedOperation search(open_follower(h, "kw"), Float32[1, 2], 3)
        close_project!(h)
    end
end

@testset "posting_list of a text and a sparse project" begin
    mktempworkdir() do workdir
        rows = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:80]]
        for textindex in (BM25(), VectorSpace())
            h = create_project(workdir, "txt_$(nameof(typeof(textindex)))", TextDocuments(FitFromCorpus(); index=textindex))
            append_items!(h, [TextItem(r["text"]; doc_id=r["doc_id"]) for r in rows])
            @test posting_list(h, "monster") == Int32[]          # nothing is indexed yet
            index!(h)
            voc = text_profile(h).model.voc
            # A token held by several documents, and the documents whose vectors hold it.
            docs = IndexEngine.document_tokens(h.engine, 1:80)
            t = argmax(id -> count(d -> id in d, docs), 1:length(voc.token))
            token = voc.token[t]
            want = Int32[i for i in 1:80 if UInt32(t) in docs[i]]
            got = posting_list(h, token)
            @test !isempty(got) && issorted(got) && got == want
            @test posting_list(h, token; limit=5) == got[1:5]
            @test posting_list(h, "zzzz-not-a-token") == Int32[]
            @test_throws InvalidOption posting_list(h, 3)
            close_project!(h)
        end

        s = create_project(workdir, "sp", SparseVectors(16))
        append_items!(s, [SparseItem(sparsevec(Int32[1 + i % 4, 9], Float32[1, 1], 16)) for i in 1:20])
        @test posting_list(s, 9) == Int32.(1:20)
        @test posting_list(s, 2) == Int32[i for i in 1:20 if 1 + i % 4 == 2]
        @test_throws InvalidOption posting_list(s, 17)
        @test_throws InvalidOption posting_list(s, "9")
        close_project!(s)

        d = create_project(workdir, "dense", DenseVectors(; index=Exhaustive()))
        append_items!(d, [DenseItem(Float32[1, 2])])
        @test_throws UnsupportedOperation posting_list(d, 1)
        close_project!(d)
    end
end
