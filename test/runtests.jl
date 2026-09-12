using Test
using SimilaritySearchEngine
using SimilaritySearch: SearchGraph, Dist, MatrixDatabase
using TextSearch: BM25InvertedFile, InvertedFile, TextInvertedFile, NormalizationConfig,
                  AppliedArtifacts, vocsize, gettrainsize, gettextconfig,
                  lineage_summary, token2id
using JSON
using RocksDB
using SparseArrays: sparsevec, SparseVector, nonzeros, nonzeroinds

const FRANKENSTEIN_PATH = joinpath(@__DIR__, "data", "frankenstein.jsonl")

function mktempworkdir(f)
    dir = mktempdir()
    try
        f(dir)
    finally
        rm(dir; recursive=true, force=true)
    end
end

# The engine takes typed items, so a corpus read off disk has to be converted -- and doing it
# here, in the caller, is the whole point of the change: the test knows that "vector" is the
# payload and "verb_count" is metadata, and the library no longer has to guess from key names.
const PAYLOAD_KEYS = ("vector", "text", "doc_id", "keywords", "ref")
_meta_of(row) = Dict{String,Any}(k => v for (k, v) in row if !(k in PAYLOAD_KEYS))
dense_items(rows) = [DenseItem(r["vector"]; doc_id=r["doc_id"], keywords=r["keywords"],
                               refs=r["ref"], meta=_meta_of(r)) for r in rows]
text_items(rows) = [TextItem(r["text"]; doc_id=r["doc_id"], keywords=r["keywords"],
                             refs=r["ref"], meta=_meta_of(r)) for r in rows]

# A sparse project's items are vectors the *caller* encoded -- the engine has no vocabulary and
# never sees a token. Hashing words into a fixed dimension is the smallest honest stand-in for
# whatever a caller's own encoder does (a learned sparse retriever, a feature table, a set of ids),
# and it is deliberately not TextSearch's encoding: that path is what `FullTextEngine` is for.
const SPARSE_DIM = 4096
function sparse_items(rows)
    map(rows) do r
        counts = Dict{Int32,Float32}()
        for w in split(lowercase(r["text"]), r"[^\p{L}\p{N}]+"; keepempty=false)
            k = Int32(mod(hash(w), SPARSE_DIM) + 1)
            counts[k] = get(counts, k, 0f0) + 1f0
        end
        ind = sort!(collect(keys(counts)))
        val = Float32[counts[i] for i in ind]
        val ./= sqrt(sum(abs2, val))          # NormCosine reads its inputs as already normalized
        SparseItem(sparsevec(ind, val, SPARSE_DIM); doc_id=r["doc_id"], keywords=r["keywords"],
                   refs=r["ref"], meta=_meta_of(r))
    end
end

# A text policy that keeps case and diacritics, which is what makes orthographic bridging
# possible at all: `derive_variants` has nothing to do under the default `TextConfig()`, whose
# normalization already folds both.
const CASED_ES = TextConfig(normalization=NormalizationConfig(lc=false, del_diac=false), language=:es)

# 60 documents spelling it "musica" with an accent against a single one without: far enough
# apart for `QueryPolicy`'s default `negligible_ratio=50` to read the bare spelling as wrong.
const ACCENT_ITEMS = vcat(
    [TextItem("la m\u00fasica cl\u00e1sica de la ciudad $i"; doc_id="acc_$i") for i in 1:60],
    [TextItem("una musica rara sin acento"; doc_id="bare")])

@testset "SimilaritySearchEngine.jl" begin

    include("policy.jl")

    @testset "dense dataset: append (stage), index!, search, calibrate, allknn, delete, fetch" begin
        mktempworkdir() do workdir
            items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:100]]

            h = create_project(workdir, "dense_ds")
            inserted = append_items!(h, dense_items(items))
            @test inserted == 100

            # append_items! only stages a SearchGraph's vectors -- nothing is searchable
            # until an explicit index! call catches up the graph-connection backlog.
            @test isempty(search(h, items[1]["vector"], 5))

            index!(h)

            res = search(h, items[1]["vector"], 5)
            @test length(res) == 5
            @test res[1].doc_id == "frankenstein_1"
            @test res[1]._id == 1
            @test res[1].distance ≈ 0.0 atol=1e-6

            # calibrate! returns a minrecall::Float32 => BeamSearch table, not a lone BeamSearch.
            bs = calibrate!(h; numqueries=20)
            @test !isempty(bs)
            @test all(v -> v.bsize isa Integer, values(bs))

            ak = allknn(h; k=5)
            @test length(ak) == 100
            @test ak[1]._id == 1
            @test ak[1].neighbors[1] == 1
            @test ak[1].dists[1] ≈ 0.0 atol=1e-6

            e_before = exists(h, ["frankenstein_2", "does-not-exist"])
            @test e_before[1].exists && !e_before[1].deleted
            @test !e_before[2].exists

            delete_item!(h, 2)
            e_after = exists(h, ["frankenstein_2"])
            @test e_after[1].exists && e_after[1].deleted

            # A soft-deleted id is reported with deleted=true, not hidden/backfilled.
            res_after_delete = search(h, items[1]["vector"], 100)
            deleted_idx = findfirst(r -> r._id == 2, res_after_delete)
            @test deleted_idx !== nothing
            @test res_after_delete[deleted_idx].deleted
            # a soft-deleted hit reports no doc_id at all, rather than the internal id dressed
            # up as one
            @test res_after_delete[deleted_idx].doc_id === nothing

            fetched = fetch_items(h, ["frankenstein_3"])
            @test length(fetched) == 1
            @test fetched[1].doc_id == "frankenstein_3"
            @test fetched[1].keywords == items[3]["keywords"]
            @test fetched[1].meta["word_count"] == items[3]["word_count"]
            # the payload comes back too -- the dictionary form could not return it at all
            @test fetched[1].payload isa Vector{Float32}
            @test fetched[1].payload ≈ Float32.(items[3]["vector"])

            filtered = search(h, items[1]["vector"], 5; filter=(record, meta) -> record.doc_id == "frankenstein_1")
            @test length(filtered) == 1
            @test filtered[1].doc_id == "frankenstein_1"

            close_project!(h)
        end
    end

    @testset "whole-dataset operations: fft, dnet, neardup, closestpairs, bichromatic_kclosestpairs" begin
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

    @testset "whole-dataset operations run safely concurrently with append_items!/index!" begin
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
            _is_expected_backlog_refusal(e) = e isa ErrorException && occursin("requires no pending backlog", e.msg)

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
            @test_throws ErrorException append_items!(h, dense_items(items[1:1]))
            @test_throws ErrorException append_items!(h, text_items(items[1:1]))
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
            @test_throws ErrorException search(h, rand(Float32, SPARSE_DIM), 5)
            @test_throws ErrorException search(h, sparsevec(Int32[1, 2], Float32[1, 0], 8), 5)
            @test length(search(h, payload(items[1]), 3)) == 3
            close_project!(h)
        end
    end

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
            @test_throws ErrorException SimilaritySearchEngine.IndexEngine.validate_backend(SparseEngine, SearchGraph)
            @test_throws ErrorException SimilaritySearchEngine.IndexEngine.validate_backend(DenseEngine, BM25InvertedFile)

            # a sparse project has to be sized; nothing else may be
            @test_throws ErrorException create_project(workdir, "nodim"; engine=SparseEngine)
            @test_throws ErrorException create_project(workdir, "dim_on_dense"; dimension=16)
            @test_throws ErrorException create_project(workdir, "dim_on_text"; engine=FullTextEngine,
                                                       textmodel=FitFromCorpus(), dimension=16)
            @test_throws ArgumentError create_project(workdir, "zerodim"; engine=SparseEngine, dimension=0)

            # a text model belongs to a text project only -- and the check is the engine's, not
            # the backend's, because InvertedFile is a legal backend for both kinds
            @test_throws ErrorException create_project(workdir, "sparse_model"; engine=SparseEngine,
                                                       dimension=16, textmodel=FitFromCorpus())
            @test !isdir(joinpath(workdir, "sparse_model"))

            # the pre-restructuring keyword names its replacement instead of being ignored
            err = try; create_project(workdir, "old"; index_type=SearchGraph); catch e; e end
            @test err isa ErrorException
            @test occursin("engine=", err.msg) && occursin("backend=", err.msg)
            @test !isdir(joinpath(workdir, "old"))

            @test_throws ErrorException create_project(workdir, "bad_engine"; engine=Int)
        end
    end

    @testset "text (bm25) dataset: stage, index! (trains + catches up), ftsearch" begin
        mktempworkdir() do workdir
            items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:50]]
            first_batch, second_batch = items[1:30], items[31:50]

            h = create_project(workdir, "text_ds"; engine=FullTextEngine, backend=BM25InvertedFile, textmodel=FitFromCorpus())
            inserted = append_items!(h, text_items(first_batch))
            @test inserted == 30

            # staged raw text, not yet trained/encoded -- nothing searchable yet.
            @test isempty(ftsearch(h, first_batch[1]["text"], 3))

            # first index! call trains the Vocabulary from everything staged so far, and
            # encodes/indexes the same backlog.
            index!(h)
            res = ftsearch(h, first_batch[1]["text"], 3)
            @test length(res) == 3
            @test res[1].doc_id == "frankenstein_1"

            # idempotent: nothing new staged, safe to call again.
            index!(h)
            @test length(ftsearch(h, first_batch[1]["text"], 3)) == 3

            # a second batch is staged but stays un-indexed until the next index! call.
            append_items!(h, text_items(second_batch))
            ids_before = Set(r.doc_id for r in ftsearch(h, second_batch[1]["text"], 100))
            @test !(second_batch[1]["doc_id"] in ids_before)

            index!(h)
            ids_after = Set(r.doc_id for r in ftsearch(h, second_batch[1]["text"], 100))
            @test second_batch[1]["doc_id"] in ids_after

            close_project!(h)
        end
    end

    @testset "text (weighted InvertedFile) dataset: stage, index!, ftsearch" begin
        mktempworkdir() do workdir
            items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:20]]
            first_batch, second_batch = items[1:10], items[11:20]

            h = create_project(workdir, "invfile_ds"; engine=FullTextEngine, backend=InvertedFile, textmodel=FitFromCorpus())
            append_items!(h, text_items(first_batch))
            @test isempty(ftsearch(h, first_batch[2]["text"], 3))

            index!(h)
            @test length(ftsearch(h, first_batch[2]["text"], 3)) == 3

            append_items!(h, text_items(second_batch))
            ids_before = Set(r.doc_id for r in ftsearch(h, second_batch[1]["text"], 100))
            @test !(second_batch[1]["doc_id"] in ids_before)

            index!(h)
            ids_after = Set(r.doc_id for r in ftsearch(h, second_batch[1]["text"], 100))
            @test second_batch[1]["doc_id"] in ids_after

            close_project!(h)
        end
    end

    @testset "dense dataset: a staged-but-not-yet-indexed backlog survives close + reopen" begin
        mktempworkdir() do workdir
            items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:20]]
            indexed, backlog = items[1:12], items[13:20]

            # The dense counterpart of the text test below, and it earns its own case because
            # its durability comes from somewhere else entirely: staged vectors live in a
            # `MMapMatrixDatabase`, whose persistence is opt-in -- `push_item!`/`append_items!`
            # advance the in-memory count and write the mapped bytes without msyncing them or
            # persisting the advanced header. `append_items!` flushing once per batch is what
            # makes this survive, and nothing else in this suite would notice if it stopped.
            h = create_project(workdir, "dense_backlog_ds")
            append_items!(h, dense_items(indexed))
            index!(h)
            append_items!(h, dense_items(backlog))   # staged, deliberately left un-indexed
            close_project!(h)

            h2 = open_project(workdir, "dense_backlog_ds")
            # the backlog is not searchable yet -- what comes back is the indexed prefix
            ids_before = [r.doc_id for r in search(h2, backlog[1]["vector"], 5)]
            @test !(backlog[1]["doc_id"] in ids_before)

            index!(h2)
            hit = search(h2, backlog[1]["vector"], 3)
            @test hit[1].doc_id == backlog[1]["doc_id"]
            @test hit[1].distance ≈ 0.0 atol=1e-6
            close_project!(h2)
        end
    end

    @testset "text dataset: a staged-but-not-yet-indexed backlog survives close + reopen" begin
        mktempworkdir() do workdir
            items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:20]]
            trained, backlog = items[1:15], items[16:20]

            h = create_project(workdir, "text_backlog_ds"; engine=FullTextEngine, backend=BM25InvertedFile, textmodel=FitFromCorpus())
            append_items!(h, text_items(trained))
            index!(h)
            append_items!(h, text_items(backlog)) # staged, deliberately left un-indexed
            close_project!(h)

            h2 = open_project(workdir, "text_backlog_ds")
            ids_before = Set(r.doc_id for r in ftsearch(h2, backlog[1]["text"], 100))
            @test !(backlog[1]["doc_id"] in ids_before)

            index!(h2)
            ids_after = Set(r.doc_id for r in ftsearch(h2, backlog[1]["text"], 100))
            @test backlog[1]["doc_id"] in ids_after

            close_project!(h2)
        end
    end

    @testset "persistence round-trip: close + reopen preserves search and tombstones" begin
        mktempworkdir() do workdir
            items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:30]]

            h = create_project(workdir, "roundtrip_ds")
            append_items!(h, dense_items(items))
            index!(h)
            before = search(h, items[1]["vector"], 5)
            delete_item!(h, 2)
            close_project!(h)

            h2 = open_project(workdir, "roundtrip_ds")
            after = search(h2, items[1]["vector"], 5)
            @test [r.doc_id for r in before if r._id != 2] == [r.doc_id for r in after]
            @test !(2 in [r._id for r in after])

            e = exists(h2, ["frankenstein_2"])
            @test e[1].exists && e[1].deleted

            close_project!(h2)
        end
    end

    @testset "read_only open avoids the write lock; a second writer collides" begin
        mktempworkdir() do workdir
            items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:10]]

            h = create_project(workdir, "lock_ds")
            append_items!(h, dense_items(items))
            index!(h)

            h_ro = open_project(workdir, "lock_ds"; read_only=true)
            @test length(search(h_ro, items[1]["vector"], 3)) == 3
            close_project!(h_ro)

            @test_throws RocksDB.RocksDBException open_project(workdir, "lock_ds")

            close_project!(h)
        end
    end


    @testset "project creation: the text model is stated explicitly, or not at all" begin
        mktempworkdir() do workdir
            # omitting it on a text project is an error rather than a silent default, because
            # the default it would have to pick freezes the vocabulary at the first index! call
            @test_throws ErrorException create_project(workdir, "no_model"; engine=FullTextEngine, backend=BM25InvertedFile)

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
            @test_throws ErrorException create_project(workdir, "dense_model"; engine=DenseEngine, backend=SearchGraph,
                                                       textmodel=FitFromCorpus())
            @test !isdir(joinpath(workdir, "dense_model"))
        end
    end

    @testset "text project: FitFromCorpus options drive the fit" begin
        mktempworkdir() do workdir
            # "comun" is in every document (a stopword by frequency); "singularidad" and
            # "irrepetible" are in exactly one (hapaxes, what a frequency floor is for)
            corpus = vcat(["comun perro claro $i" for i in 1:10],
                          ["comun gato claro $i" for i in 11:20],
                          ["comun perro singularidad irrepetible"])
            items = [TextItem(corpus[i]; doc_id="d$i") for i in eachindex(corpus)]

            function fitted(name, spec)
                h = create_project(workdir, name; engine=FullTextEngine, backend=TextInvertedFile, textmodel=spec)
                append_items!(h, items)
                index!(h)
                h
            end

            hb = fitted("fit_base", FitFromCorpus())
            base_vocsize = vocsize(text_profile(hb).model.voc)
            @test !isempty(ftsearch(hb, "singularidad", 3))
            close_project!(hb)

            # min_ndocs drops the hapaxes, which makes them unsearchable
            hp = fitted("fit_pruned", FitFromCorpus(; min_ndocs=2))
            @test vocsize(text_profile(hp).model.voc) < base_vocsize
            @test isempty(ftsearch(hp, "singularidad", 3))
            @test !isempty(ftsearch(hp, "perro", 3))
            close_project!(hp)

            # the library's own option groups pass straight through, and the lineage proves
            # which ones the fit actually ran under
            he = fitted("fit_encoder", FitFromCorpus(; encoder=(; outdim=16)))
            @test occursin("outdim=16", lineage_summary(text_profile(he)))
            close_project!(he)

            # max_documents caps what the fit reads: the sample, not the corpus, is the
            # trainsize -- which is the whole point, and also its cost
            hc = fitted("fit_capped", FitFromCorpus(; max_documents=5))
            @test gettrainsize(text_profile(hc).model.voc) == 5
            close_project!(hc)
            hf = fitted("fit_uncapped", FitFromCorpus(; max_documents=0))
            @test gettrainsize(text_profile(hf).model.voc) == length(items)
            close_project!(hf)

            # stopword detection: flagged, applied, and gone from the rebuilt vocabulary --
            # which is what the second pass over the corpus buys
            # 0.9, not 0.5: "perro" is in 11 of the 21 documents (0.52), and a threshold that
            # flags a content word as a stopword empties every query for it
            hs = fitted("fit_stopwords", FitFromCorpus(; stopwords=0.9))
            ps = text_profile(hs)
            @test ps.applied.stopwords
            @test "comun" in ps.stopwords
            @test token2id(ps.model.voc, "comun") == 0
            @test isempty(ftsearch(hs, "comun", 3))     # dropped on the query side too
            @test !isempty(ftsearch(hs, "perro", 3))
            close_project!(hs)
        end
    end

    @testset "text project: a custom TextConfig is fitted under, persisted, and restored" begin
        mktempworkdir() do workdir
            h = create_project(workdir, "cfg_ds"; engine=FullTextEngine, backend=BM25InvertedFile, textmodel=FitFromCorpus(CASED_ES))
            append_items!(h, ACCENT_ITEMS)
            index!(h)

            profile = text_profile(h)
            @test profile !== nothing
            @test gettextconfig(profile).language === :es
            @test gettextconfig(profile).normalization.lc == false
            # the lineage is written by the library's fit, since that is what runs: this
            # package delegates the whole job rather than recording its own step
            @test occursin("fit(", lineage_summary(profile))
            @test occursin("encoder=lsi", lineage_summary(profile))
            # delegating means the full pipeline, so a fitted profile carries the artifacts a
            # bare vocabulary-and-weights pass could not produce
            @test !isempty(profile.query_expansion)
            close_project!(h)

            h2 = open_project(workdir, "cfg_ds")
            @test gettextconfig(text_profile(h2)).language === :es
            @test gettextconfig(text_profile(h2)).normalization.del_diac == false
            close_project!(h2)
        end
    end

    @testset "DefaultProfile resolves through the installed profile library" begin
        # An unknown language is refused at construction, naming the ones that are known --
        # there is nothing to look up and no point deferring the error to index! time.
        @test_throws ArgumentError DefaultProfile(:xx)
        for lang in (:en, :es, :eu, :fr, :it, :pt, :ru)
            @test haskey(DEFAULT_PROFILE_NICKNAMES, lang)
            @test DefaultProfile(lang).nickname == DEFAULT_PROFILE_NICKNAMES[lang]
        end
        # a nickname overrides the per-language default, for a refit of your own or another snapshot
        @test DefaultProfile(:es; nickname="mine").nickname == "mine"

        # Re-exported remote profile functions
        @test isdefined(SimilaritySearchEngine, :download_profile)
        @test isdefined(SimilaritySearchEngine, :list_remote_profiles)

        # Resolution is by convention over $TEXTSEARCH_HOME, and a missing profile is an error
        # carrying the command that installs it -- "no such file" under ~/.textsearch is not
        # actionable to someone who has never run the CLI.
        mktempworkdir() do home
            withenv("TEXTSEARCH_HOME" => home) do
                spec = DefaultProfile(:es)
                @test SimilaritySearchEngine.IndexEngine.textsearch_home() == home
                msg = try
                    default_profile_path(spec); ""
                catch e
                    sprint(showerror, e)
                end
                @test occursin("not installed", msg)
                @test occursin("download_profile", msg)
                @test occursin("textsearch", msg)
                @test occursin(spec.nickname, msg)

                # it resolves if the legacy paragraph nickname is present
                mkpath(joinpath(home, "profiles"))
                touch(joinpath(home, "profiles", "wiki20231101-es-paragraphs.zip"))
                @test default_profile_path(spec) == joinpath(home, "profiles", "wiki20231101-es-paragraphs.zip")

                # and the official nickname takes priority once present
                touch(joinpath(home, "profiles", spec.nickname * ".zip"))
                @test default_profile_path(spec) == joinpath(home, "profiles", spec.nickname * ".zip")
            end
        end
    end

    @testset "text project: QueryPolicy corrects a query, :off searches it literally" begin
        mktempworkdir() do workdir
            h = create_project(workdir, "policy_ds"; engine=FullTextEngine, backend=BM25InvertedFile, textmodel=FitFromCorpus(CASED_ES))
            append_items!(h, ACCENT_ITEMS)
            index!(h)

            # :auto (the default) reads the bare spelling as wrong and replaces it, so the
            # accented documents are what comes back -- and never the literal one.
            corrected = ftsearch(h, "musica", 5)
            @test length(corrected) == 5
            @test all(r -> startswith(r.doc_id, "acc_"), corrected)

            # correcting is a substitution the caller can see, and undo
            lines = ftexplain(h, "musica")
            @test length(lines) == 1
            @test occursin("m\u00fasica", lines[1])

            literal = ftsearch(h, "musica", 5; policy=QueryPolicy(correction=:off))
            @test [r.doc_id for r in literal] == ["bare"]
            @test isempty(ftexplain(h, "musica"; policy=QueryPolicy(correction=:off)))

            # a well-typed query is left alone under :auto
            @test isempty(ftexplain(h, "m\u00fasica"))
            close_project!(h)

            # the variant map is derived from the restored vocabulary, so correction survives a
            # reopen without anything about it having been persisted
            h2 = open_project(workdir, "policy_ds")
            @test all(r -> startswith(r.doc_id, "acc_"), ftsearch(h2, "musica", 5))
            close_project!(h2)
        end
    end

    @testset "text project: a pre-fitted profile fixes the out-of-vocabulary limitation" begin
        mktempworkdir() do workdir
            corpus = vcat(["documento comun numero $i sobre temas generales" for i in 1:10],
                          ["aparicion tardia del termino zeppelin en el corpus $i" for i in 11:20])
            items = [TextItem(corpus[i]; doc_id="d$i") for i in eachindex(corpus)]
            first_batch, second_batch = items[1:10], items[11:20]

            # fitted over the WHOLE corpus and round-tripped through the on-disk format, the way
            # a real project would consume one of TextSearch.jl's published corpus profiles
            profile_dir = joinpath(workdir, "profile")
            save_profile(profile_dir, fit_profile(FitFromCorpus(), corpus))
            reloaded = load_profile(profile_dir)

            # deferred fit: the vocabulary is frozen over the first batch, so a term that only
            # ever appears in the second is out-of-vocabulary forever after
            h = create_project(workdir, "oov_ds"; engine=FullTextEngine, backend=BM25InvertedFile, textmodel=FitFromCorpus())
            append_items!(h, first_batch); index!(h)
            append_items!(h, second_batch); index!(h)
            @test isempty(ftsearch(h, "zeppelin", 5))
            close_project!(h)

            # same insertion sequence against the pre-fitted profile: trained before the first
            # item was staged, so the second batch's terms are searchable
            hp = create_project(workdir, "prefit_ds"; engine=FullTextEngine, backend=BM25InvertedFile, textmodel=BaseProfile(reloaded))
            @test text_profile(hp) !== nothing          # trained at creation, not at first index!
            append_items!(hp, first_batch); index!(hp)
            append_items!(hp, second_batch); index!(hp)
            hits = ftsearch(hp, "zeppelin", 5)
            @test !isempty(hits)
            @test all(r -> parse(Int, r.doc_id[2:end]) >= 11, hits)
            @test vocsize(text_profile(hp).model.voc) == vocsize(reloaded.model.voc)
            close_project!(hp)
        end
    end

    @testset "text project: a set distance encodes queries as bags, like its documents" begin
        mktempworkdir() do workdir
            corpus = vcat(["el perro ladra en el patio $i" for i in 1:10],
                          ["la bicicleta oxidada del vecino $i" for i in 11:20])
            items = [TextItem(corpus[i]; doc_id="d$i") for i in eachindex(corpus)]

            # TextInvertedFile indexes bags rather than weighted vectors under a set distance,
            # so a query has to be encoded the same way or the two sides stop being comparable
            for dist in (Dist.Sets.Jaccard(), Dist.Sets.Dice())
                name = "sets_$(nameof(typeof(dist)))"
                h = create_project(workdir, name; engine=FullTextEngine, backend=TextInvertedFile, distance=dist, textmodel=FitFromCorpus())
                append_items!(h, items)
                index!(h)
                hits = [r.doc_id for r in ftsearch(h, "perro patio", 3)]
                @test length(hits) == 3
                @test all(id -> parse(Int, id[2:end]) <= 10, hits)
                close_project!(h)

                h2 = open_project(workdir, name)
                @test [r.doc_id for r in ftsearch(h2, "perro patio", 3)] == hits
                close_project!(h2)
            end
        end
    end

    @testset "text project: QueryPolicy drives query expansion per call" begin
        mktempworkdir() do workdir
            # three disjoint topics of ten documents each, so which topic comes back says
            # exactly which query tokens were searched
            corpus = vcat(["el perro ladra en el patio $i" for i in 1:10],
                          ["el gato duerme en el sofa $i" for i in 11:20],
                          ["la bicicleta oxidada del vecino $i" for i in 21:30])
            items = [TextItem(corpus[i]; doc_id="d$i") for i in eachindex(corpus)]

            # a profile carrying (and applying) an expansion network -- the artifact a fit over
            # an indexing corpus cannot produce, and the reason a project takes a whole profile
            base = fit_profile(FitFromCorpus(), corpus)
            withnet = TextProfile(base.model;
                query_expansion=Dict("perro" => ["gato", "bicicleta"]),
                query_expansion_distances=Dict("perro" => Float32[0.2, 0.9]),
                applied=AppliedArtifacts(query_expansion=true))

            h = create_project(workdir, "expand_ds"; engine=FullTextEngine, backend=BM25InvertedFile, textmodel=BaseProfile(withnet))
            append_items!(h, items)
            index!(h)

            topic(hits) = unique(map(r -> parse(Int, r.doc_id[2:end]) <= 10 ? :perro :
                                          parse(Int, r.doc_id[2:end]) <= 20 ? :gato : :bici, hits))

            # expansion off: only what was typed, so only the "perro" documents
            @test topic(ftsearch(h, "perro", 6; policy=QueryPolicy(expansion=false))) == [:perro]
            # expansion_k=1 takes only the nearest neighbour ("gato"), which does not outrank
            # the typed term's own documents
            @test topic(ftsearch(h, "perro", 6; policy=QueryPolicy(expansion_k=1))) == [:perro]
            # the whole network widens the query far enough to change the ranking outright
            @test topic(ftsearch(h, "perro", 6)) == [:bici]

            close_project!(h)
        end
    end

    @testset "text project: TextInvertedFile selects the weighted engine and round-trips" begin
        mktempworkdir() do workdir
            items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:20]]

            h = create_project(workdir, "tif_ds"; engine=FullTextEngine, backend=TextInvertedFile, textmodel=FitFromCorpus())
            append_items!(h, text_items(items))
            index!(h)
            @test h.engine isa SimilaritySearchEngine.IndexEngine.FullTextEngine
            @test h.engine.backend.index isa TextInvertedFile
            before = [r.doc_id for r in ftsearch(h, items[3]["text"], 3)]
            @test before[1] == "frankenstein_3"
            close_project!(h)

            h2 = open_project(workdir, "tif_ds")
            @test h2.engine.backend.index isa TextInvertedFile
            @test [r.doc_id for r in ftsearch(h2, items[3]["text"], 3)] == before
            close_project!(h2)
        end
    end


    @testset "doc_id resolves through an index, and a shared doc_id returns every item" begin
        mktempworkdir() do workdir
            items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:30]]
            h = create_project(workdir, "docid_ds"; engine=DenseEngine, backend=SearchGraph)
            append_items!(h, dense_items(items))
            index!(h)

            # Exact match: "frankenstein_2" is not "frankenstein_20", even though the index
            # keys of the latter start with the former's bytes.
            one = fetch_items(h, ["frankenstein_2"])
            @test length(one) == 1 && one[1].doc_id == "frankenstein_2"

            # Nothing requires doc_id to be unique, so a repeated one brings back every item
            # carrying it, ascending by _id -- the old scan returned whichever it reached first.
            append_items!(h, [DenseItem(Float32.(items[5]["vector"]); doc_id="frankenstein_2")])
            index!(h)
            twice = fetch_items(h, ["frankenstein_2"])
            @test [it._id for it in twice] == Int32[2, 31]

            # A numeric argument still means the internal _id, not a doc_id.
            @test [it._id for it in fetch_items(h, [3])] == Int32[3]

            @test isempty(fetch_items(h, ["nope"]))
            @test exists(h, ["nope"])[1].exists == false

            # exists() summarizes the matches: found if any, deleted only when all of them are.
            @test exists(h, ["frankenstein_2"])[1].exists
            delete_item!(h, 2)
            @test exists(h, ["frankenstein_2"])[1].deleted == false   # _id 31 is still live
            delete_item!(h, 31)
            @test exists(h, ["frankenstein_2"])[1].deleted == true
            close_project!(h)
        end
    end

    @testset "a project written before the doc_id index gets it backfilled on the next open" begin
        mktempworkdir() do workdir
            P = SimilaritySearchEngine.Project
            items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:10]]
            h = create_project(workdir, "backfill_ds"; engine=DenseEngine, backend=SearchGraph)
            append_items!(h, dense_items(items))
            index!(h)

            # Wipe the column family down to what a project written before it existed holds.
            keys_to_drop = [k for (k, _) in RocksDB.DBIterator(h.project.db; cf=h.project.cf_idx_docid)]
            for k in keys_to_drop
                delete!(h.project.db, k, cf=h.project.cf_idx_docid)
            end
            @test isempty([k for (k, _) in RocksDB.DBIterator(h.project.db; cf=h.project.cf_idx_docid)])
            close_project!(h)

            h2 = open_project(workdir, "backfill_ds")
            @test [it.doc_id for it in fetch_items(h2, ["frankenstein_4"])] == ["frankenstein_4"]
            # ... and it is a one-time migration: the marker stops the next open from rescanning.
            @test P.backfill_docid_index!(h2.project) == 0
            close_project!(h2)
        end
    end

    @testset "a filtered search spends the candidate budget it was given, and says so by returning fewer" begin
        mktempworkdir() do workdir
            # Exact backend and vectors ranked by construction: item i sits at distance i-1 from
            # the query, so "the item the predicate wants" has a known rank.
            h = create_project(workdir, "budget_ds"; engine=DenseEngine, backend=ExhaustiveSearch)
            append_items!(h, [DenseItem(Float32[i - 1, 0, 0, 0]; doc_id="v$i") for i in 1:64])
            q = Float32[0, 0, 0, 0]

            wants_the_farthest = (record, meta) -> record.doc_id == "v64"
            # Default budget is 2k = 2 candidates: the item ranked 64th is not among them.
            @test isempty(search(h, q, 1; filter=wants_the_farthest))
            # Paying for the whole collection finds it. This is the documented trade-off.
            @test [r.doc_id for r in search(h, q, 1; filter=wants_the_farthest, candidates=64)] == ["v64"]

            # A predicate most items satisfy is served by the default budget.
            @test length(search(h, q, 5; filter=(record, meta) -> true)) == 5
            close_project!(h)
        end
    end

    @testset "batch search answers exactly what one-at-a-time search answers" begin
        mktempworkdir() do workdir
            h = create_project(workdir, "batch_ds"; engine=DenseEngine, backend=ExhaustiveSearch)
            vectors = [Float32[i, i % 7, i % 3, 1] for i in 1:40]
            append_items!(h, [DenseItem(v; doc_id="b$i") for (i, v) in enumerate(vectors)])
            queries = vectors[1:2:20]

            batched = search(h, queries, 5)
            @test length(batched) == length(queries)
            for (q, hits) in zip(queries, batched)
                one = search(h, q, 5)
                @test [r._id for r in hits] == [r._id for r in one]
                @test [r.doc_id for r in hits] == [r.doc_id for r in one]
                @test [r.distance for r in hits] ≈ [r.distance for r in one]
            end

            ids, dists = searchbatch(h, queries, 5)
            @test size(ids) == (5, length(queries))
            @test size(dists) == (5, length(queries))
            @test all(>(0), ids)
            @test [Int32(ids[i, 1]) for i in 1:5] == [r._id for r in batched[1]]

            # A soft-deleted hit comes back flagged and without a doc_id, same as single search.
            delete_item!(h, Int(ids[1, 1]))
            flagged = search(h, queries[1:1], 5)[1][1]
            @test flagged.deleted && flagged.doc_id === nothing
            close_project!(h)
        end
    end

    @testset "batch search is a whole-dataset operation: no backlog, no text projects" begin
        mktempworkdir() do workdir
            h = create_project(workdir, "batchguard_ds"; engine=DenseEngine, backend=SearchGraph)
            vectors = [Float32[i, 1, 2, 3] for i in 1:30]
            append_items!(h, [DenseItem(v; doc_id="g$i") for (i, v) in enumerate(vectors)])
            index!(h)
            @test size(searchbatch(h, vectors[1:3], 4)[1]) == (4, 3)

            # Staged but not indexed: refused rather than read past the adjacency list.
            append_items!(h, [DenseItem(Float32[99, 1, 2, 3]; doc_id="g_new")])
            @test_throws ErrorException searchbatch(h, vectors[1:3], 4)
            index!(h)
            @test size(searchbatch(h, vectors[1:3], 4)[1]) == (4, 3)
            close_project!(h)

            ht = create_project(workdir, "batchtext_ds"; engine=FullTextEngine,
                                backend=BM25InvertedFile, textmodel=FitFromCorpus())
            append_items!(ht, [TextItem("some text number $i"; doc_id="t$i") for i in 1:20])
            index!(ht)
            @test_throws ErrorException searchbatch(ht, ["some text", "other text"], 3)
            close_project!(ht)
        end
    end

    @testset "compaction runs for a session that wrote and is skipped for one that did not" begin
        mktempworkdir() do workdir
            h = create_project(workdir, "compact_ds"; engine=DenseEngine, backend=SearchGraph)
            vectors = [Float32[i, 2, 3, 4] for i in 1:40]
            append_items!(h, [DenseItem(v; doc_id="c$i") for (i, v) in enumerate(vectors)])
            index!(h)

            @test compact_project!(h) == true       # this session wrote
            @test compact_project!(h) == false      # ... and nothing changed since
            @test compact_project!(h; force=true) == true
            before = [r._id for r in search(h, vectors[3], 5)]
            close_project!(h)

            h2 = open_project(workdir, "compact_ds")
            @test [r._id for r in search(h2, vectors[3], 5)] == before   # survived compaction
            @test compact_project!(h2) == false     # a read-only session in practice
            close_project!(h2; compact=false)

            h3 = open_project(workdir, "compact_ds"; read_only=true)
            @test compact_project!(h3; force=true) == false   # a read-only handle cannot write
            @test [r._id for r in search(h3, vectors[3], 5)] == before
            close_project!(h3)
        end
    end

end
