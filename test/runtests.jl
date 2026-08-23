using Test
using SimilaritySearchEngine
using SimilaritySearch: SearchGraph, Dist
using TextSearch: BM25InvertedFile, InvertedFile, TextInvertedFile, NormalizationConfig,
                  AppliedArtifacts, FreqWeighting, BinaryGlobalWeighting,
                  vocsize, gettextconfig, lineage_summary, token2id
using JSON
using RocksDB

const FRANKENSTEIN_PATH = joinpath(@__DIR__, "data", "frankenstein.jsonl")

function mktempworkdir(f)
    dir = mktempdir()
    try
        f(dir)
    finally
        rm(dir; recursive=true, force=true)
    end
end

# A text policy that keeps case and diacritics, which is what makes orthographic bridging
# possible at all: `derive_variants` has nothing to do under the default `TextConfig()`, whose
# normalization already folds both.
const CASED_ES = TextConfig(normalization=NormalizationConfig(lc=false, del_diac=false), language=:es)

# 60 documents spelling it "musica" with an accent against a single one without: far enough
# apart for `QueryPolicy`'s default `negligible_ratio=50` to read the bare spelling as wrong.
const ACCENT_ITEMS = vcat(
    [Dict("doc_id" => "acc_$i", "text" => "la m\u00fasica cl\u00e1sica de la ciudad $i") for i in 1:60],
    [Dict("doc_id" => "bare", "text" => "una musica rara sin acento")])

@testset "SimilaritySearchEngine.jl" begin

    @testset "dense dataset: append (stage), index!, search, calibrate, allknn, delete, fetch" begin
        mktempworkdir() do workdir
            items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:100]]

            h = create_project(workdir, "dense_ds")
            inserted = append_items!(h, items)
            @test inserted == 100

            # append_items! only stages a SearchGraph's vectors -- nothing is searchable
            # until an explicit index! call catches up the graph-connection backlog.
            @test isempty(search(h, items[1]["vector"], 5))

            index!(h)

            res = search(h, items[1]["vector"], 5)
            @test length(res) == 5
            @test res[1].id == "frankenstein_1"
            @test res[1].doc_id == 1
            @test res[1].distance ≈ 0.0 atol=1e-6

            # calibrate! returns a minrecall::Float32 => BeamSearch table, not a lone BeamSearch.
            bs = calibrate!(h; numqueries=20)
            @test !isempty(bs)
            @test all(v -> v.bsize isa Integer, values(bs))

            ak = allknn(h; k=5)
            @test length(ak) == 100
            @test ak[1].id == 1
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
            deleted_idx = findfirst(r -> r.doc_id == 2, res_after_delete)
            @test deleted_idx !== nothing
            @test res_after_delete[deleted_idx].deleted

            fetched = fetch_items(h, ["frankenstein_3"])
            @test length(fetched) == 1
            @test fetched[1]["doc_id"] == "frankenstein_3"

            filtered = search(h, items[1]["vector"], 5; filter=(record, meta) -> record.doc_id == "frankenstein_1")
            @test length(filtered) == 1
            @test filtered[1].id == "frankenstein_1"

            close_project!(h)
        end
    end

    @testset "text (bm25) dataset: stage, index! (trains + catches up), ftsearch" begin
        mktempworkdir() do workdir
            items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:50]]
            first_batch, second_batch = items[1:30], items[31:50]

            h = create_project(workdir, "text_ds"; index_type=BM25InvertedFile, textmodel=FitFromCorpus())
            inserted = append_items!(h, first_batch)
            @test inserted == 30

            # staged raw text, not yet trained/encoded -- nothing searchable yet.
            @test isempty(ftsearch(h, first_batch[1]["text"], 3))

            # first index! call trains the Vocabulary from everything staged so far, and
            # encodes/indexes the same backlog.
            index!(h)
            res = ftsearch(h, first_batch[1]["text"], 3)
            @test length(res) == 3
            @test res[1].id == "frankenstein_1"

            # idempotent: nothing new staged, safe to call again.
            index!(h)
            @test length(ftsearch(h, first_batch[1]["text"], 3)) == 3

            # a second batch is staged but stays un-indexed until the next index! call.
            append_items!(h, second_batch)
            ids_before = Set(r.id for r in ftsearch(h, second_batch[1]["text"], 100))
            @test !(second_batch[1]["doc_id"] in ids_before)

            index!(h)
            ids_after = Set(r.id for r in ftsearch(h, second_batch[1]["text"], 100))
            @test second_batch[1]["doc_id"] in ids_after

            close_project!(h)
        end
    end

    @testset "text (weighted InvertedFile) dataset: stage, index!, ftsearch" begin
        mktempworkdir() do workdir
            items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:20]]
            first_batch, second_batch = items[1:10], items[11:20]

            h = create_project(workdir, "invfile_ds"; index_type=InvertedFile, textmodel=FitFromCorpus())
            append_items!(h, first_batch)
            @test isempty(ftsearch(h, first_batch[2]["text"], 3))

            index!(h)
            @test length(ftsearch(h, first_batch[2]["text"], 3)) == 3

            append_items!(h, second_batch)
            ids_before = Set(r.id for r in ftsearch(h, second_batch[1]["text"], 100))
            @test !(second_batch[1]["doc_id"] in ids_before)

            index!(h)
            ids_after = Set(r.id for r in ftsearch(h, second_batch[1]["text"], 100))
            @test second_batch[1]["doc_id"] in ids_after

            close_project!(h)
        end
    end

    @testset "text dataset: a staged-but-not-yet-indexed backlog survives close + reopen" begin
        mktempworkdir() do workdir
            items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:20]]
            trained, backlog = items[1:15], items[16:20]

            h = create_project(workdir, "text_backlog_ds"; index_type=BM25InvertedFile, textmodel=FitFromCorpus())
            append_items!(h, trained)
            index!(h)
            append_items!(h, backlog) # staged, deliberately left un-indexed
            close_project!(h)

            h2 = open_project(workdir, "text_backlog_ds")
            ids_before = Set(r.id for r in ftsearch(h2, backlog[1]["text"], 100))
            @test !(backlog[1]["doc_id"] in ids_before)

            index!(h2)
            ids_after = Set(r.id for r in ftsearch(h2, backlog[1]["text"], 100))
            @test backlog[1]["doc_id"] in ids_after

            close_project!(h2)
        end
    end

    @testset "persistence round-trip: close + reopen preserves search and tombstones" begin
        mktempworkdir() do workdir
            items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:30]]

            h = create_project(workdir, "roundtrip_ds")
            append_items!(h, items)
            index!(h)
            before = search(h, items[1]["vector"], 5)
            delete_item!(h, 2)
            close_project!(h)

            h2 = open_project(workdir, "roundtrip_ds")
            after = search(h2, items[1]["vector"], 5)
            @test [r.id for r in before if r.doc_id != 2] == [r.id for r in after]
            @test !(2 in [r.doc_id for r in after])

            e = exists(h2, ["frankenstein_2"])
            @test e[1].exists && e[1].deleted

            close_project!(h2)
        end
    end

    @testset "read_only open avoids the write lock; a second writer collides" begin
        mktempworkdir() do workdir
            items = [JSON.parse(l) for l in readlines(FRANKENSTEIN_PATH)[1:10]]

            h = create_project(workdir, "lock_ds")
            append_items!(h, items)
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
            @test_throws ErrorException create_project(workdir, "no_model"; index_type=BM25InvertedFile)

            msg = try
                create_project(workdir, "no_model"; index_type=BM25InvertedFile)
                ""
            catch e
                sprint(showerror, e)
            end
            @test occursin("BaseProfile", msg) && occursin("FitFromCorpus", msg)

            # ... and the failed call left nothing behind: no directory, no held write lock,
            # so the same name is still free to create properly
            @test !isdir(joinpath(workdir, "no_model"))
            h = create_project(workdir, "no_model"; index_type=BM25InvertedFile, textmodel=FitFromCorpus())
            close_project!(h)

            # handing a text model to a dense project is an error too, not an ignored keyword
            @test_throws ErrorException create_project(workdir, "dense_model"; index_type=SearchGraph,
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
            items = [Dict("doc_id" => "d$i", "text" => corpus[i]) for i in eachindex(corpus)]

            function fitted(name, spec)
                h = create_project(workdir, name; index_type=TextInvertedFile, textmodel=spec)
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

            # the weighting scheme reaches the model, and survives a reopen
            hw = fitted("fit_weights", FitFromCorpus(; local_weighting=FreqWeighting(),
                                                       global_weighting=BinaryGlobalWeighting()))
            @test text_profile(hw).model.local_weighting isa FreqWeighting
            @test text_profile(hw).model.global_weighting isa BinaryGlobalWeighting
            close_project!(hw)
            hw2 = open_project(workdir, "fit_weights")
            @test text_profile(hw2).model.local_weighting isa FreqWeighting
            @test text_profile(hw2).model.global_weighting isa BinaryGlobalWeighting
            close_project!(hw2)

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
            h = create_project(workdir, "cfg_ds"; index_type=BM25InvertedFile, textmodel=FitFromCorpus(CASED_ES))
            append_items!(h, ACCENT_ITEMS)
            index!(h)

            profile = text_profile(h)
            @test profile !== nothing
            @test gettextconfig(profile).language === :es
            @test gettextconfig(profile).normalization.lc == false
            # the deferred fit records where it came from, so a profile saved out of a project
            # can be told apart from one fitted deliberately over a chosen corpus
            @test occursin("fit(", lineage_summary(profile))
            @test occursin("source=staged", lineage_summary(profile))
            close_project!(h)

            h2 = open_project(workdir, "cfg_ds")
            @test gettextconfig(text_profile(h2)).language === :es
            @test gettextconfig(text_profile(h2)).normalization.del_diac == false
            close_project!(h2)
        end
    end

    @testset "text project: QueryPolicy corrects a query, :off searches it literally" begin
        mktempworkdir() do workdir
            h = create_project(workdir, "policy_ds"; index_type=BM25InvertedFile, textmodel=FitFromCorpus(CASED_ES))
            append_items!(h, ACCENT_ITEMS)
            index!(h)

            # :auto (the default) reads the bare spelling as wrong and replaces it, so the
            # accented documents are what comes back -- and never the literal one.
            corrected = ftsearch(h, "musica", 5)
            @test length(corrected) == 5
            @test all(r -> startswith(r.id, "acc_"), corrected)

            # correcting is a substitution the caller can see, and undo
            lines = ftexplain(h, "musica")
            @test length(lines) == 1
            @test occursin("m\u00fasica", lines[1])

            literal = ftsearch(h, "musica", 5; policy=QueryPolicy(correction=:off))
            @test [r.id for r in literal] == ["bare"]
            @test isempty(ftexplain(h, "musica"; policy=QueryPolicy(correction=:off)))

            # a well-typed query is left alone under :auto
            @test isempty(ftexplain(h, "m\u00fasica"))
            close_project!(h)

            # the variant map is derived from the restored vocabulary, so correction survives a
            # reopen without anything about it having been persisted
            h2 = open_project(workdir, "policy_ds")
            @test all(r -> startswith(r.id, "acc_"), ftsearch(h2, "musica", 5))
            close_project!(h2)
        end
    end

    @testset "text project: a pre-fitted profile fixes the out-of-vocabulary limitation" begin
        mktempworkdir() do workdir
            corpus = vcat(["documento comun numero $i sobre temas generales" for i in 1:10],
                          ["aparicion tardia del termino zeppelin en el corpus $i" for i in 11:20])
            items = [Dict("doc_id" => "d$i", "text" => corpus[i]) for i in eachindex(corpus)]
            first_batch, second_batch = items[1:10], items[11:20]

            # fitted over the WHOLE corpus and round-tripped through the on-disk format, the way
            # a real project would consume one of TextSearch.jl's published corpus profiles
            profile_dir = joinpath(workdir, "profile")
            save_profile(profile_dir, fit_profile(FitFromCorpus(), corpus))
            reloaded = load_profile(profile_dir)

            # deferred fit: the vocabulary is frozen over the first batch, so a term that only
            # ever appears in the second is out-of-vocabulary forever after
            h = create_project(workdir, "oov_ds"; index_type=BM25InvertedFile, textmodel=FitFromCorpus())
            append_items!(h, first_batch); index!(h)
            append_items!(h, second_batch); index!(h)
            @test isempty(ftsearch(h, "zeppelin", 5))
            close_project!(h)

            # same insertion sequence against the pre-fitted profile: trained before the first
            # item was staged, so the second batch's terms are searchable
            hp = create_project(workdir, "prefit_ds"; index_type=BM25InvertedFile, textmodel=BaseProfile(reloaded))
            @test text_profile(hp) !== nothing          # trained at creation, not at first index!
            append_items!(hp, first_batch); index!(hp)
            append_items!(hp, second_batch); index!(hp)
            hits = ftsearch(hp, "zeppelin", 5)
            @test !isempty(hits)
            @test all(r -> parse(Int, r.id[2:end]) >= 11, hits)
            @test vocsize(text_profile(hp).model.voc) == vocsize(reloaded.model.voc)
            close_project!(hp)
        end
    end

    @testset "text project: a set distance encodes queries as bags, like its documents" begin
        mktempworkdir() do workdir
            corpus = vcat(["el perro ladra en el patio $i" for i in 1:10],
                          ["la bicicleta oxidada del vecino $i" for i in 11:20])
            items = [Dict("doc_id" => "d$i", "text" => corpus[i]) for i in eachindex(corpus)]

            # TextInvertedFile indexes bags rather than weighted vectors under a set distance,
            # so a query has to be encoded the same way or the two sides stop being comparable
            for dist in (Dist.Sets.Jaccard(), Dist.Sets.Dice())
                name = "sets_$(nameof(typeof(dist)))"
                h = create_project(workdir, name; index_type=TextInvertedFile, distance=dist, textmodel=FitFromCorpus())
                append_items!(h, items)
                index!(h)
                hits = [r.id for r in ftsearch(h, "perro patio", 3)]
                @test length(hits) == 3
                @test all(id -> parse(Int, id[2:end]) <= 10, hits)
                close_project!(h)

                h2 = open_project(workdir, name)
                @test [r.id for r in ftsearch(h2, "perro patio", 3)] == hits
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
            items = [Dict("doc_id" => "d$i", "text" => corpus[i]) for i in eachindex(corpus)]

            # a profile carrying (and applying) an expansion network -- the artifact a fit over
            # an indexing corpus cannot produce, and the reason a project takes a whole profile
            base = fit_profile(FitFromCorpus(), corpus)
            withnet = TextProfile(base.model;
                query_expansion=Dict("perro" => ["gato", "bicicleta"]),
                query_expansion_distances=Dict("perro" => Float32[0.2, 0.9]),
                applied=AppliedArtifacts(query_expansion=true))

            h = create_project(workdir, "expand_ds"; index_type=BM25InvertedFile, textmodel=BaseProfile(withnet))
            append_items!(h, items)
            index!(h)

            topic(hits) = unique(map(r -> parse(Int, r.id[2:end]) <= 10 ? :perro :
                                          parse(Int, r.id[2:end]) <= 20 ? :gato : :bici, hits))

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

            h = create_project(workdir, "tif_ds"; index_type=TextInvertedFile, textmodel=FitFromCorpus())
            append_items!(h, items)
            index!(h)
            @test h.engine isa SimilaritySearchEngine.IndexEngine.InvertedFileEngine
            @test h.engine.index isa TextInvertedFile
            before = [r.id for r in ftsearch(h, items[3]["text"], 3)]
            @test before[1] == "frankenstein_3"
            close_project!(h)

            h2 = open_project(workdir, "tif_ds")
            @test h2.engine.index isa TextInvertedFile
            @test [r.id for r in ftsearch(h2, items[3]["text"], 3)] == before
            close_project!(h2)
        end
    end

end
