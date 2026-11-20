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

@heavy "text project: FitFromCorpus options drive the fit" begin
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

            # A file under a nickname from before format 1.1 is not a fallback: those
            # profiles are in format 1.0, which TextSearch 1.2 does not load.
            mkpath(joinpath(home, "profiles"))
            touch(joinpath(home, "profiles", "wiki20231101-es-paragraphs.zip"))
            @test_throws ProfileNotInstalled default_profile_path(spec)

            # it resolves once the official nickname is present
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

@testset "text project: edit_correction corrects a token the vocabulary lacks, only when asked" begin
    mktempworkdir() do workdir
        texts = vcat(["la guerra civil del periodo $i" for i in 1:5],
                     ["un tratado de paz firmado en $i" for i in 1:5])
        items = [TextItem(t; doc_id=(i <= 5 ? "g$i" : "p$i")) for (i, t) in enumerate(texts)]
        war(hits) = length(hits) == 5 && all(r -> startswith(r.doc_id, "g"), hits)

        # Off by default: `guerar` is not in the vocabulary, and no fold reaches `guerra`.
        h = create_project(workdir, "edits_bm25"; engine=FullTextEngine, backend=BM25InvertedFile,
                           textmodel=FitFromCorpus())
        append_items!(h, items)
        index!(h)
        @test isempty(ftsearch(h, "guerar", 5))
        @test isempty(ftexplain(h, "guerar"))
        close_project!(h)

        # Opening with it on assembles the persisted index with an edit index.
        h = open_project(workdir, "edits_bm25"; edit_correction=true)
        @test war(ftsearch(h, "guerar", 5))
        lines = ftexplain(h, "guerar")
        @test length(lines) == 1
        @test occursin("guerra (edit)", lines[1])
        # The policy still decides per query.
        @test isempty(ftsearch(h, "guerar", 5; policy=QueryPolicy(correction=:off)))
        # A token the vocabulary holds is never replaced by a guess.
        @test isempty(ftexplain(h, "guerra"))
        close_project!(h)

        # The value is not persisted: a plain open is off again.
        h = open_project(workdir, "edits_bm25")
        @test isempty(ftsearch(h, "guerar", 5))
        close_project!(h)

        # Set at creation, it reaches the index that the first index! builds, on the weighted
        # backend too.
        h = create_project(workdir, "edits_weighted"; engine=FullTextEngine, backend=TextInvertedFile,
                           textmodel=FitFromCorpus(), edit_correction=true)
        append_items!(h, items)
        index!(h)
        @test war(ftsearch(h, "guerar", 5))
        close_project!(h)

        # A BaseProfile builds its index at creation, which is the third place it applies.
        profile = fit_profile(FitFromCorpus(), texts)
        h = create_project(workdir, "edits_base"; engine=FullTextEngine, backend=BM25InvertedFile,
                           textmodel=BaseProfile(profile), edit_correction=true)
        append_items!(h, items)
        index!(h)
        @test war(ftsearch(h, "guerar", 5))
        close_project!(h)

        # A project without text has no tokens to correct.
        @test_throws InvalidOption create_project(workdir, "edits_dense"; edit_correction=true)
        close_project!(create_project(workdir, "dense_ok"; engine=DenseEngine, backend=ExhaustiveSearch))
        @test_throws InvalidOption open_project(workdir, "dense_ok"; edit_correction=true)
        # The refused open released the project's lock.
        close_project!(open_project(workdir, "dense_ok"))
    end
end

@heavy "text project: a pre-fitted profile fixes the out-of-vocabulary limitation" begin
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

@heavy "text project: a set distance encodes queries as bags, like its documents" begin
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

@testset "a text project reports what its vocabulary does not cover" begin
    mktempworkdir() do workdir
        h = create_project(workdir, "oov_ds"; engine=FullTextEngine, backend=BM25InvertedFile,
                           textmodel=FitFromCorpus())
        append_items!(h, [TextItem(t; doc_id="d$i") for (i, t) in enumerate([
            "el gato duerme sobre el tejado",
            "un perro ladra en la calle",
            "el gato y el perro se miran"])])
        index!(h)

        report = vocabulary_report(h)
        @test report["vocsize"] > 0
        @test report["trainsize"] == 3
        @test report["queries"] == 0
        @test report["query_oov_rate"] == 0.0     # nothing asked yet, nothing missing yet
        @test length(report["top_tokens"]) > 0

        # Two words the corpus has, two it does not
        ftsearch(h, "gato tejado", 3)
        ftsearch(h, "criptomoneda blockchain", 3)
        report = vocabulary_report(h)
        @test report["queries"] == 2
        @test report["query_tokens"] == 4
        @test report["query_oov_tokens"] == 2
        @test report["query_oov_rate"] == 0.5

        # The documents themselves are covered: this vocabulary was fitted from them, and
        # that is the rate a rebuild would act on
        scanned = vocabulary_report(h; scan=true)
        @test scanned["live_oov_rate"] == 0.0
        @test scanned["live_tokens"] > 0
        @test scanned["live_documents_scanned"] == 3
        @test scanned["live_documents_total"] == 3
        @test scanned["live_sampled"] == false          # three documents fit in any sample

        # The pass over the documents is bounded: with a sample of one, one document is
        # read, the report says so, and the rate is what that document shows
        partial = vocabulary_report(h; scan=true, sample=1)
        @test partial["live_documents_scanned"] == 1
        @test partial["live_documents_total"] == 3
        @test partial["live_sampled"] == true
        @test partial["live_tokens"] < scanned["live_tokens"]
        # ... and `sample=0` reads every document, which is what `describe` asks for
        @test vocabulary_report(h; scan=true, sample=0)["live_documents_scanned"] == 3

        close_project!(h)
        h2 = open_project(workdir, "oov_ds")
        # The counters describe the project, so they are still there after it is reopened
        @test vocabulary_report(h2)["queries"] == 2
        @test vocabulary_report(h2)["query_oov_tokens"] == 2
        close_project!(h2)
    end
end

@testset "vocabulary_report is for text projects with a vocabulary" begin
    mktempworkdir() do workdir
        dense = create_project(workdir, "oov_dense"; engine=DenseEngine, backend=ExhaustiveSearch)
        @test vocabulary_report(dense) === nothing
        close_project!(dense)

        # A text project that has never been indexed has no vocabulary to report on
        fresh = create_project(workdir, "oov_fresh"; engine=FullTextEngine,
                               backend=BM25InvertedFile, textmodel=FitFromCorpus())
        @test vocabulary_report(fresh) === nothing
        close_project!(fresh)
    end
end
