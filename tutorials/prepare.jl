#!/usr/bin/env julia
#
# Builds everything the paragraph-search tutorial reads: four paragraph corpora, a fitted
# TextSearch profile for each, and a SimilaritySearchEngine project for each.
#
#   julia --project=tutorials tutorials/prepare.jl [step...] [--smoke] [--force]
#
#   steps    corpora | profiles | projects   (default: all three, in that order)
#   --smoke  a few books and a few articles instead of the documented scale -- for iterating
#            on the tutorial, and for a CI run that exercises the path without paying for it
#   --force  redo a step whose output already exists
#
# Every step is idempotent and caches to disk, so re-running after a failure resumes rather
# than restarting: books already downloaded are not fetched again, a corpus already written is
# not rebuilt, a profile already fitted is not refitted.
#
# Why this is a script and not part of the tutorial page: the work here is measured in tens of
# minutes and hundreds of megabytes, and it depends on the network and on a local TextSearch
# checkout. The tutorial page renders against what this leaves behind, so it stays fast, stays
# reproducible, and stays honest -- every number it prints was computed, not typed.

using Dates
using Downloads
using JSON3

# Qualified on purpose. `append_items!`, `index!` and `search` are exported by
# SimilaritySearchEngine, SimilaritySearch and TextSearch alike, and a bare `using` of all
# three makes every one of them ambiguous at the call site.
import SimilaritySearchEngine as SSE
import TextSearch

const HERE = @__DIR__
const DATA = joinpath(HERE, "data")
const RAW = joinpath(DATA, "raw")
const PROFILES = joinpath(HERE, "profiles")
const PROJECTS = joinpath(HERE, "projects")

# The local TextSearch checkout: this tutorial needs three things from it that are not in the
# released package -- the `textsearch` CLI app that fits profiles, the parquet->JSONL converter
# that produced the corpora the published profiles were fitted on, and the Wikipedia dump
# itself. Override with TEXTSEARCH_REPO if yours lives elsewhere.
const TEXTSEARCH_REPO = get(ENV, "TEXTSEARCH_REPO", normpath(joinpath(HERE, "..", "..", "TextSearch.jl")))

# ── what gets built ──────────────────────────────────────────────────────────
#
# Four corpora, one per (source, language). Paragraphs are the retrieval unit in all four:
# a search returns the paragraph that answers it, not the book or article it sits in, which is
# also the unit the published wiki20231101-*-paragraphs profiles were fitted on.
#
# `del_diac=false, lc=false` on the Spanish and Portuguese fits, and on English too: keeping
# case and diacritics is what leaves anything for orthographic correction to bridge at query
# time (see the tutorial's section on QueryPolicy). Folding them at index time is cheaper and
# throws that away.

struct Corpus
    name::String            # also the corpus file's basename, the profile prefix and the project name
    source::Symbol          # :gutenberg | :wikipedia
    lang::String            # ISO 639-1, recorded in the profile's TextConfig
    n_full::Int             # documents (books / articles) at documented scale
    n_smoke::Int            # ... and under --smoke
end

const CORPORA = [
    Corpus("gutenberg-en", :gutenberg,  "en", 100,  3),
    Corpus("gutenberg-es", :gutenberg,  "es", 100,  3),
    Corpus("gutenberg-pt", :gutenberg,  "pt", 100,  3),
    Corpus("wikipedia-en", :wikipedia,  "en", 1000, 20),
]

# A paragraph shorter than this is a chapter heading, a page number or a stray line of
# front matter, not something anyone wants back from a search. Four tokens is the same floor
# TextSearch's own corpus-profile build applies (`--min-tokens 4`), kept identical so a corpus
# built here is comparable with the ones the published profiles were fitted on.
const MIN_TOKENS = 4

# --smoke caps paragraphs, not just documents. Capping documents alone does not bound the run:
# three Project Gutenberg volumes came to 14 MB on the first attempt, because a collected works
# or a dictionary is fifty times a short novel and the catalogue does not say which you get. A
# smoke run has to be fast whatever the draw.
const MAX_SMOKE_PARAGRAPHS = 3000

smoke = false
force = false
ndocs(c::Corpus) = smoke ? c.n_smoke : c.n_full
corpus_path(c::Corpus) = joinpath(DATA, c.name * (smoke ? "-smoke" : "") * ".jsonl")
profile_path(c::Corpus) = joinpath(PROFILES, c.name * (smoke ? "-smoke" : "") * ".zip")
project_dir() = smoke ? joinpath(PROJECTS, "smoke") : PROJECTS

say(msg...) = println("[", Dates.format(now(), "HH:MM:SS"), "] ", msg...)

# ── the build manifest ───────────────────────────────────────────────────────
#
# Every count and timing this script measures is written to data/manifest.json, and the
# tutorial page reads them from there instead of quoting numbers in prose. Two reasons. The
# page renders in seconds and must not rebuild anything to have a figure to show; and a number
# typed into a document is a number that silently goes stale, while one read from the file the
# build wrote is either current or absent.

const MANIFEST = Dict{String,Any}()

manifest_path() = joinpath(DATA, smoke ? "manifest-smoke.json" : "manifest.json")

function record!(section::AbstractString, name::AbstractString, entry)
    sec = get!(() -> Dict{String,Any}(), MANIFEST, section)
    sec[name] = entry
    nothing
end

function save_manifest()
    mkpath(DATA)
    # Merged rather than overwritten: the three steps are separately runnable, so a run of
    # `projects` alone must not erase what `corpora` measured an hour earlier.
    merged = isfile(manifest_path()) ? Dict{String,Any}(JSON3.read(read(manifest_path(), String), Dict{String,Any})) : Dict{String,Any}()
    for (section, entries) in MANIFEST
        sec = get!(() -> Dict{String,Any}(), merged, section)
        merge!(sec, entries)
    end
    merged["scale"] = smoke ? "smoke" : "full"
    merged["built_at"] = string(now())
    open(manifest_path(), "w") do io
        JSON3.pretty(io, merged)
    end
    say("manifest -> ", manifest_path())
end

# ── preconditions ────────────────────────────────────────────────────────────
#
# Checked up front, all of them, before any work starts. A missing Wikipedia dump that only
# surfaces after twenty minutes of downloading books is a worse failure than the same message
# printed immediately, and the list form means one run tells you everything to fix rather than
# one thing per run.

function check_preconditions(steps)
    problems = String[]
    isdir(TEXTSEARCH_REPO) ||
        push!(problems, "no TextSearch checkout at $TEXTSEARCH_REPO (set TEXTSEARCH_REPO)")

    if "profiles" in steps
        isfile(joinpath(TEXTSEARCH_REPO, "apps", "textsearch", "src", "main.jl")) ||
            push!(problems, "no textsearch CLI at $TEXTSEARCH_REPO/apps/textsearch -- profiles are fitted with it")
        isfile(joinpath(TEXTSEARCH_REPO, "apps", "textsearch", "Manifest.toml")) ||
            push!(problems, "the textsearch CLI environment is not instantiated: " *
                            "julia --project=$TEXTSEARCH_REPO/apps/textsearch -e 'using Pkg; Pkg.instantiate()'")
    end

    if "corpora" in steps
        isfile(wikipedia_converter()) ||
            push!(problems, "no parquet->JSONL converter at $(wikipedia_converter())")
        isempty(wikipedia_shards()) &&
            push!(problems, "no Wikipedia parquet shards under $(wikipedia_dir()) -- " *
                            "this tutorial reads the dump TextSearch already downloaded to fit its " *
                            "published profiles; see corpus-profiles/corpora/wikipedia.sh")
    end

    isempty(problems) && return
    println(stderr, "prepare.jl cannot start:")
    for p in problems
        println(stderr, "  - ", p)
    end
    exit(1)
end

wikipedia_dir() = joinpath(TEXTSEARCH_REPO, "corpus-profiles", "raw", "wikipedia", "20231101.en")
wikipedia_converter() = joinpath(TEXTSEARCH_REPO, "corpus-profiles", "lib", "parquet_to_jsonl.jl")
wikipedia_shards() = isdir(wikipedia_dir()) ? sort(filter(endswith(".parquet"), readdir(wikipedia_dir(); join=true))) : String[]

# ── step 1: corpora ──────────────────────────────────────────────────────────

"""
    paragraphs(text) -> Vector{String}

Blank-line-separated blocks, each flattened to a single line.

Project Gutenberg's plain text is hard-wrapped at ~70 columns, so a paragraph arrives as
several physical lines and the newlines inside it carry no information -- they have to go, or
every paragraph would be stored with the transcription's line breaks baked in. Blocks under
[`MIN_TOKENS`](@ref) words are dropped.

Line endings are normalised first, and that is not defensive tidying: Project Gutenberg serves
CRLF, so a blank line arrives as CR-LF-CR-LF, and a separator pattern written for two newlines
matches nothing at all. The failure is silent and total -- every book comes back as a single
"paragraph" holding the whole volume, one of them 12.5 MB -- and it looks like a corpus right
up until you count the records. `check_paragraphs` exists because of it.
"""
function paragraphs(text::AbstractString)
    out = String[]
    for block in split(replace(text, "\r\n" => "\n"), r"\n[ \t]*\n+")
        flat = strip(replace(block, r"\s+" => " "))
        isempty(flat) && continue
        count(isspace, flat) + 1 >= MIN_TOKENS || continue
        push!(out, String(flat))
    end
    out
end

"""
    check_paragraphs(name, blocks) -> blocks

Refuses a suspiciously unsplit document.

A splitter that fails to split does not raise: it returns one enormous block, and everything
downstream accepts it, so the corpus, the fit and the index all "succeed" while only the record
count says anything is wrong. Cheap to assert here, expensive to notice three steps later.
"""
function check_paragraphs(name::AbstractString, blocks::Vector{String})
    if length(blocks) <= 2 && any(b -> length(b) > 100_000, blocks)
        error("$name split into $(length(blocks)) block(s), the largest " *
              "$(maximum(length, blocks)) characters -- the paragraph separator did not match. " *
              "Check the line endings.")
    end
    blocks
end

"""
    strip_gutenberg(text) -> Union{String,Nothing}

The book itself, between Project Gutenberg's START and END markers.

`nothing` when the markers are absent, and the caller skips that book rather than indexing the
licence: the boilerplate is ~10 KB of identical legal text repeated across every volume, which
is exactly the kind of thing a document-frequency-based stopword detector would latch onto and
a search would keep returning.
"""
function strip_gutenberg(text::AbstractString)
    s = findfirst(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*"is, text)
    e = findfirst(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*"is, text)
    (s === nothing || e === nothing || last(s) >= first(e)) && return nothing
    String(text[nextind(text, last(s)):prevind(text, first(e))])
end

"""
    fetch_json(url; attempts=5) -> JSON3 value

`url` fetched and parsed, retrying with exponential backoff.

Not optional politeness: the Gutendex catalogue timed out on the third page of a hundred-book
English run ("Operation too slow. Less than 1 bytes/sec transferred"), twenty minutes into a
build. A public API that is usually fine is still an API that will fail once per long run, and
a build measured in tens of minutes has to survive that rather than restart because of it.
"""
function fetch_json(url::AbstractString; attempts::Int=5)
    for attempt in 1:attempts
        try
            return JSON3.read(sprint(io -> Downloads.download(url, io; timeout=60)))
        catch err
            attempt == attempts && rethrow()
            wait_s = 2.0^attempt
            @warn "retrying $url in $(wait_s)s" attempt err
            sleep(wait_s)
        end
    end
end

"""
    gutendex_catalog(lang, n) -> Vector{NamedTuple}

`n` books in `lang` from the Gutendex catalogue API, each with a UTF-8 plain-text URL.

Ordered by Gutenberg's own popularity ranking (Gutendex's default), which is what makes the
corpus reproducible without pinning a list of ids: the same call returns the same books.
Entries with no UTF-8 plain-text format are skipped -- some volumes are scans with no
transcription -- so this pages until it has `n` usable ones rather than reading `n` entries.

The resolved list is cached to disk. Paging through the catalogue is several round trips to a
public API before a single book is downloaded, and having a run fail there means paying for
them again; with the cache, a resumed run starts at the first book it has not fetched.
"""
function gutendex_catalog(lang::AbstractString, n::Int)
    cache = joinpath(RAW, "catalog-$lang-$n.json")
    if isfile(cache) && !force
        entries = JSON3.read(read(cache, String))
        return [(id=e.id, title=String(e.title), author=String(e.author), url=String(e.url))
                for e in entries]
    end

    books = NamedTuple[]
    url = "https://gutendex.com/books?languages=$lang"
    while length(books) < n && !isempty(url)
        page = fetch_json(url)
        for b in page.results
            length(books) < n || break
            txt = nothing
            for (k, v) in pairs(b.formats)
                if occursin("text/plain", String(k)) && occursin("utf-8", lowercase(String(k)))
                    txt = String(v)
                    break
                end
            end
            txt === nothing && continue
            author = isempty(b.authors) ? "" : String(b.authors[1].name)
            push!(books, (id=b.id, title=String(b.title), author=author, url=txt))
        end
        url = page.next === nothing ? "" : String(page.next)
    end
    length(books) == n ||
        error("Gutendex only offered $(length(books)) usable $lang books, needed $n")
    mkpath(RAW)
    open(cache, "w") do io
        JSON3.write(io, books)
    end
    books
end

function build_gutenberg(c::Corpus)
    out = corpus_path(c)
    if isfile(out) && filesize(out) > 0 && !force
        say("corpus $(c.name): already built ($(round(filesize(out)/2^20; digits=1)) MB), skipping")
        return
    end
    n = ndocs(c)
    say("corpus $(c.name): asking Gutendex for $n books")
    catalog = gutendex_catalog(c.lang, n)

    bookdir = joinpath(RAW, c.name)
    mkpath(bookdir)
    written, skipped, kept = 0, 0, 0
    open(out, "w") do io
        for (i, b) in enumerate(catalog)
            path = joinpath(bookdir, "$(b.id).txt")
            if !isfile(path) || filesize(path) == 0
                ok = false
                for attempt in 1:3
                    try
                        Downloads.download(b.url, path)
                        ok = true
                        break
                    catch err
                        attempt == 3 && @warn "giving up on book $(b.id)" title=b.title err
                        sleep(2.0 * attempt)
                    end
                end
                # Gutenberg asks robots to go easy; this is a courtesy pause, not a rate limit
                # we were told about.
                sleep(0.3)
                ok || (skipped += 1; continue)
            end

            bytes = read(path)
            if !isvalid(String, bytes)
                @warn "book $(b.id) is not valid UTF-8, skipping" title=b.title
                skipped += 1
                continue
            end
            body = strip_gutenberg(String(bytes))
            if body === nothing
                @warn "book $(b.id) has no Gutenberg START/END markers, skipping" title=b.title
                skipped += 1
                continue
            end

            kept += 1
            for (j, p) in enumerate(check_paragraphs("book $(b.id) ($(b.title))", paragraphs(body)))
                smoke && written >= MAX_SMOKE_PARAGRAPHS && break
                JSON3.write(io, (doc_id="$(c.name)-$(b.id)-p$j", text=p, source="gutenberg",
                                 lang=c.lang, book_id=b.id, title=b.title, author=b.author,
                                 paragraph=j))
                println(io)
                written += 1
            end
            smoke && written >= MAX_SMOKE_PARAGRAPHS && break
            i % 10 == 0 && say("  $(c.name): $i/$n books, $written paragraphs")
        end
    end
    record!("corpora", c.name, (source="gutenberg", lang=c.lang, documents=kept, skipped=skipped,
                                paragraphs=written, megabytes=round(filesize(out)/2^20; digits=1)))
    say("corpus $(c.name): $kept books ($skipped skipped) -> $written paragraphs, ",
        round(filesize(out)/2^20; digits=1), " MB")
end

"""
    build_wikipedia(c)

Runs TextSearch's own `parquet_to_jsonl.jl` over the first shard of the English Wikipedia dump
it already downloaded, then rewrites the result with this tutorial's record shape.

Reusing that converter rather than reading the parquet here is the point: it is what produced
the corpora the published `wiki20231101-*-paragraphs` profiles were fitted on, so a paragraph
here is split exactly the way a paragraph in those profiles' training data was. A
reimplementation that split on slightly different boundaries would make the profile and the
corpus quietly disagree about what a document is.

Note which shard: Wikipedia's dump is ordered longest-article-first, so shard 0 yields ~60
paragraphs per article against a corpus-wide ~7. A thousand articles from here is a bigger,
denser corpus than a thousand sampled at random would be.
"""
function build_wikipedia(c::Corpus)
    out = corpus_path(c)
    if isfile(out) && filesize(out) > 0 && !force
        say("corpus $(c.name): already built ($(round(filesize(out)/2^20; digits=1)) MB), skipping")
        return
    end
    n = ndocs(c)
    shard = first(wikipedia_shards())
    tmp = out * ".converter"
    say("corpus $(c.name): converting $n articles from $(basename(shard))")
    run(`julia --project=$(joinpath(TEXTSEARCH_REPO, "apps", "textsearch"))
         $(wikipedia_converter()) $tmp $shard
         --limit $n --split-paragraphs --min-tokens $MIN_TOKENS --keep-columns id,title,url`)

    written = 0
    open(out, "w") do io
        for line in eachline(tmp)
            smoke && written >= MAX_SMOKE_PARAGRAPHS && break
            o = JSON3.read(line)
            JSON3.write(io, (doc_id="$(c.name)-$(o.id)-p$(o.paragraph)", text=String(o.text),
                             source="wikipedia", lang=c.lang, article_id=String(o.id),
                             title=String(o.title), url=String(o.url), paragraph=o.paragraph))
            println(io)
            written += 1
        end
    end
    rm(tmp; force=true)
    record!("corpora", c.name, (source="wikipedia", lang=c.lang, documents=n, skipped=0,
                                paragraphs=written, megabytes=round(filesize(out)/2^20; digits=1)))
    say("corpus $(c.name): $n articles -> $written paragraphs, ",
        round(filesize(out)/2^20; digits=1), " MB")
end

step_corpora() = for c in CORPORA
    mkpath(DATA); mkpath(RAW)
    c.source === :gutenberg ? build_gutenberg(c) : build_wikipedia(c)
end

# ── step 2: profiles ─────────────────────────────────────────────────────────

"""
    fit_config(c, corpus, outdir) -> String

The TOML `textsearch fit` reads, written to a temporary file.

Fitting through the CLI rather than through the engine's own `fit_profile` is deliberate and
is what the tutorial is about: `fit_profile` produces a vocabulary and weights, which is all an
indexing corpus can yield, while `textsearch fit` additionally runs an LSI over the corpus and
derives a query-expansion network and a lemma map from it. Those are the artifacts a project
cannot estimate for itself, and having real ones is what makes the tutorial's expansion and
correction sections demonstrate something instead of describing it.
"""
function fit_config(c::Corpus, corpus::String, outdir::String)
    path, io = mktemp()
    write(io, """
    [input]
    format = "jsonl"
    path = "$corpus"
    text_key = "text"

    [output]
    dir = "$outdir"
    prefix = "$(c.name)"
    batch_size = 0

    # Case and diacritics are kept, in every language including English. Folding them is what
    # a default TextConfig() does and it costs the query side its only lever: with `leon` and
    # `León` already collapsed into one token there is no variant left for QueryPolicy to
    # bridge, and `derive_variants` returns an empty map.
    [normalization]
    del_diac = false
    del_dup = false
    del_punc = true
    group_num = true
    group_url = true
    group_usr = false
    group_emo = false
    lc = false

    [tokenization]
    nlist = [1]
    mark_token_type = true

    [stopwords]
    enabled = true
    doc_freq_threshold = 0.5

    [encoder]
    kind = "lsi"
    outdim = 128
    scaling = "none"

    [query_expansion]
    k = 8

    # Detected and carried, not applied: a lemma map baked into the TextConfig changes what a
    # token *is* on both sides, and the tutorial wants to show the carried-versus-applied
    # distinction rather than decide it here.
    [lemmas]
    algorithm = "fft"
    num_clusters = 0
    selector = "shortest"
    apply = false
    """)
    close(io)
    path
end

function fit_one(c::Corpus)
    out = profile_path(c)
    if isfile(out) && filesize(out) > 0 && !force
        say("profile $(c.name): already fitted ($(round(filesize(out)/2^20; digits=1)) MB), skipping")
        return
    end
    corpus = corpus_path(c)
    isfile(corpus) || error("corpus $(basename(corpus)) is missing -- run the `corpora` step first")

    staging = mktempdir()
    cfg = fit_config(c, corpus, staging)
    say("profile $(c.name): fitting (LSI + query expansion + lemmas; minutes, not seconds)")
    t = @elapsed run(`julia --project=$(joinpath(TEXTSEARCH_REPO, "apps", "textsearch"))
                      $(joinpath(TEXTSEARCH_REPO, "apps", "textsearch", "src", "main.jl"))
                      fit --config $cfg`)

    # `fit` numbers its output even when batch_size = 0 puts everything in one file; rename to
    # the stable name the tutorial loads, so the page never has to know about batch numbering.
    produced = sort(filter(endswith(".zip"), readdir(staging; join=true)))
    length(produced) == 1 ||
        error("expected one profile from fit, got $(length(produced)): $(basename.(produced)). " *
              "batch_size should be 0 so the whole corpus lands in a single profile.")
    mkpath(PROFILES)
    mv(only(produced), out; force=true)
    rm(staging; recursive=true, force=true)
    rm(cfg; force=true)
    record!("profiles", c.name, (megabytes=round(filesize(out)/2^20; digits=1),
                                 fit_seconds=round(t; digits=1)))
    say("profile $(c.name): ", round(filesize(out)/2^20; digits=1), " MB in ", round(t; digits=1), "s")
end

step_profiles() = foreach(fit_one, CORPORA)

# ── step 3: projects ─────────────────────────────────────────────────────────

function build_project(c::Corpus)
    dir = project_dir()
    mkpath(dir)
    if isdir(joinpath(dir, c.name)) && !force
        say("project $(c.name): already built, skipping")
        return
    end
    isdir(joinpath(dir, c.name)) && rm(joinpath(dir, c.name); recursive=true)

    profile = profile_path(c)
    isfile(profile) || error("profile $(basename(profile)) is missing -- run the `profiles` step first")

    say("project $(c.name): loading profile")
    t_load = @elapsed base = TextSearch.load_profile(profile)
    # The network is *carried* by a freshly fitted profile, not applied: `fit` computes it and
    # leaves the decision to whoever indexes with it. Turning it on here is what lets
    # QueryPolicy's `expansion` mean anything for this project.
    prof = TextSearch.with_applied(base; query_expansion=true)

    say("project $(c.name): creating (this writes the whole profile into the project)")
    t_create = @elapsed h = SSE.create_project(dir, c.name;
        engine=FullTextEngine, backend=TextSearch.BM25InvertedFile, textmodel=SSE.BaseProfile(prof))

    # The engine takes typed items, so the JSONL line is split here rather than handed over for
    # the library to pick apart by key name. This script wrote the file, so it is the one that
    # knows `text` is the payload and everything else is metadata.
    n, t_append, t_index = 0, 0.0, 0.0
    batch = SSE.TextItem[]
    for line in eachline(corpus_path(c))
        o = JSON3.read(line)
        meta = Dict{String,Any}(String(k) => v for (k, v) in pairs(o) if k !== :text && k !== :doc_id)
        push!(batch, SSE.TextItem(String(o.text); doc_id=String(o.doc_id), meta))
        if length(batch) == 5000
            t_append += @elapsed SSE.append_items!(h, batch)
            n += length(batch); empty!(batch)
            say("  $(c.name): staged $n paragraphs")
        end
    end
    isempty(batch) || (t_append += @elapsed SSE.append_items!(h, batch); n += length(batch))

    say("project $(c.name): indexing $n paragraphs")
    t_index = @elapsed SSE.index!(h)
    SSE.close_project!(h)

    bytes = sum(filesize(joinpath(r, f)) for (r, _, fs) in walkdir(joinpath(dir, c.name)) for f in fs; init=0)
    record!("projects", c.name, (paragraphs=n,
                                 vocsize=Int(TextSearch.vocsize(prof.model.voc)),
                                 load_profile_seconds=round(t_load; digits=1),
                                 create_seconds=round(t_create; digits=1),
                                 append_seconds=round(t_append; digits=1),
                                 index_seconds=round(t_index; digits=1),
                                 index_ms_per_paragraph=round(1000 * t_index / max(n, 1); digits=2),
                                 megabytes=round(bytes/2^20; digits=1)))
    say("project $(c.name): $n paragraphs | load $(round(t_load; digits=1))s ",
        "create $(round(t_create; digits=1))s append $(round(t_append; digits=1))s ",
        "index $(round(t_index; digits=1))s | ", round(bytes/2^20; digits=1), " MB on disk")
end

step_projects() = foreach(build_project, CORPORA)

# ── driver ───────────────────────────────────────────────────────────────────

function main(argv)
    global smoke, force
    smoke = "--smoke" in argv
    force = "--force" in argv
    steps = filter(a -> !startswith(a, "--"), argv)
    isempty(steps) && (steps = ["corpora", "profiles", "projects"])
    known = ("corpora", "profiles", "projects")
    for s in steps
        s in known || error("unknown step $(repr(s)); expected any of $(join(known, ", "))")
    end

    check_preconditions(steps)
    say("prepare.jl: steps=", join(steps, ","), smoke ? " (smoke)" : "", force ? " (force)" : "")
    for s in steps
        s == "corpora"  && step_corpora()
        s == "profiles" && step_profiles()
        s == "projects" && step_projects()
    end
    save_manifest()
    say("done")
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
