using Pkg
Pkg.activate(@__DIR__)
Pkg.add("JSON3")
Pkg.add("Downloads")
# TextSearch's registered release (v1.0.0) predates the LatentSemanticIndexing feature --
# `Pkg.develop` the same local dev checkout the main SimilaritySearchServer project already
# uses, so this script gets the real LSI module.
Pkg.develop(path=joinpath(@__DIR__, "..", "..", "TextSearch.jl"))

using Downloads
using JSON3
using TextSearch

const LSI_DIM = 128

const BOOKS = [
    ("frankenstein", "https://www.gutenberg.org/cache/epub/84/pg84.txt"),
    ("pride_and_prejudice", "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"),
    ("alice", "https://www.gutenberg.org/cache/epub/11/pg11.txt"),
    ("dorian_gray", "https://www.gutenberg.org/cache/epub/174/pg174.txt"),
    ("moby_dick", "https://www.gutenberg.org/cache/epub/2701/pg2701.txt"),
    ("romeo_and_juliet", "https://www.gutenberg.org/cache/epub/1513/pg1513.txt"),
    ("dracula", "https://www.gutenberg.org/cache/epub/345/pg345.txt"),
    ("two_cities", "https://www.gutenberg.org/cache/epub/98/pg98.txt"),
    ("huckleberry_finn", "https://www.gutenberg.org/cache/epub/76/pg76.txt"),
    ("yellow_wallpaper", "https://www.gutenberg.org/cache/epub/1952/pg1952.txt")
]

const VERBS = Set(["is", "are", "was", "were", "has", "have", "had", "do", "does", "did", "say", "said", "go", "went"])
const STOPWORDS = Set([
    "The", "A", "An", "I", "He", "She", "It", "We", "They", "You", 
    "In", "On", "At", "To", "And", "Or", "But", "If", "For", "With", 
    "As", "By", "Of", "This", "That", "His", "Her", "My", "Your", 
    "Their", "Our", "Mr", "Mrs", "Miss", "Project", "Gutenberg", "All", 
    "No", "Not", "What", "When", "Where", "Who", "Why", "How", "So", 
    "Then", "There", "Here", "Now", "Up", "Down", "Out", "Very", "One", 
    "Two", "Some", "Any", "Such", "Only", "Own", "More", "Most", "Other"
])

function get_main_characters(text::String, top_k::Int=10)
    words = split(text)
    counts = Dict{String, Int}()
    for w in words
        w_clean = strip(w, ['\"', '\'', '(', ')', ',', '.', '!', '?', ':', ';'])
        if length(w_clean) > 2 && isuppercase(w_clean[1]) && !in(w_clean, STOPWORDS) && !isuppercase(w_clean[2])
            counts[w_clean] = get(counts, w_clean, 0) + 1
        end
    end
    sorted = sort(collect(counts), by=x->x[2], rev=true)
    return [x[1] for x in first(sorted, min(top_k, length(sorted)))]
end

"""
    fit_lsi(paragraphs) -> TextSearch.LatentSemanticIndexing

Entrena un modelo LSI (Latent Semantic Indexing) sobre el corpus de parrafos de un libro y
lo proyecta a vectores densos de `LSI_DIM` dimensiones, para usarlos como el campo `vector`
del dataset (en lugar del histograma ASCII anterior, que no capturaba similitud semantica).
"""
function fit_lsi(paragraphs::Vector{String})
    lsi = LatentSemanticIndexing(paragraphs; maxoutdim=LSI_DIM, verbose=false)
    outdim(lsi) == LSI_DIM || error("LSI produced $(outdim(lsi))-dim vectors, expected $LSI_DIM (corpus too small)")
    return lsi
end

function process_book(name::String, url::String, outdir::String)
    println("Processing $name...")
    filepath = joinpath(outdir, "$name.txt")
    if !isfile(filepath)
        println("  Downloading $url...")
        Downloads.download(url, filepath)
    end
    
    text = read(filepath, String)
    # Limpiamos el texto para omitir las licencias de Gutenberg
    start_idx = findfirst("*** START OF THE PROJECT GUTENBERG", text)
    if start_idx === nothing
        start_idx = findfirst("*** START OF THIS PROJECT GUTENBERG", text)
    end
    
    end_idx = findfirst("*** END OF THE PROJECT GUTENBERG", text)
    if end_idx === nothing
        end_idx = findfirst("*** END OF THIS PROJECT GUTENBERG", text)
    end
    
    if start_idx !== nothing
        # Buscamos el final de la linea de inicio
        eol = findnext("\n", text, start_idx[end])
        if eol !== nothing
            text = text[eol[1]:end]
        end
    end
    if end_idx !== nothing
        text = text[1:end_idx[1]-1]
    end
    
    main_chars = get_main_characters(text, 15)
    println("  Main characters detected: ", join(main_chars, ", "))
    
    text = replace(text, "\r\n" => "\n")
    paragraphs = split(text, "\n\n")

    # Primera pasada: limpiamos los parrafos y calculamos su metadata, sin vectorizar
    # todavia -- el modelo LSI necesita ver el corpus completo del libro antes de poder
    # proyectar cualquier parrafo individual.
    cleaned = String[]
    metas = Vector{Dict{String,Any}}()
    for p in paragraphs
        p_clean = strip(p)
        if length(p_clean) < 50
            continue
        end

        # Removemos saltos de linea dentro del parrafo para hacerlo texto continuo
        p_clean = replace(p_clean, "\r\n" => " ", "\n" => " ")

        words = split(p_clean)
        word_count = length(words)

        verb_count = 0
        chars_mentioned = String[]

        for w in words
            w_lower = lowercase(strip(w, ['\"', '\'', '(', ')', ',', '.', '!', '?', ':', ';']))
            if in(w_lower, VERBS)
                verb_count += 1
            end

            w_clean = strip(w, ['\"', '\'', '(', ')', ',', '.', '!', '?', ':', ';'])
            if in(w_clean, main_chars) && !(w_clean in chars_mentioned)
                push!(chars_mentioned, w_clean)
            end
        end

        push!(cleaned, p_clean)
        push!(metas, Dict("word_count" => word_count, "verb_count" => verb_count, "keywords" => chars_mentioned))
    end

    println("  Fitting LSI ($LSI_DIM dims) on $(length(cleaned)) paragraphs...")
    lsi = fit_lsi(cleaned)

    # Record/data/metadata contract (SimilaritySearchEngine.jl's Schema.MetadataRecord +
    # Schema.split_item): "vector"/"text" are the raw data (never duplicated into meta);
    # "doc_id"/"keywords"/"ref" are the record's fixed fields (doc_id is *our* external id,
    # keywords the character names detected in this paragraph); everything else
    # (word_count, verb_count) falls through to the free-form meta blob as-is, flat, no
    # extra "meta" wrapper key of its own.
    out_jsonl = joinpath(outdir, "$name.jsonl")
    open(out_jsonl, "w") do io
        for (doc_id, (p_clean, meta)) in enumerate(zip(cleaned, metas))
            vec = vectorize(lsi, p_clean)

            record = Dict(
                "doc_id" => "$(name)_$doc_id",
                "text" => p_clean,
                "vector" => vec,
                "keywords" => meta["keywords"],
                "ref" => doc_id > 1 ? ["$(name)_$(doc_id - 1)"] : String[],  # previous paragraph, unvalidated
                "word_count" => meta["word_count"],
                "verb_count" => meta["verb_count"],
            )

            println(io, JSON3.write(record))
        end
        println("  Saved $(length(cleaned)) paragraphs to $out_jsonl")
    end
end

function main()
    outdir = joinpath(@__DIR__, "data")
    mkpath(outdir)
    
    for (name, url) in BOOKS
        try
            process_book(name, url, outdir)
        catch e
            println("  Error processing $name: $e")
        end
    end
    println("Done!")
end

main()
