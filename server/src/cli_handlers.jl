# CLI operations
using JSON3
using TOML
using HTTP
using Dates

# The value of `build --storage`: a JSON object, as the `storage` field of `POST /api/v1/datasets`.
function _storage_arg(arg)
    arg === nothing && return nothing
    try
        return JSON3.read(codeunits(arg), Dict{String,Any})
    catch e
        e isa Union{ArgumentError,MethodError} || rethrow()
        throw(SSE.InvalidOption(:storage, "--storage is a JSON object, such as '{\"kind\": \"quantized\", \"bits\": 4}'; " *
                                          "got $(repr(arg))"))
    end
end

"""
    cli_exit_code(e::SSE.EngineError) -> Cint

The CLI half of the error boundary (the HTTP half is `Server.engine_error_response`): an
engine error becomes an exit code by its *category*, so a shell script can branch on why a
command failed without reading its output.

    2  an invalid request -- wrong arguments, wrong kind of project. Retrying will not help.
    4  something named does not exist.
    5  a conflicting state -- a staged backlog, an untrained profile. The same command works
       once the project is in a different state.
    70 a storage failure (sysexits' EX_SOFTWARE): stored data this version cannot read.

`1` stays what it always was here: this command's own refusal, decided before the engine was
ever called (no such dataset directory, unreadable input file).
"""
cli_exit_code(e::SSE.EngineError) =
    Cint(e isa SSE.NotFound ? 4 :
         e isa SSE.ConflictingState ? 5 :
         e isa SSE.InvalidRequest ? 2 : 70)

"The `--config` file of the running command, named by the message of a plugin that is not loaded (CHANGELOG.md §5.13)."
const CLI_CONFIG = Ref{String}("config.toml")

"""
    @cli_engine_call expr

Runs an engine call, and on an [`SSE.EngineError`](@ref) prints it and evaluates to the exit
code [`cli_exit_code`](@ref) gives it. Use it as

    value = @cli_engine_call SSE.allknn(handle; k=k)
    value isa Cint && return value

which keeps the mapping in one place instead of a `try`/`catch` at every command.
"""
macro cli_engine_call(expr)
    quote
        try
            $(esc(expr))
        catch e
            e isa SSE.EngineError || rethrow()
            println("Error: ", sprint(showerror, e))
            e isa SSE.PluginNotLoaded && println("similarity-search loads the plugins of the [plugins] section of ",
                "the file of --config: ", CLI_CONFIG[], isfile(CLI_CONFIG[]) ? "" : ", which does not exist")
            cli_exit_code(e)
        end
    end
end

function execute_build(cmd_args::Dict)
    project_id = cmd_args["dataset"]
    input_file = cmd_args["input"]
    index_kind = cmd_args["index-kind"]
    distance = cmd_args["distance"]
    workdir = cmd_args["workdir"]
    max_k = something(get(cmd_args, "max-k", nothing), Server.DEFAULT_MAX_K)
    max_k >= 1 || SSE.Errors.invalid_option(:max_k, "--max-k must be a positive integer; got $max_k")

    println("Building index '$index_kind' for dataset '$project_id'")

    partition_by = get(cmd_args, "partition-by", nothing)
    graph_min = get(cmd_args, "partition-graph-min", nothing)
    graph_min === nothing || partition_by !== nothing ||
        SSE.Errors.invalid_option(:partition_graph_min, "--partition-graph-min needs --partition-by: it is the " *
                                  "size from which a partition has its own graph")
    spec = @cli_engine_call Server.dataset_spec(index_kind, distance, nothing, _storage_arg(get(cmd_args, "storage", nothing)))
    spec isa Cint && return spec
    handle = SSE.create_project(workdir, project_id, spec;
        partitions=partition_by === nothing ? nothing : SSE.PartitionBy(partition_by; graph_min))

    kind = SSE.payload_kind(handle.engine)
    items = SSE.AbstractItem[]
    for line in eachline(input_file)
        isempty(strip(line)) && continue
        it = Server._typed_item(kind, JSON3.read(codeunits(line), Dict{String, Any}))
        it === nothing || push!(items, it)
    end

    inserted = isempty(items) ? 0 : SSE.append_items!(handle, items)
    inserted > 0 && SSE.index!(handle)
    # The project persists itself, which is also why `search`, `describe` and every job below
    # reopen it instead of reading a separate file. The sidecar of the server holds what the
    # project does not: `max_k` (CHANGELOG.md §5.16), which `dump` and `load` carry to a server.
    SSE.close_project!(handle)
    write(joinpath(workdir, project_id, "descriptor.json"), JSON3.write(Dict(
        "id" => project_id, "index_kind" => index_kind, "distance" => distance,
        "created_at" => string(now(UTC)), "edit_correction" => false, "max_k" => max_k)))

    println("Successfully built index with $inserted items. Saved to $(joinpath(workdir, project_id))")
    return 0
end

"""
    _open_handle(project_id, workdir; read_only=false, edit_correction=false) -> Union{Nothing, SSE.EmbeddedEngine}

Reopens a dataset by id, whichever of this codebase's two layouts it lives under --
`<workdir>/datasets/<id>` for one created over HTTP, `<workdir>/<id>` for one built by the
CLI. Prints an error and returns `nothing` when neither exists.

This replaced a snapshot file per dataset, written by `build` and by every HTTP append, and
read by each command that operated on a whole dataset. The engine persists its own index, so
that file was a second and older copy of state that already had a place to live, and a job
that ran before a snapshot had been written failed with "snapshot not found". The functions
that wrote and read it were removed on 2026-09-13, together with the `JLD2` dependency.
"""
function _open_handle(project_id::String, workdir::String; read_only::Bool=false, edit_correction::Bool=false)
    for root in (joinpath(workdir, "datasets"), workdir)
        isdir(joinpath(root, project_id)) || continue
        return SSE.open_project(root, project_id; read_only, runtime=SSE.Runtime(; edit_correction))
    end
    println("Error: dataset '$project_id' not found under $workdir")
    return nothing
end

"""
    wire_kind(engine) -> String

The `index_kind` name this CLI and the HTTP API use for a project, read back from the open
project itself rather than from a descriptor file. The inverse of
`Server.parse_index_kind`.
"""
function wire_kind(engine)
    kind = SSE.payload_kind(engine)
    tag = SSE.IndexEngine.backend_tag(engine.backend)
    kind === :text && return tag === :bm25 ? "bm25_invfile" : "invfile"
    kind === :sparse && return "sparse_invfile"
    tag === :graph && return "searchgraph"
    tag === :parallel_exhaustive && return "parallel_exhaustive_search"
    return "exhaustive_search"
end

function execute_allknn(cmd_args::Dict)
    project_id = cmd_args["dataset"]
    k = cmd_args["k"]
    out_file = cmd_args["output"]
    workdir = cmd_args["workdir"]

    println("Running allknn on dataset '$project_id' (k=$k)")
    # Read-only: a heavy job runs as a subprocess and the server may still hold this
    # project open for writing, and RocksDB's write lock is exclusive per process.
    handle = _open_handle(project_id, workdir; read_only=true)
    handle === nothing && return 1
    rows = @cli_engine_call SSE.allknn(handle; k=k)
    rows isa Cint && return rows

    open(out_file, "w") do io
        for row in rows
            println(io, JSON3.write(Dict("id" => Int(row._id),
                                         "neighbors" => Int.(row.neighbors),
                                         "dists" => Float64.(row.dists))))
        end
    end
    SSE.close_project!(handle)

    println("allknn completed for $(length(rows)) items. Saved to $out_file")
    return 0
end

function execute_fft(cmd_args::Dict)
    project_id = cmd_args["dataset"]
    k = cmd_args["k"]
    out_file = cmd_args["output"]
    workdir = cmd_args["workdir"]

    println("Running fft on dataset '$project_id' (k=$k)")
    # Read-only: a heavy job runs as a subprocess and the server may still hold this
    # project open for writing, and RocksDB's write lock is exclusive per process.
    handle = _open_handle(project_id, workdir; read_only=true)
    handle === nothing && return 1
    R = @cli_engine_call SSE.fft(handle, k)
    R isa Cint && return R

    out_obj = Dict(
        "centers" => Int.(R.centers),
        "assign" => Int.(R.assign),
        "dists" => Float64.(R.assigndist),
        "covering" => Float64(R.covering),
        "separation" => Float64(R.separation),
        "costdists" => Int(R.costdists),
        "costblocks" => Int(R.costblocks),
    )
    SSE.close_project!(handle)
    open(out_file, "w") do io
        println(io, JSON3.write(out_obj))
    end

    println("fft completed with $(length(R.centers)) centers. Saved to $out_file")
    return 0
end

function execute_neardup(cmd_args::Dict)
    project_id = cmd_args["dataset"]
    epsilon = cmd_args["epsilon"]
    out_file = cmd_args["output"]
    workdir = cmd_args["workdir"]

    println("Running neardup on dataset '$project_id' (epsilon=$epsilon)")
    # Read-only: a heavy job runs as a subprocess and the server may still hold this
    # project open for writing, and RocksDB's write lock is exclusive per process.
    handle = _open_handle(project_id, workdir; read_only=true)
    handle === nothing && return 1
    R = @cli_engine_call SSE.neardup(handle, epsilon)
    R isa Cint && return R

    out_obj = Dict(
        "centers" => Int.(R.centers),
        "assign" => Int.(R.assign),
        "dists" => Float64.(R.assigndist),
        "duplicate_count" => length(R.assign) - length(R.centers),
        "covering" => Float64(R.covering),
        "epsilon" => Float64(R.epsilon),
        "costdists" => Int(R.costdists),
        "costblocks" => Int(R.costblocks),
    )
    SSE.close_project!(handle)
    open(out_file, "w") do io
        println(io, JSON3.write(out_obj))
    end

    println("neardup completed: $(out_obj["duplicate_count"]) duplicate(s) found among $(length(R.assign)) item(s). Saved to $out_file")
    return 0
end

function execute_hsp(cmd_args::Dict)
    project_id = cmd_args["dataset"]
    queries_file = cmd_args["queries"]
    k = cmd_args["k"]
    out_file = cmd_args["output"]
    workdir = cmd_args["workdir"]

    println("Running hsp on dataset '$project_id' (k=$k)")
    # Read-only: a heavy job runs as a subprocess and the server may still hold this
    # project open for writing, and RocksDB's write lock is exclusive per process.
    handle = _open_handle(project_id, workdir; read_only=true)
    handle === nothing && return 1

    query_vecs = Vector{Float32}[]
    for line in eachline(queries_file)
        isempty(strip(line)) && continue
        item = JSON3.read(codeunits(line), Dict{String, Any})
        haskey(item, "vector") || continue
        push!(query_vecs, convert(Vector{Float32}, item["vector"]))
    end
    if isempty(query_vecs)
        println("Error: no valid queries with a 'vector' field found in $queries_file")
        return 1
    end

    # The parallel half goes through the engine's own batch entry point, which takes the
    # project's lock and refuses to run against a staged backlog. The pruning step below still
    # reads the index directly: `hsp_queries` has no public wrapper yet, and this process holds
    # the project exclusively anyway (a CLI run, not a served request).
    knns = @cli_engine_call SSE.searchbatch(handle, query_vecs, k)
    knns isa Cint && return knns
    knns_ids, knns_dists = knns

    index = handle.engine.backend.index
    Q = SimilaritySearch.VectorDatabase(query_vecs)
    dist = SimilaritySearch.distance(index)
    X = SimilaritySearch.database(index)
    hsp_ids, hsp_dists, _ = SimilaritySearch.hsp_queries(dist, X, Q, knns_ids, knns_dists)

    n = size(hsp_ids, 2)
    open(out_file, "w") do io
        for i in 1:n
            neighbor_ids = Int[]
            neighbor_dists = Float64[]
            for j in 1:k
                hsp_ids[j, i] == 0 && break
                push!(neighbor_ids, Int(hsp_ids[j, i]))
                push!(neighbor_dists, Float64(hsp_dists[j, i]))
            end
            println(io, JSON3.write(Dict("query" => i, "neighbors" => neighbor_ids, "dists" => neighbor_dists)))
        end
    end

    SSE.close_project!(handle)
    println("hsp completed for $n queries. Saved to $out_file")
    return 0
end

function execute_searchbatch(cmd_args::Dict)
    project_id = cmd_args["dataset"]
    queries_file = cmd_args["queries"]
    k = cmd_args["k"]
    out_file = cmd_args["output"]
    workdir = cmd_args["workdir"]
    
    println("Running searchbatch on dataset '$project_id'")

    # Read-only: a heavy job runs as a subprocess and the server may still hold this
    # project open for writing, and RocksDB's write lock is exclusive per process.
    # `edit_correction` is refused by the engine for a dataset without text, which is an
    # engine error and exits with its code.
    handle = @cli_engine_call _open_handle(project_id, workdir; read_only=true,
                                           edit_correction=get(cmd_args, "edit-correction", false))
    handle isa Cint && return handle
    handle === nothing && return 1
    kind = SSE.payload_kind(handle.engine)
    partition = get(cmd_args, "partition", nothing)
    # `--follower name=value` searches one partition of a partition follower, and `--follower
    # name` a keyword follower with the keys of each query (CHANGELOG.md §5.12).
    fname = get(cmd_args, "follower", nothing)
    target = handle
    if fname !== nothing
        if partition !== nothing
            println("Error: --partition and --follower both restrict the search; give one of them")
            SSE.close_project!(handle; compact=false)
            return Cint(2)
        end
        target = @cli_engine_call SSE.open_follower(handle, fname)
        if target isa SSE.Follower && SSE.follower_status(target).kind != "KeywordFollower"
            println("Error: '$fname' is a $(SSE.follower_status(target).kind); search one of its partitions with ",
                    "--follower $fname=<value>")
            target = Cint(2)
        end
        target isa Cint && (SSE.close_project!(handle; compact=false); return target)
    end

    # The exit code of the first query the engine refused, or 0. A `return` inside the `do`
    # block below would leave only the block, not this function: the command then exited 0
    # after an engine error, which is how an unknown partition went unreported.
    status = open(out_file, "w") do out_io
        for line in eachline(queries_file)
            isempty(strip(line)) && continue
            item = JSON3.read(codeunits(line), Dict{String, Any})

            hits = if target isa SSE.Follower
                haskey(item, "keys") || continue
                @cli_engine_call SSE.search(target, String.(item["keys"]), k; t=Int(get(item, "t", 1)))
            elseif kind === :text
                haskey(item, "text") || continue
                @cli_engine_call(target isa SSE.FollowerView ? SSE.ftsearch(target, item["text"], k) :
                                 SSE.ftsearch(handle, item["text"], k; partition))
            else
                haskey(item, "vector") || continue
                vector = convert(Vector{Float32}, item["vector"])
                @cli_engine_call(target isa SSE.FollowerView ? SSE.search(target, vector, k) :
                                 SSE.search(handle, vector, k; partition))
            end
            hits isa Cint && return hits

            out_obj = Dict("ids" => Int32[h._id for h in hits], "dists" => Float32[h.distance for h in hits])
            println(out_io, JSON3.write(out_obj))
        end
        return Cint(0)
    end
    SSE.close_project!(handle)
    status == 0 || return status

    println("Searchbatch completed. Output saved to $out_file")
    return 0
end

function execute_closestpair(cmd_args::Dict)
    project_id = cmd_args["dataset"]
    min_k = cmd_args["min-k"]
    out_file = cmd_args["output"]
    workdir = cmd_args["workdir"]

    println("Running closestpair on dataset '$project_id' (min_k=$min_k)")
    # Read-only: a heavy job runs as a subprocess and the server may still hold this
    # project open for writing, and RocksDB's write lock is exclusive per process.
    handle = _open_handle(project_id, workdir; read_only=true)
    handle === nothing && return 1
    pairs = @cli_engine_call SSE.closestpairs(handle; k=1, min_k=min_k)
    pairs isa Cint && return pairs
    isempty(pairs) && (println("closestpair: the project has no pair to report"); SSE.close_project!(handle); return 1)
    i, j, dist = first(pairs)
    SSE.close_project!(handle)

    out_obj = Dict("i" => Int(i), "j" => Int(j), "dist" => Float64(dist))
    open(out_file, "w") do io
        println(io, JSON3.write(out_obj))
    end

    println("closestpair completed: items $(Int(i)) and $(Int(j)) at distance $(Float64(dist)). Saved to $out_file")
    return 0
end

"""
    _dense_distance_stats(handle) -> Union{Nothing, Dict}

Real, sampled nearest-neighbor distance statistics (min/mean/max over up to the first 100
items' own nearest live neighbor) -- the "distance statistics ... to help people know
about their data" CHANGELOG.md §1's `describe` note asks for. `nothing` if the index has fewer
than 2 items (no pair to measure) or isn't dense.

A quantized or sketched index holds codes, so the queries are the `Float32` vectors that the
dataset keeps (`keep_vectors`), and `nothing` when it does not keep them. A sketched dataset
without a second evaluation (`rerank` 0) answers with distances between sketches, which are not
the distance of the dataset, so it has no statistics either (CHANGELOG.md §5.14).
"""
function _dense_distance_stats(handle)
    engine = handle.engine
    engine.backend.index === nothing && return nothing
    n = length(engine.backend.index)
    n < 2 && return nothing

    sample_n = min(n, 100)
    ids = collect(1:sample_n)
    # Through the engine's batch entry point, which locks the project and refuses a staged
    # backlog, rather than calling the library against the bare index.
    queries = if IndexEngine.isquantized(engine)
        IndexEngine.issketched(engine) && engine.quant.spec.rerank == 0 && return nothing
        kept = [it.payload for it in SSE.fetch_items(handle, ids)]
        any(v -> v === nothing, kept) && return nothing
        kept
    else
        [Vector{Float32}(SimilaritySearch.database(engine.backend.index, i)) for i in ids]
    end
    raw_ids, raw_dists = SSE.searchbatch(handle, queries, 2)

    nn_dists = Float64[]
    for (col, i) in enumerate(ids)
        for row in 1:size(raw_ids, 1)
            cand = raw_ids[row, col]
            cand == 0 && continue
            if cand != i
                push!(nn_dists, Float64(raw_dists[row, col]))
                break
            end
        end
    end
    isempty(nn_dists) && return nothing

    return Dict(
        "sample_size" => length(nn_dists),
        "nn_dist_min" => minimum(nn_dists),
        "nn_dist_mean" => sum(nn_dists) / length(nn_dists),
        "nn_dist_max" => maximum(nn_dists),
    )
end

function execute_describe(cmd_args::Dict)
    project_id = cmd_args["dataset"]
    workdir = cmd_args["workdir"]
    out_file = get(cmd_args, "output", nothing)

    handle = _open_handle(project_id, workdir; read_only=true)
    handle === nothing && return 1
    ds = handle.project
    engine = handle.engine

    is_text = SSE.payload_kind(engine) === :text
    doc_count = engine.backend.index === nothing ? 0 : length(engine.backend.index)
    tombstones = length(engine.deleted_ids)

    desc = Dict{String, Any}(
        "id" => project_id,
        "kind" => wire_kind(engine),
        "doc_count" => doc_count,
        "tombstone_count" => tombstones,
        "tombstone_ratio" => doc_count == 0 ? 0.0 : tombstones / doc_count,
    )
    # The largest k of a search when a server serves the dataset (CHANGELOG.md §5.16), from the
    # sidecar of the server; a dataset without one has the default.
    descriptor_file = joinpath(handle.dir, "descriptor.json")
    desc["max_k"] = isfile(descriptor_file) ?
        Server._descriptor_max_k(JSON3.read(read(descriptor_file), Dict{String, Any})) : Server.DEFAULT_MAX_K

    if is_text
        # The vocabulary and its out-of-vocabulary rates belong to the engine, which fitted
        # it. `sample=0` asks for every document rather than the bounded sample the HTTP
        # endpoint takes: this command is offline and nobody is waiting on a connection, so it
        # answers exactly (CHANGELOG.md §5's drift signal, see `execute_rebuild`).
        desc["vocab"] = SSE.vocabulary_report(handle; scan=true, sample=0)
    else
        # The distance of the dataset: a quantized or sketched index searches with the distance
        # of its codes (CHANGELOG.md §5.14).
        spec = SSE.project_spec(handle)
        desc["distance"] = string(nameof(typeof(spec.distance)))
        if spec isa SSE.DenseVectors
            desc["storage"] = Server.storage_dict(spec.storage)
            desc["encoder"] = Server.encoder_dict(handle)
        end
        desc["distance_stats"] = _dense_distance_stats(handle)
        bs = IndexEngine.current_beamsearch(engine)
        desc["beamsearch_baseline"] = bs === nothing ? nothing :
            Dict("bsize" => Int(bs.bsize), "delta" => Float64(bs.Δ), "maxvisits" => Int(bs.maxvisits))
    end
    # The followers are read from their saved fields: none is open, so `stats` is null.
    desc["followers"] = [Server.follower_status_dict(s) for s in SSE.followers(handle)]

    SSE.close_project!(handle; compact=false)

    out_json = JSON3.write(desc)
    if out_file === nothing
        println(out_json)
    else
        open(out_file, "w") do io
            println(io, out_json)
        end
        println("Description saved to $out_file")
    end
    return 0
end

# The name of a follower without its leader: "repos/by_repo" -> "by_repo".
_follower_short_name(canonical::AbstractString) = String(last(split(canonical, '/'; limit=2)))

"""
    _report_followers(project_id, statuses)

Prints one line per follower after `rebuild` or `load`: whether it is ready, and the command
that makes it ready when it is not.
"""
function _report_followers(project_id::String, statuses::Vector{SSE.FollowerStatus})
    for st in statuses
        name = _follower_short_name(st.name)
        args = "--dataset $project_id --name $name"
        state = st.missing ? "its directory does not exist; remove the name with similarity-search follower-delete $args" :
                st.stale ? "stale; build it again with similarity-search follower-rebuild $args" :
                st.synchronized ? "synchronized" :
                "$(st.lag) item(s) behind its leader; process them with similarity-search follower-sync $args"
        println("  follower $(st.name): $state")
    end
end

"""
    _live_items(handle) -> (all_ids::Vector{Int}, live::Vector{Tuple{Int,Dict}})

Scans the metadata column family of the project of `handle` once, returning every doc_id
currently stored (`all_ids`, needed to wipe stale rows during a rebuild) alongside the
reconstructed raw item (`declared_fields` merged with the decoded `extra` JSON -- see
`Schema.MetadataRecord`) for every *live* (non-tombstoned) one, sorted by original doc_id.
The index of a quantized or sketched project holds codes, so its vectors are the ones it
keeps (`keep_vectors`), read with `fetch_items`.
"""
function _live_items(handle)
    ds, engine = handle.project, handle.engine
    quantized = IndexEngine.isquantized(engine)
    all_ids = Int[]
    live = Tuple{Int, Dict{String, Any}}[]
    for (_, v) in RocksDB.DBIterator(ds.db; cf=ds.cf_records)
        record = JSON3.read(v, Schema.MetadataRecord)
        push!(all_ids, record._id)
        record._id in engine.deleted_ids && continue
        
        raw = Dict{String, Any}(
            "doc_id" => record.doc_id,
            "keywords" => record.keywords,
            "refs" => record.refs
        )
        
        meta = Project.get_meta(ds, record._id; lazy=false)
        if meta !== nothing
            merge!(raw, Dict{String, Any}("meta" => meta))
        end
        
        payload = quantized ? only(SSE.fetch_items(handle, [record._id])).payload :
                              IndexEngine.stored_payload(engine, record._id)
        if payload !== nothing
            raw[payload isa String ? "text" : "vector"] = payload
        end
        
        push!(live, (record._id, raw))
    end
    sort!(live, by=first)
    return all_ids, live
end

"""
    execute_rebuild(cmd_args::Dict) -> Int

`rebuild` (CHANGELOG.md §1): purges soft-deleted (tombstoned) documents and, for a text index,
retrains the `Vocabulary` from scratch against the dataset's current live corpus -- the
only way to recover from vocabulary drift (CHANGELOG.md §1's note: `TextSearch.jl`'s
`Vocabulary` never mutates after training, so append-time out-of-vocabulary tokens are
silently and permanently dropped otherwise).

!!! note "Full reindex only -- there is no lighter 'compact posting lists' mode"
    An earlier draft of this plan described a second, lighter `rebuild` mode for text
    indices (`filter_lists!`, compacting posting lists without retraining the vocabulary).
    Verified against the current `TextSearch.jl`/`SimilaritySearch.jl` source: no such
    function exists anywhere in either package today (grepped both repos; `BM25InvertedFile`
    has no compaction method at all). This command therefore only implements the one mode
    that *is* real: a full reindex from the live corpus.

Since every dense/text index in this app is a dense array with no notion of a "gap",
purging tombstones means the survivors get **renumbered** to fresh sequential doc_ids
(`1..n'`) matching their new index positions -- their old metadata rows are deleted and
rewritten under the new ids, not merely filtered in place. For a `SearchGraph`, the fresh
index is additionally passed through `SimilaritySearch.rebuild` (a real library function,
not hand-rolled): it recomputes graph topology seeing the whole survivor set at once,
rather than the incremental one-vertex-at-a-time bias `add_item!` reintroduces -- CHANGELOG.md
§1's "use specific functions when available" guidance for a case where one exists.
"""
function execute_rebuild(cmd_args::Dict)
    project_id = cmd_args["dataset"]
    workdir = cmd_args["workdir"]

    # Read-only: a heavy job runs as a subprocess and the server may still hold this
    # project open for writing, and RocksDB's write lock is exclusive per process.
    handle = _open_handle(project_id, workdir; read_only=true)
    handle === nothing && return 1
    dir = handle.dir
    engine = handle.engine

    kind = SSE.payload_kind(engine)
    field = kind === :text ? "text" : "vector"
    # The fresh project has the spec of the old one and is partitioned like it: the field lives
    # in the project, and the partition lists are rebuilt from the items' `meta` as they are
    # appended again.
    spec = SSE.project_spec(handle)
    # A quantized or sketched project encodes its vectors again, with a new encoder, from the
    # Float32 vectors it keeps. Without them there is nothing to encode (CHANGELOG.md §5.14).
    if spec isa SSE.DenseVectors && spec.storage isa Union{SSE.Quantized,SSE.Sketched} && !spec.storage.keep_vectors
        SSE.close_project!(handle; compact=false)
        println("Error: dataset '$project_id' stores its vectors as $(spec.storage isa SSE.Quantized ? "quantized codes" : "sketches") " *
                "and does not keep the Float32 vectors (keep_vectors=false), so rebuild has no vectors to encode again. " *
                "Build the dataset again from its source.")
        return Cint(2)
    end
    partition_by = SSE.partition_field(handle)
    partitions = partition_by === nothing ? nothing :
                 SSE.PartitionBy(partition_by; graph_min=SSE.partition_graph_min(handle))

    # The followers stay in `<dir>.followers`, beside the project, and are attached to the new
    # project (CHANGELOG.md §5.12). A follower whose directory is missing is not kept.
    old_followers = SSE.followers(handle)
    follower_names = [_follower_short_name(st.name) for st in old_followers if !st.missing]
    for st in old_followers
        st.missing && println("The follower '$(st.name)' has no directory and is not kept.")
    end

    all_ids, live = _live_items(handle)
    live_valid = [(id, item) for (id, item) in live if haskey(item, field)]
    dropped_missing_field = length(live) - length(live_valid)

    println("Rebuilding dataset '$project_id': $(length(all_ids)) total record(s), $(length(engine.deleted_ids)) tombstoned, $(length(live_valid)) will be reindexed" *
        (dropped_missing_field > 0 ? " ($dropped_missing_field live record(s) missing '$field', also dropped)" : ""))

    items = SSE.AbstractItem[]
    for (_, item) in live_valid
        it = Server._typed_item(kind, item)
        it === nothing || push!(items, it)
    end
    SSE.close_project!(handle)

    # A rebuild is a fresh project written from the live items, not a surgical edit of the old
    # one: tombstoned rows disappear, ids are renumbered from 1, and a text project refits its
    # vocabulary over what actually survived. The old directory is moved aside rather than
    # deleted until the new one is complete, so an interrupted rebuild leaves the previous
    # project intact under `<dir>.old`.
    root, name = dirname(dir), basename(dir)
    stale = dir * ".old"
    ispath(stale) && rm(stale; recursive=true)
    mv(dir, stale)

    fresh = SSE.create_project(root, name, spec; partitions)
    reindexed = isempty(items) ? 0 : SSE.append_items!(fresh, items)
    reindexed > 0 && SSE.index!(fresh)
    # The new project has a new generation: an attached follower is stale until it is rebuilt,
    # here with --rebuild-followers or later with `follower-rebuild`.
    code = Cint(0)
    for fname in follower_names
        r = @cli_engine_call SSE.attach_follower!(fresh, fname)
        r isa Cint && (code = r; continue)
        if get(cmd_args, "rebuild-followers", false)
            r = @cli_engine_call SSE.rebuild_follower!(fresh, fname)
            r isa Cint && (code = r)
        end
    end
    statuses = SSE.followers(fresh)
    SSE.close_project!(fresh)
    # `descriptor.json` is the HTTP server's own sidecar (index kind, distance, join group,
    # meta schema), written next to the project rather than inside it, and `create_project`
    # above knows nothing about it. Carrying it over is what lets a live server reload the
    # rebuilt dataset -- without it, `POST /admin/datasets/{id}/reload` answers 404 for a
    # dataset that is sitting right there on disk.
    stale_descriptor = joinpath(stale, "descriptor.json")
    isfile(stale_descriptor) && cp(stale_descriptor, joinpath(dir, "descriptor.json"); force=true)
    rm(stale; recursive=true)

    println("Rebuild completed for '$project_id': $reindexed item(s) reindexed into $dir")
    _report_followers(project_id, statuses)
    return code
end

"""
    _dense_index_kind_name(index) -> String

Best-effort `index_kind` string for a dense index instance, matching the strings
`IndexEngine.create_engine`/the HTTP API's `index_type` accept -- used by `execute_dump`
only when a dataset has no `descriptor.json` to read the *original* creation-time string
back from (CLI-built datasets never write one, see the two-layout watch-out elsewhere in
this file).
"""
_dense_index_kind_name(::SimilaritySearch.SearchGraph) = "searchgraph"
_dense_index_kind_name(::SimilaritySearch.ExhaustiveSearch) = "exhaustive_search"
_dense_index_kind_name(::SimilaritySearch.ParallelExhaustiveSearch) = "parallel_exhaustive_search"

"""
    execute_dump(cmd_args::Dict) -> Int

`dump` (CHANGELOG.md §4.4/§1): exports a dataset as a portable bundle directory --
`dataset.jsonl` (the metadata of every item, deleted or not, one `Persistence.DumpRecord` per
line, CHANGELOG.md §5.16), `project/` (a copy of the project directory, which is the index: `load`
restores it as it is and never rebuilds it from the lines, so the bundle comes back with the
same graph topology and posting lists it had when it was written), `followers/`, and
`manifest.json` (index_kind/distance/edit_correction/max_k -- everything `load` needs to
recreate a descriptor.json at the target). A dataset without items writes an empty
`dataset.jsonl`.
It opens the dataset read-only so it can run concurrently with a live server serving the
same dataset (see `describe`'s identical `read_only=true` reasoning).
"""
function execute_dump(cmd_args::Dict)
    project_id = cmd_args["dataset"]
    workdir = cmd_args["workdir"]
    output_dir = cmd_args["output"]

    if ispath(output_dir)
        println("Error: output '$output_dir' already exists")
        return 1
    end

    handle = _open_handle(project_id, workdir; read_only=true)
    handle === nothing && return 1
    dir = handle.dir
    ds = handle.project
    engine = handle.engine

    records = Persistence.DumpRecord[]
    # The keys of `cf_records` are big-endian `_id`s, so the lines come in the order of `_id`.
    for (_, v) in RocksDB.DBIterator(ds.db; cf=ds.cf_records)
        record = JSON3.read(v, Schema.MetadataRecord)
        push!(records, Persistence.DumpRecord(record, Project.get_raw_meta(ds, record._id),
                                              record._id in engine.deleted_ids))
    end

    descriptor_file = joinpath(dir, "descriptor.json")
    descriptor = isfile(descriptor_file) ? JSON3.read(read(descriptor_file), Dict{String, Any}) : nothing
    follower_names = [_follower_short_name(st.name) for st in SSE.followers(handle) if !st.missing]

    is_text = SSE.payload_kind(engine) === :text
    if descriptor !== nothing
        index_kind = get(descriptor, "index_kind", "unknown")
        distance = get(descriptor, "distance", nothing)
    elseif is_text
        index_kind = wire_kind(engine)
        distance = nothing
    else
        index_kind = wire_kind(engine)
        distance = string(nameof(typeof(SimilaritySearch.distance(engine.backend.index))))
    end
    SSE.close_project!(handle; compact=false)

    mkpath(output_dir)
    Persistence.write_dump_records(joinpath(output_dir, "dataset.jsonl"), records)

    # The followers are copied before the project (CHANGELOG.md §5.12). A server can append to the
    # dataset while this runs. A follower copied first has processed at most the items of the
    # copied project, and processes the rest after `load`. A follower copied after the project
    # could hold items that the copy of the project does not have.
    isempty(follower_names) || mkpath(joinpath(output_dir, "followers"))
    for fname in follower_names
        cp(SSE.follower_dir(dirname(dir), project_id, fname), joinpath(output_dir, "followers", fname))
    end

    # The project directory, copied whole: it is the index (posting lists, document vectors,
    # the adjacency of the graph, the dense vector file). A bundle therefore restores only into
    # an engine that reads this on-disk layout, and `dataset.jsonl` beside it is the part that
    # any program reads, with a JSON parser.
    project_copy = joinpath(output_dir, "project")
    ispath(project_copy) && rm(project_copy; recursive=true)
    cp(dir, project_copy)

    manifest = Dict(
        "project_id" => project_id,
        "index_kind" => index_kind,
        "distance" => distance,
        "edit_correction" => descriptor === nothing ? false : get(descriptor, "edit_correction", false),
        "max_k" => descriptor === nothing ? Server.DEFAULT_MAX_K : Server._descriptor_max_k(descriptor),
        "record_count" => length(records),
        "followers" => follower_names,
        "dumped_at" => string(now(UTC)),
    )
    write(joinpath(output_dir, "manifest.json"), JSON3.write(manifest))

    println("Dump completed: $(length(records)) record(s) and $(length(follower_names)) follower(s) from '$project_id'. Saved to $output_dir")
    return 0
end

"""
    execute_add_token(cmd_args::Dict) -> Int

`add-token`: writes one access token straight into the token database of a working
directory. This is how the first token is created, because a server started with
`[auth] enabled` refuses to run while no token exists, and the administrative endpoint that
creates tokens is itself behind authentication.

It opens the token database for writing, so it cannot run while a server holds that working
directory open. Later tokens are created through `POST /api/v1/admin/tokens`, or with
`similarity-search-ctl add-token`, which is the same endpoint.
"""
function execute_add_token(cmd_args::Dict)
    workdir = cmd_args["workdir"]
    user = cmd_args["user"]
    raw = split(cmd_args["permissions"], ','; keepempty=false)
    permissions = String[strip(p) for p in raw]
    expires_at = get(cmd_args, "expires-at", nothing)

    given = Tokens.normalize_permissions(permissions)
    if isempty(given)
        println("Error: no valid permission in $(repr(cmd_args["permissions"])). Each one is `operation:dataset`, ",
                "with operation in read|write|admin and dataset an id or `*`.")
        return 1
    end
    dropped = length(permissions) - length(given)
    dropped > 0 && println("Ignored $dropped permission(s) that are not `operation:dataset`.")

    isdir(workdir) || mkpath(workdir)
    mgr = try
        Tokens.open_token_manager(workdir)
    catch e
        println("Error: cannot open the token database in $workdir. A running server holds it open; stop it first.")
        println("  ", sprint(showerror, e))
        return 1
    end
    token = try
        Tokens.create_token!(mgr, user, permissions; expires_at=expires_at)
    catch e
        e isa ArgumentError || rethrow()
        println("Error: ", e.msg)
        Tokens.close_token_manager(mgr)
        return 2
    end
    Tokens.close_token_manager(mgr)

    println("Token created for user $(repr(user)) with permissions $(join(given, ", ")):")
    println(token)
    println("This is the only time it is printed.")
    return 0
end

"""
    execute_load(cmd_args::Dict) -> Int

`load` (CHANGELOG.md §4.4/§1): the inverse of `execute_dump` -- always targets the HTTP-layout
directory (`<workdir>/datasets/{id}/`), so a subsequently-(re)started `serve` picks it up
via `Server.reload_datasets!` with no extra step, regardless of which layout the original
dump came from. Refuses to overwrite an existing target (`--dataset` must be a fresh id).
Copies `project/` back as it is (see `execute_dump` for why this is a copy and not a
reconstruction from the lines). When the bundle has `dataset.jsonl`, it then writes the
record and the `meta` of each line over the copy, so that metadata edited in the bundle with a
text editor or `jq` takes effect (CHANGELOG.md §5.16). The lines are read before anything is
copied, so a line that cannot be read leaves no dataset behind. The deleted items come from the
copy of the project, not from `deleted`. A bundle without `dataset.jsonl` is loaded from
`project/` alone. A bundle written before 2026-11-23 has `dataset.avro` instead, which is
ignored, with a warning: the copy of the project holds the same metadata.
"""
function execute_load(cmd_args::Dict)
    bundle_dir = cmd_args["bundle"]
    target_id = cmd_args["dataset"]
    workdir = cmd_args["workdir"]

    manifest_path = joinpath(bundle_dir, "manifest.json")
    project_src = joinpath(bundle_dir, "project")
    lines_src = joinpath(bundle_dir, "dataset.jsonl")
    isfile(manifest_path) || (println("Error: bundle '$bundle_dir' is missing manifest.json"); return 1)
    isdir(project_src) || (println("Error: bundle '$bundle_dir' is missing project/"); return 1)
    manifest = JSON3.read(read(manifest_path), Dict{String, Any})
    records = isfile(lines_src) ? (@cli_engine_call Persistence.read_dump_records(lines_src)) : Persistence.DumpRecord[]
    records isa Cint && return records
    if !isfile(lines_src) && isfile(joinpath(bundle_dir, "dataset.avro"))
        println("Warning: bundle '$bundle_dir' has dataset.avro, written before 2026-11-23; it is ignored, ",
                "and the metadata come from project/")
    end

    target_path = joinpath(workdir, "datasets", target_id)
    if ispath(target_path)
        println("Error: target dataset '$target_id' already exists at $target_path")
        return 1
    end
    followers_src = joinpath(bundle_dir, "followers")
    followers_dst = target_path * ".followers"
    if isdir(followers_src) && ispath(followers_dst)
        println("Error: $followers_dst exists. It holds the followers of an earlier dataset '$target_id'; remove it first")
        return 1
    end
    # The project comes back whole, then the lines are written over it: the copy carries the
    # index, and the lines are what makes metadata edited in `dataset.jsonl` take effect.
    mkpath(dirname(target_path))   # `cp` creates the project directory itself, not its parent
    cp(project_src, target_path)
    isdir(followers_src) && cp(followers_src, followers_dst)

    # Opened through the engine, not `Project.open_project`: a restored project carries the
    # engine's own column families too, and RocksDB refuses an open that does not list every
    # column family the directory holds.
    handle = SSE.open_project(joinpath(workdir, "datasets"), target_id)
    restored = 0
    for rec in records
        meta_record = Schema.MetadataRecord(rec._id, rec.schema_version, rec.doc_id, rec.keywords, rec.refs)
        meta = rec.meta === nothing ? nothing : JSON3.read(codeunits(rec.meta), Dict{String, Any})
        Project.put_metadata!(handle.project, meta_record, meta)
        restored += 1
    end
    # The project lists its followers already. Attaching each one writes the new id of its
    # leader in it. The project and its followers come from the same bundle, so they have the
    # same generation and the followers are not stale (CHANGELOG.md §5.12).
    code = Cint(0)
    if isdir(followers_dst)
        for fname in sort(readdir(followers_dst))
            r = @cli_engine_call SSE.attach_follower!(handle, fname)
            r isa Cint && (code = r)
        end
    end
    statuses = SSE.followers(handle)
    SSE.close_project!(handle)

    descriptor = Dict(
        "id" => target_id,
        "index_kind" => get(manifest, "index_kind", "unknown"),
        "distance" => get(manifest, "distance", nothing),
        "created_at" => string(now(UTC)),
        "edit_correction" => get(manifest, "edit_correction", false),
        "max_k" => Server._descriptor_max_k(manifest),
    )
    write(joinpath(target_path, "descriptor.json"), JSON3.write(descriptor))

    println("Load completed: '$target_id' restored from bundle '$bundle_dir', with $restored line(s) of dataset.jsonl ",
            "written over its metadata. Saved to $target_path")
    _report_followers(target_id, statuses)
    return code
end

# --- Followers (CHANGELOG.md §5.12, issue #5, phase 5b) ----------------------------------------

"""
    follower_request(cmd_args) -> Union{Dict{String,Any}, String}

The body of `POST /api/v1/datasets/{id}/followers` that the arguments of `follower-create`
describe (CHANGELOG.md §5.12), or the message of an invalid combination. `similarity-search` reads it
with `Server.parse_follower_request`, and `similarity-search-ctl` sends it to the server.
"""
function follower_request(cmd_args::AbstractDict)
    d = Dict{String,Any}("name" => cmd_args["name"], "kind" => cmd_args["kind"], "source" => cmd_args["source"])
    kind = cmd_args["kind"]
    # The options of a key source or of a kind of a plugin go in an object with its name
    # (CHANGELOG.md §5.13).
    for (flag, field) in (("source-options", "source"), ("kind-options", "kind"))
        text = get(cmd_args, flag, nothing)
        text === nothing && continue
        Server.is_plugin_name(d[field]) || return "--$flag applies to a $(field == "kind" ? "kind" : "key source") of a plugin, " *
            "--$field Plugin.name; got --$field $(d[field])"
        options = try
            JSON3.read(codeunits(text), Dict{String,Any})
        catch
            return "--$flag must be a JSON object, such as '{\"field\": \"year\"}'; got $(repr(text))"
        end
        haskey(options, "name") && return "--$flag cannot hold 'name': the name is --$field"
        d[field] = merge(Dict{String,Any}("name" => d[field]), options)
    end
    if get(cmd_args, "non-disjoint", false)
        kind == "partition" || return "--non-disjoint applies to a partition follower"
        d["disjoint"] = false
    end
    graph_min = get(cmd_args, "graph-min", nothing)
    if graph_min !== nothing
        kind == "partition" || return "--graph-min applies to a partition follower"
        d["graph_min"] = graph_min
    end
    distance = get(cmd_args, "distance", nothing)
    if distance !== nothing
        kind == "keyword" || return "--distance applies to a keyword follower"
        d["distance"] = distance
    end
    policy = get(cmd_args, "policy", "on_index")
    seconds, items = get(cmd_args, "periodic-seconds", nothing), get(cmd_args, "periodic-items", nothing)
    if policy == "periodic"
        seconds === nothing && items === nothing &&
            return "--policy periodic needs --periodic-seconds, --periodic-items, or both"
        p = Dict{String,Any}()
        seconds === nothing || (p["seconds"] = seconds)
        items === nothing || (p["items"] = items)
        d["policy"] = Dict{String,Any}("periodic" => p)
    else
        seconds === nothing && items === nothing ||
            return "--periodic-seconds and --periodic-items apply to --policy periodic"
        d["policy"] = policy
    end
    return d
end

"""
    _open_writable(project_id, workdir) -> Union{SSE.EmbeddedEngine, Cint}

Opens a dataset for writing, for a command that changes it or its followers. Returns the exit
code of the failure: `1` when the dataset does not exist, and `5` when another process holds
it open, such as a server that has it loaded. RocksDB allows one writer per database.
"""
function _open_writable(project_id::String, workdir::String)
    handle = try
        _open_handle(project_id, workdir)
    catch e
        e isa SSE.EngineError && rethrow()
        occursin("lock", lowercase(sprint(showerror, e))) || rethrow()
        println("Error: the dataset '$project_id' is open in another process, such as a running server. ",
                "With a server, run the same command with similarity-search-ctl. Otherwise, stop the other process first.")
        return Cint(5)
    end
    return handle === nothing ? Cint(1) : handle
end

"""
    _with_dataset(f, cmd_args; read_only) -> Cint

Opens the dataset `cmd_args["dataset"]`, calls `f(handle)`, and closes it. An engine error
inside `f` prints its message and exits with its code (`cli_exit_code`). `f` returns `nothing`,
or the exit code of a failure that it printed itself.
"""
function _with_dataset(f::Function, cmd_args::AbstractDict; read_only::Bool)
    project_id, workdir = cmd_args["dataset"], cmd_args["workdir"]
    handle = @cli_engine_call(read_only ? _open_handle(project_id, workdir; read_only=true) :
                              _open_writable(project_id, workdir))
    handle === nothing && return Cint(1)
    handle isa Cint && return handle
    try
        r = @cli_engine_call f(handle)
        r isa Cint && return r
        return Cint(0)
    finally
        SSE.close_project!(handle; compact=false)
    end
end

_print_follower_status(s::SSE.FollowerStatus) = println(JSON3.write(Server.follower_status_dict(s)))

# A command on one follower takes its name, not `name=value`.
function _plain_follower_name(cmd_args::AbstractDict)
    name = cmd_args["name"]
    occursin('=', name) || return name
    println("Error: --name is the name of the follower, without '=value'; got $(repr(name))")
    return Cint(2)
end

"""
    execute_follower_create(cmd_args) -> Cint

`follower-create` (CHANGELOG.md §5.12): creates a follower of the dataset, processes every indexed
item, and prints its status. A follower with `--policy periodic` is processed by a running
server; this command line processes it with `follower-sync`.
"""
function execute_follower_create(cmd_args::AbstractDict)
    req = follower_request(cmd_args)
    req isa String && (println("Error: ", req); return Cint(2))
    parsed = @cli_engine_call Server.parse_follower_request(req)
    parsed isa Cint && return parsed
    parsed isa String && (println("Error: ", parsed); return Cint(2))
    _with_dataset(cmd_args; read_only=false) do handle
        f = SSE.create_follower(handle, parsed.name, parsed.spec; policy=parsed.policy)
        st = SSE.follower_status(f)
        println("Created the follower '$(st.name)': $(st.position) item(s) processed.")
        _print_follower_status(st)
        nothing
    end
end

"""
    execute_follower_list(cmd_args) -> Cint

`follower-list` (CHANGELOG.md §5.12): prints the status of every follower of the dataset, as one JSON
object. It opens the dataset read-only, so it runs while a server holds the dataset.
"""
function execute_follower_list(cmd_args::AbstractDict)
    _with_dataset(cmd_args; read_only=true) do handle
        println(JSON3.write(Dict("id" => cmd_args["dataset"],
                                 "followers" => [Server.follower_status_dict(s) for s in SSE.followers(handle)])))
        nothing
    end
end

"""
    execute_follower_sync(cmd_args) -> Cint

`follower-sync` (CHANGELOG.md §5.12): processes the indexed items of the leader that the follower has
not processed yet, and prints its status.
"""
function execute_follower_sync(cmd_args::AbstractDict)
    name = _plain_follower_name(cmd_args)
    name isa Cint && return name
    _with_dataset(cmd_args; read_only=false) do handle
        f = SSE.open_follower(handle, name)
        n = SSE.sync!(f)
        st = SSE.follower_status(f)
        println("Synchronized the follower '$(st.name)': $n item(s) processed; its position is $(st.position).")
        _print_follower_status(st)
        nothing
    end
end

"""
    execute_follower_rebuild(cmd_args) -> Cint

`follower-rebuild` (CHANGELOG.md §5.12): builds the follower again from every indexed item of its
leader, with the spec and the policy it was created with. A follower is stale after its leader is
rebuilt, and this command is what makes it usable again.
"""
function execute_follower_rebuild(cmd_args::AbstractDict)
    name = _plain_follower_name(cmd_args)
    name isa Cint && return name
    _with_dataset(cmd_args; read_only=false) do handle
        st = SSE.follower_status(SSE.rebuild_follower!(handle, name))
        println("Rebuilt the follower '$(st.name)': $(st.position) item(s) processed.")
        _print_follower_status(st)
        nothing
    end
end

"""
    execute_follower_delete(cmd_args) -> Cint

`follower-delete` (CHANGELOG.md §5.12): deletes the directory of the follower and its name in the list
of the leader.
"""
function execute_follower_delete(cmd_args::AbstractDict)
    name = _plain_follower_name(cmd_args)
    name isa Cint && return name
    _with_dataset(cmd_args; read_only=false) do handle
        SSE.delete_follower!(handle, name)
        println("Deleted the follower '$(cmd_args["dataset"])/$name'.")
        nothing
    end
end

"""
    execute_migrate_partitions(cmd_args) -> Cint

`migrate-partitions` (CHANGELOG.md §5.11): moves the partitions of a dataset created with
`--partition-by` to a partition follower named `--name`. `--disjoint` is required: the command
does not choose between disjoint and non-disjoint partitions.
"""
function execute_migrate_partitions(cmd_args::AbstractDict)
    _with_dataset(cmd_args; read_only=false) do handle
        f = SSE.migrate_partitions!(handle, cmd_args["name"]; disjoint=cmd_args["disjoint"] == "true")
        st = SSE.follower_status(f)
        println("Migrated the partitions of '$(cmd_args["dataset"])' to the follower '$(st.name)'. ",
                "Search one partition with searchbatch --follower $(cmd_args["name"])=<value>.")
        _print_follower_status(st)
        nothing
    end
end

"""
    execute_postings(cmd_args) -> Cint

`postings` (CHANGELOG.md §5.11): prints the ids of the items that hold one key, in ascending order,
as one JSON object. The key is a token of a text dataset, a dimension of a sparse dataset, or,
with `--follower`, a key of a keyword follower. `--limit N` returns the first `N` ids. It opens
the dataset read-only, so it runs while a server holds the dataset.
"""
function execute_postings(cmd_args::AbstractDict)
    out_file = get(cmd_args, "output", nothing)
    _with_dataset(cmd_args; read_only=true) do handle
        key, limit = cmd_args["key"], get(cmd_args, "limit", nothing)
        fname = get(cmd_args, "follower", nothing)
        ids = if fname === nothing
            k = SSE.payload_kind(handle.engine) === :sparse ? tryparse(Int, key) : key
            k === nothing && (println("Error: the key of a sparse dataset is a dimension, an integer; got $(repr(key))");
                              return Cint(2))
            SSE.posting_list(handle, k; limit)
        else
            SSE.posting_list(SSE.open_follower(handle, fname), key; limit)
        end
        body = Dict{String,Any}("id" => cmd_args["dataset"], "key" => key, "ids" => ids)
        fname === nothing || (body["follower"] = fname)
        if out_file === nothing
            println(JSON3.write(body))
        else
            open(io -> println(io, JSON3.write(body)), out_file, "w")
            println("$(length(ids)) id(s) saved to $out_file")
        end
        nothing
    end
end

# --- Plugins (CHANGELOG.md §5.13, issue #5, phase 6b) ------------------------------------------

"""
    needs_plugins(cmd_name, cmd_args) -> Bool

Whether the command opens a follower, whose spec or key source can belong to a plugin:
`follower-create`, `follower-sync`, `follower-rebuild`, `rebuild --rebuild-followers`, and
`searchbatch` and `postings` with `--follower`. Loading a plugin compiles its code, so the other
commands do not load them.
"""
function needs_plugins(cmd_name::AbstractString, cmd_args::AbstractDict)
    cmd_name in ("follower-create", "follower-sync", "follower-rebuild") && return true
    cmd_name == "rebuild" && return get(cmd_args, "rebuild-followers", false) === true
    cmd_name in ("searchbatch", "postings") && return get(cmd_args, "follower", nothing) !== nothing
    return false
end

"""
    run_cli_command(config_path, cmd_name, cmd_args) -> Int

Runs one command of `similarity-search`. A command that opens a follower ([`needs_plugins`](@ref))
first loads the plugins of the `[plugins]` section of `config_path`, as `serve` does. A file that
does not exist is not an error: a follower that needs a plugin then raises `PluginNotLoaded`,
and the message names the file. A plugin that cannot be loaded exits with code 1.

The command runs with `Base.invokelatest`, because the code of a plugin is defined after this
function was compiled.
"""
function run_cli_command(config_path::AbstractString, cmd_name::String, cmd_args::AbstractDict)
    CLI_CONFIG[] = String(config_path)
    if needs_plugins(cmd_name, cmd_args) && isfile(config_path)
        try
            Server.load_plugins!(TOML.parsefile(config_path); base_dir=dirname(abspath(config_path)))
        catch e
            e isa Union{Server.PluginLoadError,TOML.ParserError} || rethrow()
            println("Error: ", sprint(showerror, e))
            println("Correct or remove the plugin in ", config_path, ", or run the command with another --config.")
            return 1
        end
    end
    return Base.invokelatest(dispatch_command, cmd_name, cmd_args)
end

"""
    dispatch_command(cmd_name::String, cmd_args::AbstractDict) -> Int

Runs one of the data-operation subcommands (`build` through `load`) given its
already-assembled `cmd_args` (the same `Dict{String,Any}` shape `parse_commandline`
produces for that subcommand). Shared by `main`'s direct non-interactive dispatch and
`run_interactive`'s confirm-then-run step (`interactive.jl`), so there is exactly one
command-name → `execute_*` mapping regardless of how the arguments were collected.
"""
function dispatch_command(cmd_name::String, cmd_args::AbstractDict)
    cmd_name == "build" && return execute_build(cmd_args)
    cmd_name == "searchbatch" && return execute_searchbatch(cmd_args)
    cmd_name == "allknn" && return execute_allknn(cmd_args)
    cmd_name == "fft" && return execute_fft(cmd_args)
    cmd_name == "neardup" && return execute_neardup(cmd_args)
    cmd_name == "hsp" && return execute_hsp(cmd_args)
    cmd_name == "closestpair" && return execute_closestpair(cmd_args)
    cmd_name == "describe" && return execute_describe(cmd_args)
    cmd_name == "rebuild" && return execute_rebuild(cmd_args)
    cmd_name == "dump" && return execute_dump(cmd_args)
    cmd_name == "load" && return execute_load(cmd_args)
    cmd_name == "add-token" && return execute_add_token(cmd_args)
    cmd_name == "follower-create" && return execute_follower_create(cmd_args)
    cmd_name == "follower-list" && return execute_follower_list(cmd_args)
    cmd_name == "follower-sync" && return execute_follower_sync(cmd_args)
    cmd_name == "follower-rebuild" && return execute_follower_rebuild(cmd_args)
    cmd_name == "follower-delete" && return execute_follower_delete(cmd_args)
    cmd_name == "migrate-partitions" && return execute_migrate_partitions(cmd_args)
    cmd_name == "postings" && return execute_postings(cmd_args)
    error("dispatch_command: unknown command '$cmd_name'")
end
