# CLI operations
using JSON3
using TOML
using HTTP
using Dates

function execute_build(cmd_args::Dict)
    project_id = cmd_args["dataset"]
    input_file = cmd_args["input"]
    index_kind = cmd_args["index-kind"]
    distance = cmd_args["distance"]
    workdir = cmd_args["workdir"]

    println("Building index '$index_kind' for dataset '$project_id'")

    engine_type, backend_type = Server.parse_index_kind(index_kind)
    handle = SSE.create_project(workdir, project_id;
        engine=engine_type, backend=backend_type,
        distance=Server.parse_distance(distance),
        textmodel=Server.default_textmodel(engine_type))

    kind = SSE.payload_kind(handle.engine)
    items = SSE.AbstractItem[]
    for line in eachline(input_file)
        isempty(strip(line)) && continue
        it = Server._typed_item(kind, JSON3.read(line, Dict{String, Any}))
        it === nothing || push!(items, it)
    end

    inserted = isempty(items) ? 0 : SSE.append_items!(handle, items)
    inserted > 0 && SSE.index!(handle)
    # Nothing else to write: the project persists itself, which is also why `search`,
    # `describe` and every job below reopen it instead of reading a separate file.
    SSE.close_project!(handle)

    println("Successfully built index with $inserted items. Saved to $(joinpath(workdir, project_id))")
    return 0
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
            cli_exit_code(e)
        end
    end
end

"""
    _open_handle(project_id, workdir; read_only=false) -> Union{Nothing, SSE.EmbeddedEngine}

Reopens a dataset by id, whichever of this codebase's two layouts it lives under --
`<workdir>/datasets/<id>` for one created over HTTP, `<workdir>/<id>` for one built by the
CLI. Prints an error and returns `nothing` when neither exists.

This replaced a snapshot file per dataset, written by `build` and by every HTTP append, and
read by each command that operated on a whole dataset. The engine persists its own index, so
that file was a second and older copy of state that already had a place to live, and a job
that ran before a snapshot had been written failed with "snapshot not found". The functions
that wrote and read it were removed on 2026-09-13, together with the `JLD2` dependency.
"""
function _open_handle(project_id::String, workdir::String; read_only::Bool=false)
    for root in (joinpath(workdir, "datasets"), workdir)
        isdir(joinpath(root, project_id)) || continue
        return SSE.open_project(root, project_id; read_only)
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
        item = JSON3.read(line, Dict{String, Any})
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
    handle = _open_handle(project_id, workdir; read_only=true)
    handle === nothing && return 1
    kind = SSE.payload_kind(handle.engine)

    open(out_file, "w") do out_io
        for line in eachline(queries_file)
            isempty(strip(line)) && continue
            item = JSON3.read(line, Dict{String, Any})

            hits = if kind === :text
                haskey(item, "text") || continue
                @cli_engine_call SSE.ftsearch(handle, item["text"], k)
            else
                haskey(item, "vector") || continue
                @cli_engine_call SSE.search(handle, convert(Vector{Float32}, item["vector"]), k)
            end
            hits isa Cint && return hits

            out_obj = Dict("ids" => Int32[h._id for h in hits], "dists" => Float32[h.distance for h in hits])
            println(out_io, JSON3.write(out_obj))
        end
    end
    SSE.close_project!(handle)

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
about their data" PLAN.md §1's `describe` note asks for. `nothing` if the index has fewer
than 2 items (no pair to measure) or isn't dense.
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
    queries = [Vector{Float32}(SimilaritySearch.database(engine.backend.index, i)) for i in ids]
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

    if is_text
        # The vocabulary and its out-of-vocabulary rates belong to the engine, which fitted
        # it. `sample=0` asks for every document rather than the bounded sample the HTTP
        # endpoint takes: this command is offline and nobody is waiting on a connection, so it
        # answers exactly (PLAN.md §5's drift signal, see `execute_rebuild`).
        desc["vocab"] = SSE.vocabulary_report(handle; scan=true, sample=0)
    else
        desc["distance"] = string(nameof(typeof(SimilaritySearch.distance(engine.backend.index))))
        desc["distance_stats"] = _dense_distance_stats(handle)
        bs = IndexEngine.current_beamsearch(engine)
        desc["beamsearch_baseline"] = bs === nothing ? nothing :
            Dict("bsize" => Int(bs.bsize), "delta" => Float64(bs.Δ), "maxvisits" => Int(bs.maxvisits))
    end

    Project.close_project(ds)

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

"""
    _live_items(ds, engine) -> (all_ids::Vector{Int}, live::Vector{Tuple{Int,Dict}})

Scans `ds`'s metadata column family once, returning every doc_id currently stored
(`all_ids`, needed to wipe stale rows during a rebuild) alongside the reconstructed raw
item (`declared_fields` merged with the decoded `extra` JSON -- see `Schema.MetadataRecord`)
for every *live* (non-tombstoned) one, sorted by original doc_id.
"""
function _live_items(ds, engine)
    all_ids = Int[]
    live = Tuple{Int, Dict{String, Any}}[]
    for (_, v) in RocksDB.DBIterator(ds.db; cf=ds.cf_records)
        record = JSON3.read(String(copy(v)), Schema.MetadataRecord)
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
        
        payload = IndexEngine.stored_payload(engine, record._id)
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

`rebuild` (PLAN.md §1): purges soft-deleted (tombstoned) documents and, for a text index,
retrains the `Vocabulary` from scratch against the dataset's current live corpus -- the
only way to recover from vocabulary drift (PLAN.md §1's note: `TextSearch.jl`'s
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
rather than the incremental one-vertex-at-a-time bias `add_item!` reintroduces -- PLAN.md
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
    index_kind = wire_kind(engine)
    distance = kind === :text ? nothing : SimilaritySearch.distance(engine.backend.index)

    all_ids, live = _live_items(handle.project, engine)
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

    engine_type, backend_type = Server.parse_index_kind(index_kind)
    fresh = SSE.create_project(root, name;
        engine=engine_type, backend=backend_type, distance=distance,
        textmodel=Server.default_textmodel(engine_type))
    reindexed = isempty(items) ? 0 : SSE.append_items!(fresh, items)
    reindexed > 0 && SSE.index!(fresh)
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
    return 0
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

`dump` (PLAN.md §4.4/§1): exports a dataset as a portable bundle directory --
`dataset.avro` (every metadata record, deleted or not, via `Persistence.DumpRecord`),
`project/` (a copy of the project directory, which is the index: `load` restores it as it
is and never rebuilds it from the Avro rows, so the bundle comes back with the same graph
topology and posting lists it had when it was written), and `manifest.json` (index_kind/distance/
meta_schema/join_group/holds_metadata/key -- everything `load` needs to recreate a
descriptor.json at the target, on top of the dataset it opens read-only so it can run
concurrently with a live server serving the same dataset (see `describe`'s identical
`read_only=true` reasoning).
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
    for (_, v) in RocksDB.DBIterator(ds.db; cf=ds.cf_records)
        record = JSON3.read(String(copy(v)), Schema.MetadataRecord)
        meta_json = Project.get_raw_meta(ds, record._id)
        push!(records, Persistence.DumpRecord(
            record,
            meta_json,
            record._id in engine.deleted_ids,
        ))
    end

    if isempty(records)
        println("Error: dataset '$project_id' has no records to dump")
        Project.close_project(ds)
        return 1
    end

    descriptor_file = joinpath(dir, "descriptor.json")
    descriptor = isfile(descriptor_file) ? JSON3.read(read(descriptor_file, String), Dict{String, Any}) : nothing

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
    Project.close_project(ds)

    mkpath(output_dir)
    Persistence.write_dump_records(joinpath(output_dir, "dataset.avro"), records)

    # The project directory, copied whole: it is the index (posting lists, document vectors,
    # the adjacency of the graph, the dense vector file). A bundle therefore restores only into
    # an engine that reads this on-disk layout, and `dataset.avro` beside it remains the
    # portable half, independent of the engine.
    project_copy = joinpath(output_dir, "project")
    ispath(project_copy) && rm(project_copy; recursive=true)
    cp(dir, project_copy)

    manifest = Dict(
        "project_id" => project_id,
        "index_kind" => index_kind,
        "distance" => distance,
        "join_group" => descriptor === nothing ? nothing : get(descriptor, "join_group", nothing),
        "holds_metadata" => descriptor === nothing ? false : get(descriptor, "holds_metadata", false),
        "key" => descriptor === nothing ? nothing : get(descriptor, "key", nothing),
        "edit_correction" => descriptor === nothing ? false : get(descriptor, "edit_correction", false),
        "record_count" => length(records),
        "dumped_at" => string(now(UTC)),
    )
    write(joinpath(output_dir, "manifest.json"), JSON3.write(manifest))

    println("Dump completed: $(length(records)) record(s) from '$project_id'. Saved to $output_dir")
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

`load` (PLAN.md §4.4/§1): the inverse of `execute_dump` -- always targets the HTTP-layout
directory (`<workdir>/datasets/{id}/`), so a subsequently-(re)started `serve` picks it up
via `Server.reload_datasets!` with no extra step, regardless of which layout the original
dump came from. Refuses to overwrite an existing target (`--dataset` must be a fresh id).
Copies `project/` back as it is (see `execute_dump` for why this is a copy and not a
reconstruction from the rows) and replays every `dataset.avro` row into
fresh `Schema.MetadataRecord`s under its original `doc_id` -- both `declared_json` and
`extra_json` were captured separately at dump time (not pre-merged), so this reconstructs
the exact original `MetadataRecord`, not a re-derived approximation of one.
"""
function execute_load(cmd_args::Dict)
    bundle_dir = cmd_args["bundle"]
    target_id = cmd_args["dataset"]
    workdir = cmd_args["workdir"]

    manifest_path = joinpath(bundle_dir, "manifest.json")
    project_src = joinpath(bundle_dir, "project")
    avro_src = joinpath(bundle_dir, "dataset.avro")
    isfile(manifest_path) || (println("Error: bundle '$bundle_dir' is missing manifest.json"); return 1)
    isfile(avro_src) || (println("Error: bundle '$bundle_dir' is missing dataset.avro"); return 1)
    isdir(project_src) || (println("Error: bundle '$bundle_dir' is missing project/"); return 1)
    manifest = JSON3.read(read(manifest_path, String), Dict{String, Any})

    target_path = joinpath(workdir, "datasets", target_id)
    if ispath(target_path)
        println("Error: target dataset '$target_id' already exists at $target_path")
        return 1
    end
    # The project comes back whole, then the Avro records are replayed over it: the copy
    # carries the index, and the replay is what makes a hand-edited `dataset.avro` (the
    # documented way to fix metadata in a bundle) take effect.
    mkpath(dirname(target_path))   # `cp` creates the project directory itself, not its parent
    cp(project_src, target_path)

    # Opened through the engine, not `Project.open_project`: a restored project carries the
    # engine's own column families too, and RocksDB refuses an open that does not list every
    # column family the directory holds.
    handle = SSE.open_project(joinpath(workdir, "datasets"), target_id)
    restored = 0
    for rec in Persistence.read_dump_records(avro_src)
        meta_record = Schema.MetadataRecord(rec._id, rec.schema_version, rec.doc_id, rec.keywords, rec.refs)
        meta = rec.meta_json === nothing ? nothing : JSON3.read(rec.meta_json, Dict{String, Any})
        Project.put_metadata!(handle.project, meta_record, meta)
        restored += 1
    end
    SSE.close_project!(handle)

    descriptor = Dict(
        "id" => target_id,
        "index_kind" => get(manifest, "index_kind", "unknown"),
        "distance" => get(manifest, "distance", nothing),
        "created_at" => string(now(UTC)),
        "join_group" => get(manifest, "join_group", nothing),
        "holds_metadata" => get(manifest, "holds_metadata", false),
        "key" => get(manifest, "key", nothing),
        "edit_correction" => get(manifest, "edit_correction", false),
    )
    write(joinpath(target_path, "descriptor.json"), JSON3.write(descriptor))

    println("Load completed: $restored record(s) restored into '$target_id' from bundle '$bundle_dir'. Saved to $target_path")
    return 0
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
    error("dispatch_command: unknown command '$cmd_name'")
end
