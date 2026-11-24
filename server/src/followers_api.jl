# Followers over HTTP (PLAN.md §5.12, issue #5, phase 5a). Included in `module Server`.
#
# A follower is processed in this process, because a job is a read-only subprocess. `OnIndex`
# and `OnWrite` run inside the request that writes the leader (`append` calls `index!`), and
# `Periodic` runs in the task that `start_follower_task!` starts.

"The default of `[followers] check_seconds`: how often `serve` looks for periodic followers to process."
const DEFAULT_FOLLOWER_CHECK_SECONDS = 10.0

_record_follower_error!(app, canonical::String, e) =
    lock(() -> (app.followers.errors[canonical] = sprint(showerror, e)), app.followers.lock)
_clear_follower_error!(app, canonical::String) =
    lock(() -> delete!(app.followers.errors, canonical), app.followers.lock)

"""
    open_dataset_followers!(app, id, handle) -> Nothing

Opens every follower that the leader `id` lists (PLAN.md §5.12). A follower that cannot be
opened, because it is stale, needs a plugin that is not loaded, or has lost its directory, stays
closed, and its error is kept for its status. The leader stays available.
"""
function open_dataset_followers!(app, id::String, handle::SSE.EmbeddedEngine)
    for st in SSE.followers(handle)
        name = String(last(split(st.name, '/'; limit=2)))
        try
            SSE.open_follower(handle, name)
            _clear_follower_error!(app, st.name)
        catch e
            e isa SSE.EngineError || rethrow()
            _record_follower_error!(app, st.name, e)
            @warn "the follower $(st.name) was not opened" exception=e
        end
    end
    return nothing
end

# --- Wire format -----------------------------------------------------------------------

const _FOLLOWER_FIELDS = ("name", "kind", "source", "disjoint", "graph_min", "distance", "policy")

const _SET_DISTANCES = Dict(
    "jaccard" => () -> SimilaritySearch.Dist.Sets.Jaccard(),
    "dice" => () -> SimilaritySearch.Dist.Sets.Dice(),
    "intersection" => () -> SimilaritySearch.Dist.Sets.Intersection(),
    "cosine_set" => () -> SimilaritySearch.Dist.Sets.CosineSet())

"""
    parse_key_source(source) -> Union{SSE.KeySource, String}

The key source that a wire name selects: `keywords`, `refs`, `doc_id`, `meta:<field>`, or
`Plugin.name` of a plugin (PLAN.md §5.13). A key source of a plugin with options is an object,
`{"name": "Plugin.name", ...}`. A `String` result is the message of a `400`. A plugin that is not
loaded raises `PluginNotLoaded`.
"""
function parse_key_source(source)
    if source isa AbstractDict
        name = get(source, "name", nothing)
        name isa AbstractString || return "'source' as an object needs 'name', the wire name of a key source of a plugin (Plugin.name)"
        options = Dict{String,Any}(String(k) => v for (k, v) in source if k != "name")
        is_plugin_name(name) && return plugin_key_source(name, options)
        isempty(options) || return "only a key source of a plugin (Plugin.name) takes options; got $(repr(name)) with options"
        return parse_key_source(String(name))
    end
    source isa AbstractString || return "'source' must be a string (keywords, refs, doc_id, meta:<field>, or Plugin.name) " *
        "or an object with 'name' and the options of a key source of a plugin"
    source == "keywords" && return SSE.Keywords()
    source == "refs" && return SSE.Refs()
    source == "doc_id" && return SSE.DocId()
    if startswith(source, "meta:")
        field = source[6:end]
        isempty(field) && return "'source' meta:<field> needs the name of a field of meta"
        return SSE.MetaValue(field)
    end
    is_plugin_name(source) && return plugin_key_source(source, Dict{String,Any}())
    return "unknown 'source' $(repr(source)); use keywords, refs, doc_id, meta:<field>, or Plugin.name of a plugin"
end

"""
    parse_sync_policy(policy) -> Union{SSE.SyncPolicy, String}

`on_index`, `on_write`, `manual`, or `{"periodic": {"seconds": 60, "items": 1000}}`. A `String`
result is the message of a `400`.
"""
function parse_sync_policy(policy)
    policy === nothing && return SSE.OnIndex()
    policy == "on_index" && return SSE.OnIndex()
    policy == "on_write" && return SSE.OnWrite()
    policy == "manual" && return SSE.Manual()
    if policy isa AbstractDict && length(policy) == 1 && haskey(policy, "periodic") && policy["periodic"] isa AbstractDict
        p = policy["periodic"]
        seconds, items = get(p, "seconds", nothing), get(p, "items", nothing)
        seconds === nothing || seconds isa Real || return "'periodic.seconds' must be a number"
        items === nothing || items isa Integer || return "'periodic.items' must be an integer"
        return try
            SSE.Periodic(; seconds, items)
        catch e
            e isa SSE.EngineError || rethrow()
            sprint(showerror, e)
        end
    end
    return "unknown 'policy' $(repr(policy)); use on_index, on_write, manual, or {\"periodic\": {\"seconds\": ..., \"items\": ...}}"
end

"""
    parse_follower_request(data) -> Union{NamedTuple, String}

The name, the spec and the policy of `POST /api/v1/datasets/{id}/followers` (PLAN.md §5.12):
`(name, spec, policy)`, or the message of a `400`. `kind` is `partition` (with `disjoint` and
`graph_min`), `keyword` (with `distance`), or a kind of a plugin (PLAN.md §5.13): `Plugin.name`,
or `{"name": "Plugin.name", ...}` with its options. A plugin that is not loaded raises
`PluginNotLoaded`.
"""
function parse_follower_request(data::AbstractDict)
    unknown = [k for k in keys(data) if !(k in _FOLLOWER_FIELDS)]
    isempty(unknown) || return "unknown field(s) $(join(repr.(sort(unknown)), ", ")); a follower takes $(join(_FOLLOWER_FIELDS, ", "))"
    name = get(data, "name", nothing)
    name isa AbstractString || return "'name' must be the name of the follower (a string)"
    source = parse_key_source(get(data, "source", nothing))
    source isa String && return source
    policy = parse_sync_policy(get(data, "policy", nothing))
    policy isa String && return policy
    kind = get(data, "kind", nothing)
    kind_name = kind isa AbstractDict ? get(kind, "name", nothing) : kind
    if kind_name isa AbstractString && is_plugin_name(kind_name)
        for field in ("disjoint", "graph_min", "distance")
            haskey(data, field) && return "'$field' applies to a partition or a keyword follower; the options of a " *
                "kind of a plugin go in the object 'kind'"
        end
        options = kind isa AbstractDict ? Dict{String,Any}(String(k) => v for (k, v) in kind if k != "name") : Dict{String,Any}()
        spec = plugin_follower_spec(kind_name, source, options)
        spec isa String && return spec
        return (name=String(name), spec, policy)
    end
    kind isa AbstractDict && return "'kind' as an object names a kind of a plugin, {\"name\": \"Plugin.name\", ...}; " *
        "partition and keyword are strings"
    if kind == "partition"
        haskey(data, "distance") && return "'distance' applies to a keyword follower"
        disjoint = get(data, "disjoint", true)
        disjoint isa Bool || return "'disjoint' must be true or false"
        graph_min = get(data, "graph_min", nothing)
        graph_min === nothing || graph_min isa Integer || return "'graph_min' must be an integer"
        return (name=String(name), spec=SSE.PartitionFollower(source; disjoint, graph_min), policy)
    elseif kind == "keyword"
        (haskey(data, "disjoint") || haskey(data, "graph_min")) &&
            return "'disjoint' and 'graph_min' apply to a partition follower"
        dname = get(data, "distance", "jaccard")
        dist = get(_SET_DISTANCES, dname isa AbstractString ? lowercase(dname) : "", nothing)
        dist === nothing && return "unknown 'distance' $(repr(dname)); use $(join(sort(collect(keys(_SET_DISTANCES))), ", "))"
        return (name=String(name), spec=SSE.KeywordFollower(source; distance=dist()), policy)
    end
    return "'kind' must be partition, keyword, or Plugin.name of a plugin"
end

"""
    follower_status_dict(s::SSE.FollowerStatus) -> Dict
    follower_status_dict(app, s::SSE.FollowerStatus) -> Dict

The wire form of a follower status: the fields of `SSE.FollowerStatus` by name, and `stats` as an
object. `error` is present when the follower has one: the last `sync!` of the engine that failed.
With `app`, a follower that `serve` could not open or process also has `error`. The command lines
print the form without `app`.
"""
function follower_status_dict(s::SSE.FollowerStatus)
    d = Dict{String,Any}("name" => s.name, "leader" => s.leader, "kind" => s.kind, "source" => s.source,
        "policy" => s.policy, "plugin" => s.plugin, "position" => s.position, "leader_indexed" => s.leader_indexed,
        "lag" => s.lag, "synchronized" => s.synchronized, "stale" => s.stale, "missing" => s.missing,
        "open" => s.open, "stats" => s.stats === nothing ? nothing : Dict(string(k) => v for (k, v) in pairs(s.stats)))
    s.error === nothing || (d["error"] = s.error)
    return d
end

function follower_status_dict(app, s::SSE.FollowerStatus)
    d = follower_status_dict(s)
    haskey(d, "error") && return d
    err = lock(() -> get(app.followers.errors, s.name, nothing), app.followers.lock)
    err === nothing || (d["error"] = err)
    return d
end

dataset_follower_statuses(app, handle::SSE.EmbeddedEngine) =
    [follower_status_dict(app, s) for s in SSE.followers(handle)]

# --- Handlers ------------------------------------------------------------------------

# The handle of a loaded leader, or the response of a dataset that is not loaded.
function _leader_or_404(app, id::String)
    valid_project_id(id) || return json_response(400, Dict("error" => "invalid dataset id"))
    handle = lock(() -> get(app.handles, id, nothing), app.lock)
    handle === nothing && return json_response(404, Dict("error" => "dataset_not_found"))
    return handle
end

_follower_param(f::AbstractString) = HTTP.URIs.unescapeuri(String(f))
_follower_base(f::String) = (i = findfirst('=', f); i === nothing ? f : f[1:prevind(f, i)])

function _engine_call(f::Function)
    try
        f()
    catch e
        e isa SSE.EngineError || rethrow()
        engine_error_response(e)
    end
end

_body(req::HTTP.Request) = (b = String(req.body); isempty(b) ? Dict{String,Any}() : JSON3.read(codeunits(b), Dict{String,Any}))

"`POST /api/v1/datasets/{id}/followers`: creates a follower (PLAN.md §5.12)."
function handle_create_follower(req::HTTP.Request, app, id::String)
    handle = _leader_or_404(app, id)
    handle isa HTTP.Response && return handle
    _engine_call() do
        # A name of a plugin that is not loaded raises `PluginNotLoaded` here.
        parsed = parse_follower_request(_body(req))
        parsed isa String && return json_response(400, Dict("error" => parsed))
        f = lock(app.lock) do
            SSE.create_follower(handle, parsed.name, parsed.spec; policy=parsed.policy)
        end
        json_response(201, merge(Dict{String,Any}("status" => "created"), follower_status_dict(app, SSE.follower_status(f))))
    end
end

"`GET /api/v1/datasets/{id}/followers`: the status of every follower of a leader."
function handle_list_followers(req::HTTP.Request, app, id::String)
    handle = _leader_or_404(app, id)
    handle isa HTTP.Response && return handle
    _engine_call(() -> json_response(200, Dict("id" => id, "followers" => dataset_follower_statuses(app, handle))))
end

"`GET /api/v1/datasets/{id}/followers/{f}`: the status of one follower."
function handle_get_follower(req::HTTP.Request, app, id::String, f::String)
    handle = _leader_or_404(app, id)
    handle isa HTTP.Response && return handle
    name = _follower_base(_follower_param(f))
    _engine_call() do
        statuses = SSE.followers(handle)
        i = findfirst(s -> s.name == "$id/$name", statuses)
        i === nothing && throw(SSE.UnknownFollower(name, "the dataset '$id' has no follower named '$name'"))
        json_response(200, follower_status_dict(app, statuses[i]))
    end
end

"`DELETE /api/v1/datasets/{id}/followers/{f}`: closes and deletes a follower."
function handle_delete_follower(req::HTTP.Request, app, id::String, f::String)
    handle = _leader_or_404(app, id)
    handle isa HTTP.Response && return handle
    name = _follower_base(_follower_param(f))
    _engine_call() do
        lock(app.lock) do
            SSE.close_follower!(handle, name)
            SSE.delete_follower!(handle, name)
        end
        _clear_follower_error!(app, "$id/$name")
        json_response(200, Dict("status" => "deleted", "id" => id, "follower" => name))
    end
end

"`POST /api/v1/datasets/{id}/followers/{f}/sync`: processes the new items of the leader now."
function handle_sync_follower(req::HTTP.Request, app, id::String, f::String)
    handle = _leader_or_404(app, id)
    handle isa HTTP.Response && return handle
    name = _follower_base(_follower_param(f))
    _engine_call() do
        fo = lock(() -> SSE.open_follower(handle, name), app.lock)
        n = SSE.sync!(fo)
        _clear_follower_error!(app, "$id/$name")
        json_response(200, merge(Dict{String,Any}("status" => "synchronized", "processed" => n),
                                 follower_status_dict(app, SSE.follower_status(fo))))
    end
end

"`POST /api/v1/datasets/{id}/followers/{f}/rebuild`: builds the follower again from its leader."
function handle_rebuild_follower(req::HTTP.Request, app, id::String, f::String)
    handle = _leader_or_404(app, id)
    handle isa HTTP.Response && return handle
    name = _follower_base(_follower_param(f))
    _engine_call() do
        fo = lock(() -> SSE.rebuild_follower!(handle, name), app.lock)
        _clear_follower_error!(app, "$id/$name")
        json_response(200, merge(Dict{String,Any}("status" => "rebuilt"), follower_status_dict(app, SSE.follower_status(fo))))
    end
end

"`GET /api/v1/datasets/{id}/followers/{f}/partitions`: the partitions of a partition follower."
function handle_follower_partitions(req::HTTP.Request, app, id::String, f::String)
    handle = _leader_or_404(app, id)
    handle isa HTTP.Response && return handle
    name = _follower_base(_follower_param(f))
    _engine_call() do
        fo = lock(() -> SSE.open_follower(handle, name), app.lock)
        json_response(200, Dict("id" => id, "follower" => name,
                                "partitions" => [Dict("name" => p.name, "items" => p.items, "graph" => p.graph)
                                                 for p in SSE.partitions(fo)]))
    end
end

# Runs a search of a follower and logs it on the leader, as `_run_search` does for a dataset.
function _follower_search(app, req::HTTP.Request, id::String, handle, k::Int, run::Function)
    stats = SSE.SearchStats()
    t0 = time()
    hits = _with_query_slot(() -> run(stats), app)
    Telemetry.log_request!(handle.project, nothing, "search", id, t0, nothing;
        identity=_request_identity(app, req), distance_name=_engine_distance_name(handle.engine),
        dimension=_engine_dimension(handle.engine), distance_evaluations=stats.distance_evaluations, metrics=app.metrics)
    results, insufficient = _format_results(hits, k)
    body = Dict{String,Any}("status" => "searched", "id" => id, "results" => results)
    insufficient && (body["insufficient_results"] = true)
    return body
end

"""
`POST /api/v1/datasets/{id}/followers/{f}/search`: with `{f} = name=value`, `search` in a
partition of a dense leader (`vector`, `k`, `filter`, `minrecall`); with a keyword follower, a
search by keys (`keys`, `k`, `t`, `filter`). `k` is limited by the `max_k` of the leader
(PLAN.md §5.16).
"""
function handle_follower_search(req::HTTP.Request, app, id::String, f::String)
    handle = _leader_or_404(app, id)
    handle isa HTTP.Response && return handle
    data = _body(req)
    bad_k = check_search_k(app, (id,), data)
    bad_k === nothing || return bad_k
    k = Int(get(data, "k", 10))
    filter_spec = get(data, "filter", nothing)
    filter = filter_spec === nothing ? nothing : _filter_predicate(filter_spec, SSE.meta_schema(handle))
    fname = _follower_param(f)
    response = _engine_call() do
        fo = lock(() -> SSE.open_follower(handle, fname), app.lock)
        if fo isa SSE.FollowerView
            haskey(data, "vector") || return json_response(400, Dict("error" => "a partition search requires a 'vector' field"))
            vector = convert(Vector{Float32}, data["vector"])
            minrecall = get(data, "minrecall", nothing)
            _follower_search(app, req, id, handle, k,
                stats -> SSE.search(fo, vector, k; filter, candidates=FILTER_CANDIDATES(k), minrecall, stats))
        else
            keys_ = get(data, "keys", nothing)
            keys_ isa AbstractVector && all(x -> x isa AbstractString, keys_) ||
                return json_response(400, Dict("error" => "a keyword search requires 'keys', a list of strings"))
            t = Int(get(data, "t", 1))
            _follower_search(app, req, id, handle, k,
                stats -> SSE.search(fo, String.(keys_), k; t, filter, candidates=FILTER_CANDIDATES(k), stats))
        end
    end
    response isa HTTP.Response ? response : json_response(200, response)
end

"`POST /api/v1/datasets/{id}/followers/{name=value}/ftsearch`: `ftsearch` in a partition of a text leader."
function handle_follower_ftsearch(req::HTTP.Request, app, id::String, f::String)
    handle = _leader_or_404(app, id)
    handle isa HTTP.Response && return handle
    data = _body(req)
    text = get(data, "text", nothing)
    text isa AbstractString || return json_response(400, Dict("error" => "ftsearch requires 'text', a string"))
    bad_k = check_search_k(app, (id,), data)
    bad_k === nothing || return bad_k
    k = Int(get(data, "k", 10))
    filter_spec = get(data, "filter", nothing)
    filter = filter_spec === nothing ? nothing : _filter_predicate(filter_spec, SSE.meta_schema(handle))
    response = _engine_call() do
        view = lock(() -> SSE.open_follower(handle, _follower_param(f)), app.lock)
        view isa SSE.FollowerView ||
            return json_response(400, Dict("error" => "ftsearch on a follower needs a partition: followers/{name}={value}/ftsearch"))
        _follower_search(app, req, id, handle, k,
            stats -> SSE.ftsearch(view, text, k; filter, candidates=FILTER_CANDIDATES(k), stats))
    end
    response isa HTTP.Response ? response : json_response(200, response)
end

"`POST /api/v1/datasets/{id}/followers/{name=value}/calibrate`: calibrates the graph of a partition."
function handle_follower_calibrate(req::HTTP.Request, app, id::String, f::String)
    handle = _leader_or_404(app, id)
    handle isa HTTP.Response && return handle
    data = _body(req)
    minrecall = Float64(get(data, "minrecall", 0.9))
    numqueries = Int(get(data, "numqueries", 64))
    ksearch = Int(get(data, "ksearch", 10))
    _engine_call() do
        view = lock(() -> SSE.open_follower(handle, _follower_param(f)), app.lock)
        view isa SSE.FollowerView ||
            return json_response(400, Dict("error" => "calibrate on a follower needs a partition: followers/{name}={value}/calibrate"))
        baseline, calibration = _calibration_response(only(SSE.calibrate!(view; levels=minrecall, numqueries, ksearch)))
        json_response(200, Dict("status" => "calibrated", "id" => id, "follower" => _follower_param(f),
                                "baseline" => baseline, "calibration" => calibration))
    end
end

function _postings_params(req::HTTP.Request)
    params = HTTP.queryparams(HTTP.URI(req.target))
    key = get(params, "key", nothing)
    key === nothing && return "postings requires the query parameter 'key'"
    limit = get(params, "limit", nothing)
    if limit !== nothing
        limit = tryparse(Int, limit)
        limit === nothing && return "'limit' must be an integer"
    end
    return (key=key, limit=limit)
end

"`GET /api/v1/datasets/{id}/followers/{f}/postings?key=&limit=`: the posting list of one key of a keyword follower."
function handle_follower_postings(req::HTTP.Request, app, id::String, f::String)
    handle = _leader_or_404(app, id)
    handle isa HTTP.Response && return handle
    p = _postings_params(req)
    p isa String && return json_response(400, Dict("error" => p))
    _engine_call() do
        fo = lock(() -> SSE.open_follower(handle, _follower_base(_follower_param(f))), app.lock)
        ids = SSE.posting_list(fo, p.key; limit=p.limit)
        json_response(200, Dict("id" => id, "follower" => fo.name, "key" => p.key, "ids" => ids))
    end
end

"""
`GET /api/v1/datasets/{id}/postings?key=&limit=`: the posting list of a token of a text dataset,
or of a dimension of a sparse dataset (the key is then an integer).
"""
function handle_postings(req::HTTP.Request, app, id::String)
    handle = _leader_or_404(app, id)
    handle isa HTTP.Response && return handle
    p = _postings_params(req)
    p isa String && return json_response(400, Dict("error" => p))
    key = SSE.payload_kind(handle.engine) === :sparse ? tryparse(Int, p.key) : p.key
    key === nothing && return json_response(400, Dict("error" => "the key of a sparse dataset is a dimension (an integer)"))
    _engine_call(() -> json_response(200, Dict("id" => id, "key" => p.key, "ids" => SSE.posting_list(handle, key; limit=p.limit))))
end

"`POST /api/v1/datasets/{id}/migrate-partitions`: moves the partitions inside a dataset to a follower."
function handle_migrate_partitions(req::HTTP.Request, app, id::String)
    handle = _leader_or_404(app, id)
    handle isa HTTP.Response && return handle
    data = _body(req)
    name = get(data, "name", nothing)
    name isa AbstractString || return json_response(400, Dict("error" => "'name' must be the name of the new follower"))
    disjoint = get(data, "disjoint", nothing)
    disjoint isa Bool || return json_response(400, Dict("error" =>
        "'disjoint' is required, true or false: a migration does not choose the kind of partition"))
    _engine_call() do
        f = lock(() -> SSE.migrate_partitions!(handle, name; disjoint), app.lock)
        json_response(200, merge(Dict{String,Any}("status" => "migrated"), follower_status_dict(app, SSE.follower_status(f))))
    end
end

"`GET /api/v1/datasets/{id}/group`: the leader and the status of all its followers."
function handle_get_group(req::HTTP.Request, app, id::String)
    handle = _leader_or_404(app, id)
    handle isa HTTP.Response && return handle
    _engine_call(() -> json_response(200, Dict("leader" => dataset_descriptor(app, id),
                                               "followers" => dataset_follower_statuses(app, handle))))
end

# --- Periodic followers ------------------------------------------------------------------

"""
    process_periodic_followers!(app; now=time()) -> Int

Processes each open follower with a `Periodic` policy whose period has passed since its last
processing, or whose lag has reached its number of items (PLAN.md §5.12). Returns the number of
followers processed. An error is kept for the status of the follower, and the others are still
processed.

An `append` already processes a follower whose lag reaches its number of items, at the event of
its leader (PLAN.md §5.15). The check of the number of items here processes a follower that was
opened with that lag, before the next event of its leader.
"""
function process_periodic_followers!(app; now::Float64=time())
    done = 0
    for (id, handle) in lock(() -> collect(app.handles), app.lock)
        for f in lock(() -> SSE.open_followers(handle), app.lock)
            policy = f.policy
            policy isa SSE.Periodic || continue
            canonical = "$id/$(f.name)"
            last = lock(() -> get(app.followers.last_sync, canonical, 0.0), app.followers.lock)
            due = (policy.seconds !== nothing && now - last >= policy.seconds) ||
                  (policy.items !== nothing && SSE.follower_status(f).lag >= policy.items)
            due || continue
            try
                lock(() -> f.closed || SSE.sync!(f), app.lock)
                lock(() -> (app.followers.last_sync[canonical] = now), app.followers.lock)
                _clear_follower_error!(app, canonical)
                done += 1
            catch e
                # `sync!` wrote the error to the log, and the status of the follower holds it.
                e isa SSE.EngineError || e isa ErrorException || rethrow()
                _record_follower_error!(app, canonical, e)
            end
        end
    end
    return done
end

"""
    start_follower_task!(app) -> Task

Starts the task of `serve` that processes the periodic followers every
`app.followers.check_seconds` seconds (PLAN.md §5.12).
"""
start_follower_task!(app) = errormonitor(@async while true
    sleep(app.followers.check_seconds)
    try
        # In the newest world, so that the methods of a plugin loaded after this task started
        # are visible (PLAN.md §5.13).
        Base.invokelatest(process_periodic_followers!, app)
    catch e
        @error "the periodic follower task failed; it continues" exception=(e, catch_backtrace())
    end
end)
