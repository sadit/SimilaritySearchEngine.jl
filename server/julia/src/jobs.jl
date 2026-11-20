"""
    KnnRow

One row of [`allknn`](@ref): the item `_id` and its `k` nearest neighbors, by ascending
distance. The fields are those of the engine's `KnnRow`. `_id` and `neighbors` are internal ids.
"""
struct KnnRow
    _id::Int32
    neighbors::Vector{Int32}
    dists::Vector{Float32}
end

"""
    CenterSelectionResult

The result of [`fft`](@ref). The fields are those of the engine's `CenterSelectionResult`:
`centers` are internal ids, `assign[i]` is the position in `centers` of the center of item `i`,
and `assigndist[i]` is the distance from item `i` to that center. `covering` is the largest
`assigndist`, and `separation` is the smallest distance between two centers.
"""
struct CenterSelectionResult
    centers::Vector{Int32}
    assign::Vector{Int32}
    assigndist::Vector{Float32}
    covering::Float32
    separation::Float32
    costdists::Int
    costblocks::Int
end

"The engine's name for the result of [`fft`](@ref)."
const FFTResult = CenterSelectionResult

"""
    NearDupResult

The result of [`neardup`](@ref). The fields are those of the engine's `NearDupResult`: the
selected `centers`, the position in `centers` of the center of each item (`assign`), its
distance to it (`assigndist`), the largest of those distances (`covering`), and `epsilon`.
"""
struct NearDupResult
    centers::Vector{Int32}
    assign::Vector{Int32}
    assigndist::Vector{Float32}
    covering::Float32
    epsilon::Float32
    costdists::Int
    costblocks::Int
end

"""
    JobFailed <: RemoteError

A job that ended in the `failed` state. `job_id` and `kind` name the job, and `msg` is the
error the server recorded.

`exit_code` is the exit status of the job's process, or `nothing` when the server did not record
it. The command line of the server exits with a code per category of engine error, and
`category` is that category: `2` is [`InvalidRequest`](@ref), `4` is [`NotFound`](@ref), `5` is
[`ConflictingState`](@ref) and `70` is [`StorageFailure`](@ref). Any other code, for example `1`
for a dataset directory that does not exist, has `category === nothing`. The message of the
engine error is in the output of the job's process on the server, which the API does not return.
"""
struct JobFailed <: RemoteError
    job_id::String
    kind::String
    exit_code::Union{Nothing,Int}
    category::Union{Nothing,Type}
    msg::String
end

function Base.showerror(io::IO, e::JobFailed)
    print(io, "JobFailed: ", e.kind, " job ", e.job_id, ": ", e.msg)
    e.category === nothing || print(io, " (", nameof(e.category), ")")
end

"""
    JobTimeout <: RemoteError

[`wait_job`](@ref) waited `seconds` and the job `job_id` had not finished. The job keeps
running on the server.
"""
struct JobTimeout <: RemoteError
    job_id::String
    seconds::Float64
end

Base.showerror(io::IO, e::JobTimeout) =
    print(io, "JobTimeout: job ", e.job_id, " did not finish in ", e.seconds, " s; it is still on the server")

"The engine error category that an exit code of the server's command line names."
exit_category(code::Integer) = code == 2 ? InvalidRequest : code == 4 ? NotFound :
                               code == 5 ? ConflictingState : code == 70 ? StorageFailure : nothing
exit_category(::Nothing) = nothing

"""
    submit_job(conn::Connection, kind; params...) -> String

Submits a job (`POST /api/v1/jobs/{kind}`) and returns its id. `kind` is `"allknn"`, `"fft"`,
`"neardup"`, `"hsp"`, `"dump"` or `"load"`, and `params` are the fields of the request, for
example `dataset="papers", k=16`. The server manual lists the fields of each kind. Another kind,
such as `"searchbatch"`, is submitted with `command=[...]`, the arguments of the server's command
line. The job runs in a separate process on the server; [`wait_job`](@ref) waits for it.
"""
function submit_job(conn::Connection, kind::AbstractString; params...)
    body = Dict{String,Any}(string(k) => v for (k, v) in params)
    return string(request(conn, "POST", "/api/v1/jobs/" * HTTP.escapeuri(kind); body)[:job_id])
end

"""
    job_info(conn::Connection, job_id) -> @NamedTuple{status::Symbol, record::Dict{String,Any}}

The state of a job (`:queued`, `:running`, `:blocked`, `:completed` or `:failed`) and its
record: `kind`, `command`, the times, and `error` and `exit_code` for a failed job.
"""
function job_info(conn::Connection, job_id::AbstractString)
    resp = request(conn, "GET", "/api/v1/jobs/" * HTTP.escapeuri(job_id))
    return (status=Symbol(resp[:status]), record=_plain(resp[:job]))
end

"""
    wait_job(conn::Connection, job_id; poll_interval=1.0, timeout=Inf) -> Dict{String,Any}

Asks for the state of the job every `poll_interval` seconds until it ends, and returns its
record. A failed job raises [`JobFailed`](@ref). After `timeout` seconds it raises
[`JobTimeout`](@ref), and the job keeps running on the server.

To run several jobs at the same time, call this function in several tasks, for example with
`@async` or `Threads.@spawn`.
"""
function wait_job(conn::Connection, job_id::AbstractString; poll_interval::Real=1.0, timeout::Real=Inf)
    t0 = time()
    while true
        info = job_info(conn, job_id)
        info.status === :completed && return info.record
        if info.status === :failed
            code = get(info.record, "exit_code", nothing)
            throw(JobFailed(String(job_id), string(get(info.record, "kind", "")), code, exit_category(code),
                            string(get(info.record, "error", "the job failed"))))
        end
        time() - t0 > timeout && throw(JobTimeout(String(job_id), Float64(timeout)))
        sleep(poll_interval)
    end
end

"""
    job_result(conn::Connection, job_id) -> String

The result of a completed job, as the server sends it (`GET /api/v1/jobs/{id}/result`): the
file that the job wrote. Its format depends on the kind of the job. [`allknn`](@ref),
[`fft`](@ref) and [`neardup`](@ref) read it into the engine's types. A job that has not
completed raises [`StateConflict`](@ref).
"""
function job_result(conn::Connection, job_id::AbstractString)
    resp = HTTP.request("GET", conn.url * "/api/v1/jobs/" * HTTP.escapeuri(job_id) * "/result", conn.headers;
                        status_exception=false, read_idle_timeout=conn.read_idle_timeout)
    200 <= resp.status < 300 || throw(response_error(resp.status, resp.body))
    return String(resp.body)
end

"""
    list_jobs(conn::Connection; status=nothing, kind=nothing)
        -> Vector{@NamedTuple{id::String, kind::Union{Nothing,String}, status::Symbol}}

The jobs of the server, optionally only those in one `status` (`:queued`, `:running`,
`:blocked`, `:completed`, `:failed`) or of one `kind`.
"""
function list_jobs(conn::Connection; status=nothing, kind::Union{Nothing,AbstractString}=nothing)
    query = Dict{String,String}()
    status === nothing || (query["status"] = string(status))
    kind === nothing || (query["kind"] = kind)
    resp = request(conn, "GET", "/api/v1/jobs"; query)
    return [(id=string(j[:id]), kind=(k = get(j, :kind, nothing); k === nothing ? nothing : string(k)),
             status=Symbol(j[:status])) for j in resp[:jobs]]
end

"""
    cancel_job!(conn::Connection, job_id) -> nothing

Cancels a job that has not started: a `queued` or `blocked` job ends as `failed`. A running or
finished job raises [`StateConflict`](@ref); [`kill_job!`](@ref) stops a running one. Needs the
`admin` permission, as the next three functions do.
"""
cancel_job!(conn::Connection, job_id::AbstractString) =
    (request(conn, "DELETE", "/api/v1/jobs/" * HTTP.escapeuri(job_id)); nothing)

"""
    kill_job!(conn::Connection, job_id) -> nothing

Stops the process of a running job, which ends as `failed`. A job that is not running raises
[`StateConflict`](@ref).
"""
kill_job!(conn::Connection, job_id::AbstractString) =
    (request(conn, "POST", "/api/v1/jobs/" * HTTP.escapeuri(job_id) * "/kill"); nothing)

"""
    block_job!(conn::Connection, job_id) -> nothing

Moves a `queued` job to `blocked`, where the server does not start it until
[`resume_job!`](@ref). A job in another state raises [`StateConflict`](@ref).
"""
block_job!(conn::Connection, job_id::AbstractString) =
    (request(conn, "POST", "/api/v1/jobs/" * HTTP.escapeuri(job_id) * "/block"); nothing)

"""
    resume_job!(conn::Connection, job_id) -> nothing

Moves a `blocked` job back to `queued`. A job in another state raises [`StateConflict`](@ref).
"""
resume_job!(conn::Connection, job_id::AbstractString) =
    (request(conn, "POST", "/api/v1/jobs/" * HTTP.escapeuri(job_id) * "/resume"); nothing)

"Submits a job, waits for it, and returns the text of its result."
function _run_job(conn::Connection, kind::String, poll_interval::Real; params...)
    id = submit_job(conn, kind; params...)
    wait_job(conn, id; poll_interval)
    return job_result(conn, id)
end

"""
    allknn(h::RemoteProject; k=10, poll_interval=1.0) -> Vector{KnnRow}

The `k` nearest neighbors of every item of the dataset, as in the engine. It runs as an `allknn`
job on the server: this call submits it, waits for it with [`wait_job`](@ref), and reads its
result. A failed job raises [`JobFailed`](@ref).
"""
function allknn(h::RemoteProject; k::Integer=10, poll_interval::Real=1.0)
    text = _run_job(h.conn, "allknn", poll_interval; dataset=h.id, k=Int(k))
    rows = KnnRow[]
    for line in eachline(IOBuffer(text))
        isempty(strip(line)) && continue
        r = JSON3.read(line)
        push!(rows, KnnRow(Int32(r[:id]), Int32.(r[:neighbors]), Float32.(r[:dists])))
    end
    return rows
end

"""
    fft(h::RemoteProject, k; poll_interval=1.0) -> CenterSelectionResult

`k` centers selected by farthest-first traversal, as in the engine. It runs as an `fft` job,
as [`allknn`](@ref) does.
"""
function fft(h::RemoteProject, k::Integer; poll_interval::Real=1.0)
    r = JSON3.read(_run_job(h.conn, "fft", poll_interval; dataset=h.id, k=Int(k)))
    return CenterSelectionResult(Int32.(r[:centers]), Int32.(r[:assign]), Float32.(r[:dists]),
                                 Float32(r[:covering]), Float32(r[:separation]),
                                 Int(r[:costdists]), Int(r[:costblocks]))
end

"""
    neardup(h::RemoteProject, epsilon; poll_interval=1.0) -> NearDupResult

The near duplicates of the dataset at distance `epsilon`, as in the engine: `centers` are the
items kept, and `assign` gives the center of each item. It runs as a `neardup` job, as
[`allknn`](@ref) does.
"""
function neardup(h::RemoteProject, epsilon::Real; poll_interval::Real=1.0)
    r = JSON3.read(_run_job(h.conn, "neardup", poll_interval; dataset=h.id, epsilon=Float64(epsilon)))
    return NearDupResult(Int32.(r[:centers]), Int32.(r[:assign]), Float32.(r[:dists]),
                         Float32(r[:covering]), Float32(r[:epsilon]), Int(r[:costdists]), Int(r[:costblocks]))
end

"""
    dump_project(h::RemoteProject; poll_interval=1.0) -> @NamedTuple{bundle_dir::String, manifest::Dict{String,Any}}

Writes a copy of the dataset to a bundle directory on the server, with a `dump` job, and returns
the path of that directory and its manifest. The path is on the server's file system: the bundle
does not travel over HTTP. [`load_project`](@ref) creates a dataset from it.
"""
function dump_project(h::RemoteProject; poll_interval::Real=1.0)
    r = JSON3.read(_run_job(h.conn, "dump", poll_interval; dataset=h.id))
    manifest = get(r, :manifest, nothing)
    return (bundle_dir=string(r[:bundle_dir]),
            manifest=manifest === nothing ? Dict{String,Any}() : _plain(manifest))
end

"""
    load_project(conn::Connection, bundle_dir, dataset; poll_interval=1.0) -> RemoteProject

Creates the dataset `dataset` from the bundle at `bundle_dir`, a path on the server's file
system that [`dump_project`](@ref) returned, with a `load` job. Then it asks the server to load
the new dataset (`POST /api/v1/admin/datasets/{id}/reload`), which needs the `admin` permission,
and returns its handle. A `dataset` that already exists makes the job fail.
"""
function load_project(conn::Connection, bundle_dir::AbstractString, dataset::AbstractString; poll_interval::Real=1.0)
    id = submit_job(conn, "load"; dataset=String(dataset), bundle=String(bundle_dir))
    wait_job(conn, id; poll_interval)
    request(conn, "POST", "/api/v1/admin/datasets/" * HTTP.escapeuri(dataset) * "/reload")
    return RemoteProject(conn, String(dataset))
end
