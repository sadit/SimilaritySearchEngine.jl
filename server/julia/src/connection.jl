"""
    Connection(url="http://127.0.0.1:8080"; token=nothing, read_idle_timeout=60)

The address of a `similarity-search-serve` process, and the token sent with every request.

`url` is the root of the server, without `/api/v1`. `token`, if given, is sent as
`Authorization: Bearer <token>`. A server that has issued tokens refuses a request without a
valid one, and the client raises [`Unauthenticated`](@ref) or [`Forbidden`](@ref).
`read_idle_timeout` is the number of seconds the client waits without receiving data, and `0`
waits without a limit. A calibration or a large append can take minutes.

A `Connection` is a value: it holds no socket. HTTP.jl keeps a pool of connections for every
request of the process.
"""
struct Connection
    url::String
    headers::Vector{Pair{String,String}}
    read_idle_timeout::Int
end

function Connection(url::AbstractString="http://127.0.0.1:8080"; token::Union{Nothing,AbstractString}=nothing,
                    read_idle_timeout::Integer=60)
    headers = Pair{String,String}["Content-Type" => "application/json"]
    token === nothing || push!(headers, "Authorization" => "Bearer $token")
    Connection(rstrip(String(url), '/'), headers, Int(read_idle_timeout))
end

Base.show(io::IO, c::Connection) = print(io, "Connection(", repr(c.url), ")")

"""
    request(conn, method, path; body=nothing, query=nothing) -> Union{Nothing,JSON3.Object}

Sends one request to `conn.url * path`, with `body` encoded as JSON. Returns the parsed body of a
2xx response, or `nothing` when the body is empty. Any other status raises the error that
[`response_error`](@ref) selects.
"""
function request(conn::Connection, method::String, path::String; body=nothing, query=nothing)
    payload = body === nothing ? UInt8[] : JSON3.write(body)
    resp = HTTP.request(method, conn.url * path, conn.headers, payload; query,
                        status_exception=false, read_idle_timeout=conn.read_idle_timeout)
    200 <= resp.status < 300 || throw(response_error(resp.status, resp.body))
    isempty(resp.body) && return nothing
    return JSON3.read(resp.body)
end

"The path of a route of dataset `id`, with the id escaped."
dataset_path(id::AbstractString, rest::AbstractString="") = "/api/v1/datasets/" * HTTP.escapeuri(id) * rest
