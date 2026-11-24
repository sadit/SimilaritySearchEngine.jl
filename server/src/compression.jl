# Compression of the responses (CHANGELOG.md §5.16). Included in `module Server`.
#
# A middleware compresses a response body with gzip when the request accepts it and the body
# has at least `[server] gzip_min_bytes` bytes. The handlers do not change. The result of a job
# is a file, which `handle_get_job_result` sends in parts, compressed while it is read.

"The default of `[server] gzip`: whether the server compresses large responses with gzip."
const DEFAULT_GZIP = true

"""
The default of `[server] gzip_min_bytes`: the size, in bytes, from which a response body is
compressed. A smaller body is sent as it is, because gzip adds about 20 bytes to it and the
time of the compression.
"""
const DEFAULT_GZIP_MIN_BYTES = 1024

"""
    accepts_gzip(req::HTTP.Request) -> Bool

Whether the request accepts a body compressed with gzip: its `Accept-Encoding` header names
`gzip`, `x-gzip` or `*` without `q=0`.
"""
function accepts_gzip(req::HTTP.Request)
    for item in split(HTTP.header(req, "Accept-Encoding", ""), ',')
        parts = split(item, ';')
        lowercase(strip(first(parts))) in ("gzip", "x-gzip", "*") || continue
        q = 1.0
        for p in parts[2:end]
            kv = split(strip(p), '='; limit=2)
            length(kv) == 2 && lowercase(strip(kv[1])) == "q" &&
                (q = something(tryparse(Float64, strip(kv[2])), 1.0))
        end
        q > 0 && return true
    end
    return false
end

# The bytes of a body held in memory, or `nothing` for a body that is sent in parts.
_body_bytes(body::AbstractVector{UInt8}) = body
_body_bytes(body::HTTP.BytesBody) = codeunits(String(body))
_body_bytes(body::AbstractString) = codeunits(body)
_body_bytes(body) = nothing

# The headers of `res` without the ones that the compression changes.
_headers_without_length(res::HTTP.Response) =
    Pair{String,String}[String(k) => String(v) for (k, v) in res.headers if lowercase(k) != "content-length"]

"""
    gzip_middleware(min_bytes::Integer) -> Function

The middleware of `serve` that compresses a response with gzip (CHANGELOG.md §5.16). It compresses
when the request accepts gzip, the response has no `Content-Encoding`, and its body is held
in memory and has at least `min_bytes` bytes. It sets `Content-Encoding: gzip` and
`Vary: Accept-Encoding`. It runs after the handlers and their authorization, so it also
compresses an error response.
"""
function gzip_middleware(min_bytes::Integer)
    return handler -> function (req::HTTP.Request)
        res = handler(req)
        res isa HTTP.Response || return res
        accepts_gzip(req) || return res
        HTTP.hasheader(res, "Content-Encoding") && return res
        bytes = _body_bytes(res.body)
        (bytes === nothing || length(bytes) < min_bytes) && return res
        headers = _headers_without_length(res)
        push!(headers, "Content-Encoding" => "gzip", "Vary" => "Accept-Encoding")
        return HTTP.Response(res.status, headers, transcode(CodecZlib.GzipCompressor, Vector{UInt8}(bytes)))
    end
end

"""
    file_response(req, path; headers) -> HTTP.Response

A response that sends the file `path` in parts, without reading it into memory. When the
request accepts gzip, the parts are compressed while the file is read, and the response has
`Content-Encoding: gzip`. The length of the body is not known in advance, so it is sent with
chunked transfer encoding.
"""
function file_response(req::HTTP.Request, path::AbstractString; headers=JSON_HEADERS)
    io = open(path, "r")
    gz = accepts_gzip(req)
    src = gz ? CodecZlib.GzipCompressorStream(io) : io
    body = HTTP.CallbackBody(dst -> readbytes!(src, dst, length(dst)), () -> close(src))
    out = Pair{String,String}[String(k) => String(v) for (k, v) in headers]
    gz && push!(out, "Content-Encoding" => "gzip", "Vary" => "Accept-Encoding")
    return HTTP.Response(200, out, body)
end
