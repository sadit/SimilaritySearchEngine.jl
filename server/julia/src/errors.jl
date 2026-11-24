"""
    RemoteError <: Exception

Every error this package raises from a server response. Errors of the connection itself (a
refused connection, a timeout) are not wrapped: they reach the caller as the exceptions of
HTTP.jl.

[`EngineError`](@ref) is the part of this hierarchy that has the names of the engine's own
errors. [`AccessDenied`](@ref), [`ServerFailure`](@ref) and [`UnexpectedResponse`](@ref) are
the server's own answers, which the engine does not produce.
"""
abstract type RemoteError <: Exception end

"""
    EngineError <: RemoteError

An error of the engine, reported by the server. It has the same four categories as the engine:
[`InvalidRequest`](@ref) (HTTP 400), [`NotFound`](@ref) (404), [`ConflictingState`](@ref) (409)
and [`StorageFailure`](@ref) (500). The category is the contract, as in the engine. A script
that catches `NotFound` works the same against an embedded project and against a remote one.

The server sends the name of the engine's concrete type in the `kind` field of its response,
and this package raises the type of the same name. A response without a `kind`, such as the
server's `dataset_not_found`, raises the generic type of its category:
[`RequestRejected`](@ref), [`ResourceNotFound`](@ref) or [`StateConflict`](@ref).

Every concrete type has two fields: `status`, the HTTP status, and `msg`, the server's message.
The engine's types carry more fields (`WrongDimension` carries `expected` and `got`). The server
does not send them, so they are not here.
"""
abstract type EngineError <: RemoteError end

"The request cannot be served as written. Sending it again unchanged fails again. HTTP 400."
abstract type InvalidRequest <: EngineError end

"Something the request names does not exist. HTTP 404."
abstract type NotFound <: EngineError end

"The request is valid, but the dataset is not in a state that can serve it. HTTP 409."
abstract type ConflictingState <: EngineError end

"The server read stored data that it cannot interpret. HTTP 500 with an engine `kind`."
abstract type StorageFailure <: EngineError end

"The server refused the token of the request: HTTP 401 or 403."
abstract type AccessDenied <: RemoteError end

for (name, parent, doc) in (
        (:PayloadMismatch, :InvalidRequest, "An item or a query of the wrong kind for the dataset."),
        (:WrongDimension, :InvalidRequest, "A vector whose length is not the dataset's dimension."),
        (:UnknownBackend, :InvalidRequest, "An index type that the server does not know."),
        (:InvalidOption, :InvalidRequest, "An option that does not apply, is missing, or has a value the server cannot use."),
        (:UnsupportedOperation, :InvalidRequest, "An operation that this kind of dataset does not implement."),
        (:RequestRejected, :InvalidRequest, "A 400 response without an engine `kind`: the server rejected the request before it reached the engine."),
        (:ProfileNotInstalled, :NotFound, "A linguistic profile that the server does not have."),
        (:UnknownPartition, :NotFound, "A partition name that the dataset does not have. The message lists the closest names."),
        (:UnknownFollower, :NotFound, "A follower name that the leader dataset does not have. The message lists the closest names."),
        (:PluginNotLoaded, :NotFound, "A follower that needs a plugin the server has not loaded."),
        (:ResourceNotFound, :NotFound, "A 404 response without an engine `kind`, for example a dataset that does not exist or is not loaded."),
        (:PendingBacklog, :ConflictingState, "Items are appended but not indexed yet."),
        (:NothingStaged, :ConflictingState, "Nothing was ever appended, so there is nothing to index."),
        (:NoIndex, :ConflictingState, "A text dataset with a profile and no index."),
        (:NotTrained, :ConflictingState, "A text dataset without a fitted vocabulary, or a partition that was not calibrated."),
        (:EmptyProject, :ConflictingState, "An operation over the whole dataset, on a dataset with no items."),
        (:ProjectClosed, :ConflictingState, "The server closed the dataset while the request was served, for example to reload or delete it."),
        (:StaleFollower, :ConflictingState, "A follower built before its leader was rebuilt. Rebuilding the follower makes it usable again."),
        (:HasFollowers, :ConflictingState, "A request to delete a dataset that has followers. The followers are deleted first."),
        (:StateConflict, :ConflictingState, "A 409 response without an engine `kind`."),
        (:CorruptedStorage, :StorageFailure, "Stored data that the server cannot interpret."),
        (:Unauthenticated, :AccessDenied, "The request has no token, or its token is not valid. HTTP 401."),
        (:Forbidden, :AccessDenied, "The token of the request does not have the permission the route requires. HTTP 403."),
    )
    @eval begin
        @doc $doc struct $name <: $parent
            status::Int
            msg::String
        end
    end
end

"""
    ServerFailure <: RemoteError

A 500 response without an engine `kind`: an error inside the server that is not one of the
engine's.
"""
struct ServerFailure <: RemoteError
    status::Int
    msg::String
end

"""
    UnexpectedResponse <: RemoteError

A response with a status that this package does not map, for example a 502 from a proxy in front
of the server. `msg` holds the body of the response.
"""
struct UnexpectedResponse <: RemoteError
    status::Int
    msg::String
end

Base.showerror(io::IO, e::RemoteError) = print(io, nameof(typeof(e)), " (HTTP ", e.status, "): ", e.msg)

"The types a `kind` field can name. A test of the server suite checks it against the engine."
const ERROR_KINDS = Dict{String,Type}(
    "PayloadMismatch" => PayloadMismatch, "WrongDimension" => WrongDimension,
    "UnknownBackend" => UnknownBackend, "InvalidOption" => InvalidOption,
    "UnsupportedOperation" => UnsupportedOperation, "ProfileNotInstalled" => ProfileNotInstalled,
    "UnknownPartition" => UnknownPartition, "PendingBacklog" => PendingBacklog,
    "NothingStaged" => NothingStaged, "NoIndex" => NoIndex, "NotTrained" => NotTrained,
    "EmptyProject" => EmptyProject, "ProjectClosed" => ProjectClosed,
    "CorruptedStorage" => CorruptedStorage,
    "UnknownFollower" => UnknownFollower, "PluginNotLoaded" => PluginNotLoaded,
    "StaleFollower" => StaleFollower, "HasFollowers" => HasFollowers,
    "Unauthenticated" => Unauthenticated, "Forbidden" => Forbidden,
)

"""
    response_error(status, body) -> RemoteError

The error that a response with `status` and `body` raises. The `kind` field of the body selects
the type. Without a known `kind`, the status selects the generic type of its category.
"""
function response_error(status::Integer, body::AbstractVector{UInt8})
    msg, kind = String(copy(body)), nothing
    parsed = try
        JSON.parse(body)
    catch
        nothing
    end
    if parsed isa JSON.Object
        haskey(parsed, :error) && (msg = string(parsed[:error]))
        kind = get(parsed, :kind, nothing)
    end
    T = kind === nothing ? nothing : get(ERROR_KINDS, string(kind), nothing)
    T === nothing || return T(status, msg)
    status == 400 && return RequestRejected(status, msg)
    status == 401 && return Unauthenticated(status, msg)
    status == 403 && return Forbidden(status, msg)
    status == 404 && return ResourceNotFound(status, msg)
    status == 409 && return StateConflict(status, msg)
    status == 500 && return ServerFailure(status, msg)
    return UnexpectedResponse(status, msg)
end
