module Errors

export EngineError, InvalidRequest, NotFound, ConflictingState, StorageFailure
export PayloadMismatch, WrongDimension, UnknownBackend, InvalidOption, UnsupportedOperation
export ProfileNotInstalled, PendingBacklog, NothingStaged, NoIndex, NotTrained, EmptyProject
export CorruptedStorage

"""
    EngineError <: Exception

Everything this package raises on purpose.

Four abstract categories sit under it -- [`InvalidRequest`](@ref), [`NotFound`](@ref),
[`ConflictingState`](@ref), [`StorageFailure`](@ref) -- and the concrete types under those
carry the detail. **The category is the contract.** A consumer maps categories to its own
vocabulary (the HTTP server to status codes, a CLI to exit codes) and reads the message only
to show a human; a consumer that wants to distinguish two situations inside one category
matches the concrete type instead.

This exists because the alternative is classifying by message text, which every consumer is
forced into when a library raises bare `error("...")`, and which breaks silently the first
time somebody improves the wording. The engine itself knows nothing about status codes or
exit codes and should not learn: mapping lives at each boundary, not here.
"""
abstract type EngineError <: Exception end

"""
    InvalidRequest <: EngineError

The call cannot be honoured as written: wrong kind of item for the project, a keyword that
does not apply, a dimension that does not match. Retrying it unchanged will fail again.
Boundaries that speak HTTP report this as 400.
"""
abstract type InvalidRequest <: EngineError end

"""
    NotFound <: EngineError

Something the call named does not exist. HTTP: 404.
"""
abstract type NotFound <: EngineError end

"""
    ConflictingState <: EngineError

The project is real and the request is well formed, but the project is not in a state where
it can be served -- items staged but not indexed, a text project not trained yet, an empty
collection. The same call succeeds after the state changes, which is what separates this from
[`InvalidRequest`](@ref). HTTP: 409.
"""
abstract type ConflictingState <: EngineError end

"""
    StorageFailure <: EngineError

What was read back from storage cannot be interpreted: an unknown tag, a project recorded in
a shape this version does not know. Not the caller's fault and not fixable by retrying.
HTTP: 500.
"""
abstract type StorageFailure <: EngineError end

# --- InvalidRequest ------------------------------------------------------------------

"An item or query of the wrong kind for this project (a `DenseItem` for a text project, ...)."
struct PayloadMismatch <: InvalidRequest
    msg::String
end

"A vector whose length is not the project's own."
struct WrongDimension <: InvalidRequest
    expected::Int
    got::Int
    msg::String
end

"An engine/backend pairing this package does not have, or a backend name it does not know."
struct UnknownBackend <: InvalidRequest
    msg::String
end

"A keyword that does not apply here, is missing, or holds a value this call cannot use."
struct InvalidOption <: InvalidRequest
    option::Symbol
    msg::String
end

"An operation this kind of project or index does not implement."
struct UnsupportedOperation <: InvalidRequest
    operation::Symbol
    msg::String
end

# --- NotFound ------------------------------------------------------------------------

"A linguistic profile the project asks for, absent from the local profile library."
struct ProfileNotInstalled <: NotFound
    nickname::String
    msg::String
end

# --- ConflictingState ----------------------------------------------------------------

"""
Items are staged but not indexed, and this operation reads structures that only the next
`index!` call grows. Carries both counts, so a caller can say how far behind it is.
"""
struct PendingBacklog <: ConflictingState
    operation::Symbol
    staged::Int
    connected::Int
    msg::String
end

"Nothing has ever been staged, so there is nothing to index."
struct NothingStaged <: ConflictingState
    msg::String
end

"A text project with a profile but no index on disk -- `index!` builds one from staged text."
struct NoIndex <: ConflictingState
    msg::String
end

"A text project whose profile has not been fitted yet."
struct NotTrained <: ConflictingState
    msg::String
end

"An operation over the whole collection, on a collection with nothing in it."
struct EmptyProject <: ConflictingState
    operation::Symbol
    msg::String
end

# --- StorageFailure ------------------------------------------------------------------

"Stored bytes or recorded state that this version cannot interpret."
struct CorruptedStorage <: StorageFailure
    msg::String
end

Base.showerror(io::IO, e::EngineError) = print(io, e.msg)

# --- throwing helpers ----------------------------------------------------------------
#
# One per concrete type, shaped like `error` itself (variadic, concatenated) so that a call
# site reads the same as the `error("...")` it replaced -- same parentheses, same string
# interpolation, same multi-line `"""..."""` messages. They are not exported from the package:
# a caller *catches* these types, it does not raise them, and the surface stays small.
export payload_mismatch, wrong_dimension, unknown_backend, invalid_option, unsupported_operation
export profile_not_installed, pending_backlog, nothing_staged, no_index, not_trained
export empty_project, corrupted_storage

payload_mismatch(msg...) = throw(PayloadMismatch(string(msg...)))
wrong_dimension(expected::Integer, got::Integer, msg...) =
    throw(WrongDimension(Int(expected), Int(got), string(msg...)))
unknown_backend(msg...) = throw(UnknownBackend(string(msg...)))
invalid_option(option::Symbol, msg...) = throw(InvalidOption(option, string(msg...)))
unsupported_operation(operation::Symbol, msg...) =
    throw(UnsupportedOperation(operation, string(msg...)))
profile_not_installed(nickname, msg...) =
    throw(ProfileNotInstalled(String(nickname), string(msg...)))
pending_backlog(operation::Symbol, staged::Integer, connected::Integer, msg...) =
    throw(PendingBacklog(operation, Int(staged), Int(connected), string(msg...)))
nothing_staged(msg...) = throw(NothingStaged(string(msg...)))
no_index(msg...) = throw(NoIndex(string(msg...)))
not_trained(msg...) = throw(NotTrained(string(msg...)))
empty_project(operation::Symbol, msg...) = throw(EmptyProject(operation, string(msg...)))
corrupted_storage(msg...) = throw(CorruptedStorage(string(msg...)))

end # module
