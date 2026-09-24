module Tokens

using RocksDB
using JSON3
using Random
using Dates

export TokenManager, TokenRecord, open_token_manager, close_token_manager, create_token!, get_token, revoke_token!, list_tokens, prune_expired!,
       validate_token, has_permission, parse_permission, normalize_permissions, is_expired, any_token_exists

"""
    TokenRecord

Represents a token and its Access Control List (ACL).

# Fields
- `token_str::String`: The secret token string.
- `user::String`: The user associated with the token.
- `permissions::Vector{String}`: A list of permitted actions.
- `created_at::String`: Token creation timestamp.
- `expires_at::Union{String, Nothing}`: Expiration timestamp or nothing if it doesn't expire.
"""
struct TokenRecord
    token_str::String
    user::String
    permissions::Vector{String}
    created_at::String
    expires_at::Union{String, Nothing}
end

"""
    TokenManager

Manages the global token database.

# Fields
- `db::RocksDB.DB`: The main RocksDB database handle for tokens.
"""
mutable struct TokenManager
    db::RocksDB.DB
end

"""
    open_token_manager(workdir::String) -> TokenManager

Opens or creates the global token RocksDB database in the working directory.

# Arguments
- `workdir::String`: The working directory containing the database.

# Returns
- `TokenManager`: The initialized token manager.
"""
function open_token_manager(workdir::String)
    db_path = joinpath(workdir, "system_tokens")
    db = RocksDB.opendb(db_path, create_if_missing=true)
    return TokenManager(db)
end

function close_token_manager(manager::TokenManager)
    close(manager.db)
end

function generate_token_string()
    return bytes2hex(rand(UInt8, 16)) # 32-character hex string token
end

"""
    create_token!(manager::TokenManager, user::String, permissions::Vector{String}; expires_at::Union{String, Nothing}=nothing) -> String

Creates a new token and stores it in the database.

# Arguments
- `manager::TokenManager`: The token manager instance.
- `user::String`: User name or identifier.
- `permissions::Vector{String}`: List of permissions.
- `expires_at::Union{String, Nothing}`: Optional expiration timestamp.

# Returns
- `String`: The generated token string.
"""
function create_token!(manager::TokenManager, user::String, permissions::Vector{String}; expires_at::Union{String, Nothing}=nothing)
    token_str = generate_token_string()

    # Rejected here rather than at validation time: a timestamp that does not parse would
    # otherwise produce a token that is refused on every request, with no indication of why at
    # the moment it was created.
    expires_at === nothing || try
        DateTime(expires_at)
    catch
        throw(ArgumentError("expires_at must be an ISO 8601 timestamp, e.g. 2027-01-01T00:00:00; got $(repr(expires_at))"))
    end
    permissions = normalize_permissions(permissions)

    record = Dict(
        "token_str" => token_str,
        "user" => user,
        "permissions" => permissions,
        "created_at" => string(now(UTC)),
        "expires_at" => expires_at
    )
    
    key_bytes = Vector{UInt8}(token_str)
    val_bytes = Vector{UInt8}(JSON3.write(record))

    RocksDB.put!(manager.db, key_bytes, val_bytes)
    return token_str
end

"""
    get_token(manager::TokenManager, token_str::String) -> Union{TokenRecord, Nothing}

Retrieves a token by its string representation.

# Arguments
- `manager::TokenManager`: The token manager instance.
- `token_str::String`: The token string to search for.

# Returns
- `TokenRecord`: The retrieved token record, or `nothing` if not found.
"""
function get_token(manager::TokenManager, token_str::String)
    key_bytes = Vector{UInt8}(token_str)
    val_bytes = get(manager.db, key_bytes)

    if val_bytes === nothing
        return nothing
    end
    
    data = JSON3.read(val_bytes, Dict{String, Any})
    return TokenRecord(
        data["token_str"],
        data["user"],
        Vector{String}(data["permissions"]),
        data["created_at"],
        data["expires_at"]
    )
end

"""
    revoke_token!(manager::TokenManager, token_str::String)

Revokes (deletes) a token from the database.

# Arguments
- `manager::TokenManager`: The token manager instance.
- `token_str::String`: The token string to delete.
"""
function revoke_token!(manager::TokenManager, token_str::String)
    key_bytes = Vector{UInt8}(token_str)
    delete!(manager.db, key_bytes)
end

"""
    list_tokens(manager::TokenManager) -> Vector{TokenRecord}

Lists every token currently stored, expired or not -- `TokenManager`'s `db` has no column
families (a plain default-CF RocksDB, unlike `ProjectManager`), so this is a straight
`RocksDB.DBIterator(manager.db)` scan, no `cf=` needed. Backs both `similarity-search-ctl
log-tokens` (an audit listing) and `prune_expired!`'s own scan below.
"""
function list_tokens(manager::TokenManager)
    records = TokenRecord[]
    for (_, v) in RocksDB.DBIterator(manager.db)
        data = JSON3.read(v, Dict{String, Any})
        push!(records, TokenRecord(
            data["token_str"],
            data["user"],
            Vector{String}(data["permissions"]),
            data["created_at"],
            data["expires_at"],
        ))
    end
    return records
end

"""
    prune_expired!(manager::TokenManager) -> Int

Deletes every token whose `expires_at` is a real (non-`nothing`) timestamp already in the
past, returning how many were removed. Backs `similarity-search-ctl prune-tokens`.
`expires_at` is a free-form, user-supplied string at `create_token!` time (never validated
against one format anywhere in this codebase) -- a token whose `expires_at` doesn't parse
as a `Dates.DateTime` is skipped rather than aborting the whole sweep over one bad value.
"""
function prune_expired!(manager::TokenManager)
    now_ts = now(UTC)
    pruned = 0
    for rec in list_tokens(manager)
        rec.expires_at === nothing && continue
        expiry = try
            DateTime(rec.expires_at)
        catch
            continue
        end
        expiry < now_ts || continue
        revoke_token!(manager, rec.token_str)
        pruned += 1
    end
    return pruned
end

# ==========================================
# Authorization (PLAN.md §4.1)
# ==========================================

"""
    PERMISSION_OPS

The three operations a permission can grant, ordered from least to most: `:read` covers the
queries (`search`, `ftsearch`, `fetch`, `exists`) and the `GET` endpoints, `:write` covers
insertion, deletion, calibration and job submission, and `:admin` covers the creation and
deletion of datasets, the token endpoints, and the control of jobs.

An operation includes the ones before it for the same dataset: a token with `write:corpus`
may also read `corpus`, and `admin:*` grants everything.
"""
const PERMISSION_OPS = (:read, :write, :admin)

_op_rank(op::Symbol) = op === :read ? 1 : op === :write ? 2 : op === :admin ? 3 : 0

"""
    parse_permission(s::AbstractString) -> Union{Nothing, Tuple{Symbol, String}}

One permission string as the pair it means: `"write:corpus_es"` is `(:write, "corpus_es")`,
and the bare form `"read"` is `(:read, "*")`, which is how a permission that names no dataset
applies to every dataset. `nothing` for a string that is not a permission, so that a token
carrying one is not thereby granted something.
"""
function parse_permission(s::AbstractString)
    parts = split(s, ':'; limit=2)
    op = Symbol(strip(parts[1]))
    op in PERMISSION_OPS || return nothing
    scope = length(parts) == 2 ? strip(parts[2]) : "*"
    isempty(scope) && (scope = "*")
    return (op, String(scope))
end

"""
    normalize_permissions(permissions) -> Vector{String}

The same permissions in the canonical `operation:dataset` form, with unrecognized entries
removed. Applied when a token is created, so that what is stored is what will be compared.
"""
function normalize_permissions(permissions)
    out = String[]
    for p in permissions
        parsed = parse_permission(String(p))
        parsed === nothing && continue
        push!(out, string(parsed[1], ':', parsed[2]))
    end
    return out
end

"""
    has_permission(record::TokenRecord, op::Symbol, dataset::AbstractString) -> Bool

Whether `record` grants `op` on `dataset`. A permission matches when its operation is at
least `op` (see [`PERMISSION_OPS`](@ref)) and its scope is either `dataset` itself or `*`.

`dataset` is `"*"` for an endpoint that is not about one dataset -- listing datasets, the
job endpoints, the token endpoints -- and only a permission scoped to `*` matches it. A
token restricted to one dataset therefore uses that dataset's endpoints and cannot list what
else the server holds.
"""
function has_permission(record::TokenRecord, op::Symbol, dataset::AbstractString)
    want = _op_rank(op)
    want == 0 && return false
    for p in record.permissions
        parsed = parse_permission(p)
        parsed === nothing && continue
        granted_op, scope = parsed
        _op_rank(granted_op) >= want || continue
        (scope == "*" || scope == dataset) && return true
    end
    return false
end

"""
    is_expired(record::TokenRecord, at::DateTime=now(UTC)) -> Bool

Whether the token's `expires_at` is in the past. A token with no `expires_at` does not
expire. A timestamp that does not parse counts as expired: `create_token!` rejects such a
value, so a stored one is either damaged or was written by an older version, and refusing it
is the safe reading.
"""
function is_expired(record::TokenRecord, at::DateTime=now(UTC))
    record.expires_at === nothing && return false
    expiry = try
        DateTime(record.expires_at)
    catch
        return true
    end
    return expiry < at
end

"""
    validate_token(manager::TokenManager, token_str) -> Union{Nothing, TokenRecord}

The record for `token_str` if the token exists and has not expired, and `nothing` otherwise.
This is the function a request handler calls; [`get_token`](@ref) is the plain lookup and
does not consider expiry.
"""
function validate_token(manager::TokenManager, token_str::Union{Nothing, AbstractString})
    token_str === nothing && return nothing
    isempty(token_str) && return nothing
    record = get_token(manager, String(token_str))
    record === nothing && return nothing
    is_expired(record) && return nothing
    return record
end

"""
    any_token_exists(manager::TokenManager) -> Bool

Whether the database holds at least one token. `serve` uses it to refuse to start with
authentication enabled and no token defined, a state in which no request could be answered.
"""
any_token_exists(manager::TokenManager) = !isempty(list_tokens(manager))

end # module
