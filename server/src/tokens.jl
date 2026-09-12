module Tokens

using RocksDB
using JSON3
using Random
using Dates

export TokenManager, TokenRecord, open_token_manager, close_token_manager, create_token!, get_token, revoke_token!, list_tokens, prune_expired!

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
    
    data = JSON3.read(String(val_bytes), Dict{String, Any})
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
        data = JSON3.read(String(v), Dict{String, Any})
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

end # module
