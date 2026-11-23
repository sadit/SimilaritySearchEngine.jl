# A plugin of the server's tests (PLAN.md §5.13): a key source and a kind of follower, declared
# with the wire names that the server reads. It depends on the engine only.
module SSETestPlugin

using SimilaritySearchEngine
const SSE = SimilaritySearchEngine

"The settings that `setup` received from `[plugins.SSETestPlugin.settings]`."
const SETTINGS = Dict{String,Any}()
setup(settings::Dict{String,Any}) = (empty!(SETTINGS); merge!(SETTINGS, settings); nothing)

"The decade of an integer field of `meta`: 1987 is 1980s. An item without it has no key."
struct Decade <: SSE.KeySource
    field::String
end
Base.show(io::IO, s::Decade) = print(io, "Decade(", repr(s.field), ")")

function SSE.keys_of(s::Decade, item::SSE.StoredItem)
    year = get(item.meta, s.field, nothing)
    year isa Integer ? string(10 * fld(year, 10), "s") : nothing
end

"""
A kind that counts the items of each key. It saves the keys of each processed item after each
block, so processing an item twice gives the same counts.
"""
struct KeyCounts <: SSE.FollowerSpec
    source::SSE.KeySource
    min_count::Int
end

mutable struct KeyCountsState
    store::Any
    min_count::Int
    items::Dict{Int,Vector{String}}
end

SSE.create_follower_state(spec::KeyCounts, leader, store, db) = KeyCountsState(store, spec.min_count, Dict{Int,Vector{String}}())
SSE.open_follower_state(spec::KeyCounts, leader, store, db) =
    KeyCountsState(store, spec.min_count, SSE.Persistence.load_field(store, :key_counts_items, Dict{Int,Vector{String}}()))

function SSE.process_items!(st::KeyCountsState, leader, ids, keys)
    for (id, ks) in zip(ids, keys)
        st.items[id] = ks
    end
    SSE.Persistence.save_field!(st.store, :key_counts_items, st.items)
end

function SSE.follower_stats(st::KeyCountsState)
    counts = Dict{String,Int}()
    for ks in values(st.items), k in ks
        counts[k] = get(counts, k, 0) + 1
    end
    return (items=length(st.items), keys=length(counts), frequent=count(>=(st.min_count), values(counts)))
end

const KEY_SOURCES = Dict{String,Function}("decade" => opts -> Decade(String(opts["field"])))
const FOLLOWER_KINDS = Dict{String,Function}(
    "key_counts" => (source, opts) -> KeyCounts(source, Int(get(opts, "min_count", 1))))

end
