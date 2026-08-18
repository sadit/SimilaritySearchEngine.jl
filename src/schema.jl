module Schema

using Avro
using JSON

export DataType, StringType, Int64Type, Float64Type, BoolType, TimestampType
export MetaSchemaField, MetaSchema, MetadataRecord
export get_field, matches_filter

@enum DataType begin
    StringType
    Int64Type
    Float64Type
    BoolType
    TimestampType
end

"""
    MetaSchemaField

Definition of a single typed field in the MetaSchema.

# Fields
- `name::String`: The field name.
- `type::DataType`: The data type of the field.
- `indexed::Bool`: Whether this field should be indexed.
"""
struct MetaSchemaField
    name::String
    type::DataType
    indexed::Bool
end

"""
    MetaSchema

Dataset-level schema declaring strongly typed and optionally indexed fields.

# Fields
- `version::Int`: The schema version.
- `fields::Vector{MetaSchemaField}`: The list of declared fields.
"""
struct MetaSchema
    version::Int
    fields::Vector{MetaSchemaField}
end

MetaSchema() = MetaSchema(1, MetaSchemaField[])

"""
    MetadataRecord

A single record's metadata. 

# Fields
- `doc_id::Int`: The integer ID of the document.
- `schema_version::Int`: The schema version this record corresponds to.
- `declared_fields::Dict{String, Any}`: Holds the typed values matching the `MetaSchema`.
- `extra::Vector{UInt8}`: Holds any arbitrary JSON-encoded byte data for flexibility.
"""
struct MetadataRecord
    doc_id::Int
    schema_version::Int
    declared_fields::Dict{String, Any}
    extra::Vector{UInt8} # JSON-encoded bytes for flexibility
end

function MetadataRecord(doc_id::Int, schema::MetaSchema, raw_dict::AbstractDict)
    declared = Dict{String, Any}()
    extra_dict = Dict{String, Any}()
    
    schema_field_names = Set(f.name for f in schema.fields)
    
    for (k, v) in raw_dict
        if k in schema_field_names
            # type validation based on schema field type
            field = nothing
            for f in schema.fields
                if f.name == k
                    field = f
                    break
                end
            end
            if field !== nothing
                if field.type == StringType && !(v isa String)
                    v = string(v)
                elseif field.type == Int64Type && !(v isa Integer)
                    v = parse(Int64, string(v))
                elseif field.type == Float64Type && !(v isa AbstractFloat)
                    v = parse(Float64, string(v))
                elseif field.type == BoolType && !(v isa Bool)
                    v = parse(Bool, string(v))
                end
            end
            declared[k] = v
        else
            extra_dict[k] = v
        end
    end
    
    extra_bytes = Vector{UInt8}(JSON.json(extra_dict))

    return MetadataRecord(doc_id, schema.version, declared, extra_bytes)
end

"""
    get_field(record::MetadataRecord, name::String)

Looks up `name` in `record`: checks the declared/typed fields first, falling back to
decoding `extra` (the free-form catch-all, see §4.5's declared+`extra` split) on demand.
Returns `nothing` if `name` isn't present in either.
"""
function get_field(record::MetadataRecord, name::String)
    haskey(record.declared_fields, name) && return record.declared_fields[name]
    isempty(record.extra) && return nothing
    # String(::Vector{UInt8}) takes ownership of its argument and empties it as a side
    # effect -- copy first so `record.extra` stays readable for any later call on the
    # same record (e.g. a second get_field, or the caller decoding `extra` itself).
    extra = JSON.parse(String(copy(record.extra)))
    return get(extra, name, nothing)
end

"""
    matches_filter(record::MetadataRecord, filter::Dict) -> Bool

Post-filter predicate for `/search`-family endpoints (PLAN.md §5.4). `filter` maps field
names to either a bare value (equality) or a spec dict supporting `gte`/`lte`/`gt`/`lt`
(range) and `in` (set membership). A field missing from the record fails the filter.
"""
function matches_filter(record::MetadataRecord, filter::AbstractDict)
    for (field, spec) in filter
        value = get_field(record, field)
        value === nothing && return false

        if spec isa AbstractDict
            haskey(spec, "gte") && !(value >= spec["gte"]) && return false
            haskey(spec, "lte") && !(value <= spec["lte"]) && return false
            haskey(spec, "gt") && !(value > spec["gt"]) && return false
            haskey(spec, "lt") && !(value < spec["lt"]) && return false
            haskey(spec, "in") && !(value in spec["in"]) && return false
        elseif value != spec
            return false
        end
    end
    return true
end

end # module
