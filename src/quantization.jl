# Quantized dense projects (issue #1). Part of `IndexEngine`: the encoder and the index over the
# codes live in memory, and `embedded.jl` reads and writes the files that hold them.
#
# A quantized project stores each vector as the codes of a scalar quantizer of
# `SimilaritySearch.ScalarQuant`, and searches with the distances of that module. The index is
# the one the project was created with (`SearchGraph`, `ExhaustiveSearch` or
# `ParallelExhaustiveSearch`), built over a `QuantDatabase` instead of a `VectorDatabase`.
#
# Append stages the original vectors. `index!` fits the encoder on a sample of the staged vectors
# the first time, encodes the staged vectors, and inserts the codes. Before the first `index!`, a
# quantized project has no index to search.

using SimilaritySearch.ScalarQuant: ScalarQuant, SQEncoder, QuantDatabase, SQMinC, SQVec
using SimilaritySearch: Projections, MMapMatrixDatabase
using SimilaritySearch: encode, encodequery

export QuantizerInfo

# The library module of a quantizer: one global range or one range per vector, at 8, 4 or 2 bits.
function _quant_module(q::SQ)
    glob = q.range isa GlobalRange
    q.bits == 8 && return glob ? ScalarQuant.SQgu8 : ScalarQuant.SQu8
    q.bits == 4 && return glob ? ScalarQuant.SQgu4 : ScalarQuant.SQu4
    return glob ? ScalarQuant.SQgu2 : ScalarQuant.SQu2
end

"""
    QuantizerInfo

What `quantizer_info(h)` reports about the quantizer of a dense project.

- `storage`: the [`Quantized`](@ref) storage the project was created with.
- `fitted`: whether the encoder exists. A global quantizer without `minmax` is fitted at the
  first `index!`.
- `min`, `max`: the range of a fitted global quantizer, or `nothing`.
- `dimension`: the dimension of the vectors, or `nothing` before the encoder exists.
- `bytes_per_vector`: the bytes the codes of one vector take, including the 8 bytes of the
  range of a per-vector quantizer, or `nothing` before the encoder exists.
"""
struct QuantizerInfo
    storage::Quantized
    fitted::Bool
    min::Union{Nothing,Float32}
    max::Union{Nothing,Float32}
    dimension::Union{Nothing,Int}
    bytes_per_vector::Union{Nothing,Int}
end

"""
    QuantState

The codes of a dense engine whose vectors are not stored as `Float32` values: the storage it
was created with, [`Quantized`](@ref) or [`Sketched`](@ref) (`sketches.jl`), the distance the
caller asked for, the encoder once it is fitted (an `SQEncoder` or a `SketchEncoder`), the files
that hold the codes, and how many staged vectors are not encoded yet.

`codes` is the `MMapMatrixDatabase{UInt8}` of the codes. `params` is the
`MMapMatrixDatabase{Float32}` of the `(min, c)` pair of each vector of a per-vector quantizer,
and `nothing` for a global one. Both are `nothing` before the encoder exists.
"""
mutable struct QuantState
    spec::Union{Quantized,Sketched}
    distance::SimilaritySearch.PreMetric
    encoder::Any
    codes::Union{Nothing,MMapMatrixDatabase}
    params::Union{Nothing,MMapMatrixDatabase}
    npending::Int
end

"""
    quantized_distance(distance) -> PreMetric

The distance of `SimilaritySearch.ScalarQuant` that a quantized project searches with, for the
distance the project was created with. Raises `InvalidOption` for a distance that has no
counterpart over codes.
"""
quantized_distance(::SimilaritySearch.Dist.SqL2) = ScalarQuant.SqL2()
quantized_distance(::SimilaritySearch.Dist.L2) = ScalarQuant.L2()
quantized_distance(::SimilaritySearch.Dist.L1) = ScalarQuant.L1()
quantized_distance(::SimilaritySearch.Dist.Cosine) = ScalarQuant.Cosine()
quantized_distance(::SimilaritySearch.Dist.NormCosine) = ScalarQuant.NormCosine()
quantized_distance(d) =
    invalid_option(:quantizer, "a quantized project searches with SqL2, L2, L1, Cosine or NormCosine; " *
          "$(nameof(typeof(d))) has no distance over codes")

"""
    enable_quantization!(engine::AbstractSearchEngine, spec::Union{Quantized,Sketched}, distance) -> DenseEngine

Returns `engine` as an engine that stores codes: a quantized or a sketched engine. `engine` must be empty and have no partitions yet, and
`distance` is the distance it was created with.

An exact engine is rebuilt around an `ExactBackend` whose type names the index type without its
parameters (`ExactBackend{ExhaustiveSearch}`). `install_encoder!` replaces the empty index over
`Float32` vectors with an index over codes, whose concrete type is different, and the field of a
backend whose parameter is concrete cannot hold it.
"""
function enable_quantization!(engine::AbstractSearchEngine, spec::Union{Quantized,Sketched}, distance)
    spec isa Quantized && quantized_distance(distance)
    if engine.backend isa ExactBackend
        b = engine.backend
        T = b.index isa ParallelExhaustiveSearch ? ParallelExhaustiveSearch : ExhaustiveSearch
        engine = DenseEngine(ExactBackend{T}(b.index, b.ctx), engine.search_ctx_pool, engine.deleted_ids, engine.lock)
    end
    engine.quant = QuantState(spec, distance, nothing, nothing, nothing, 0)
    return engine
end


# The number of vectors whose codes are stored.
quantized_length(q::QuantState) = q.codes === nothing ? 0 : length(q.codes)

"""
    check_quantized_dimension(spec, dim)

Raises `InvalidOption` when vectors of dimension `dim` cannot be stored with `spec`.
"""
function check_quantized_dimension(spec::Quantized, dim::Integer)
    bits = spec.quantizer.bits
    perbyte = 8 ÷ bits
    dim % perbyte == 0 ||
        invalid_option(:quantizer, "a quantizer of $bits bits packs $perbyte coordinates per byte, so the dimension " *
              "must be a multiple of $perbyte; the vectors have dimension $dim")
    spec.rotation isa HadamardRotation && !ispow2(dim) &&
        invalid_option(:quantizer, "HadamardRotation needs a dimension that is a power of two; the vectors have " *
              "dimension $dim. Use QRRotation()")
    return nothing
end

_rotation(::NoRotation, dim::Int, rng::AbstractRNG) = nothing
_rotation(::QRRotation, dim::Int, rng::AbstractRNG) = Projections.qr(rng, Float32, dim, dim)
_rotation(::HadamardRotation, dim::Int, rng::AbstractRNG) = Projections.RandomizedHadamard(dim; rng)

"""
    fit_encoder(spec, distance, sample::AbstractMatrix{Float32}, rng) -> SQEncoder

The encoder of a quantized project. `sample` holds one staged vector per column. A global
quantizer estimates its range from `sample` unless its `minmax` gives it. A per-vector
quantizer calibrates its range policy on `sample`.
"""
function fit_encoder(spec::Quantized, distance, sample::AbstractMatrix{Float32}, rng::AbstractRNG)
    dim = size(sample, 1)
    check_quantized_dimension(spec, dim)
    quant = _quant_module(spec.quantizer)
    rot = _rotation(spec.rotation, dim, rng)
    dist = quantized_distance(distance)
    range = spec.quantizer.range
    if range isa GlobalRange && range.minmax !== nothing
        return SQEncoder(quant, rot, dim; dist, minmax=range.minmax)
    end
    return SQEncoder(quant, rot, sample; dist, samplesize=size(sample, 2), rng)
end

"""
    install_encoder!(engine, encoder, codes, params, per_vector_params)

Gives the quantized `engine` its encoder and an index over the codes in `codes`. `params` and
`per_vector_params` are the file and the values of the `(min, c)` pairs of a per-vector
quantizer, or `nothing` and `SQMinC[]` for a global one. The index replaces the empty one the
engine was created with. A graph keeps its `BeamSearch`, its hints and its context.
"""
function install_encoder!(engine::AbstractSearchEngine, encoder::SQEncoder, codes::MMapMatrixDatabase,
                          params::Union{Nothing,MMapMatrixDatabase}, per_vector_params::Vector{SQMinC})
    q = engine.quant
    B = ScalarQuant.codewidth(encoder)
    E = ScalarQuant.isglobal(encoder) ? encoder.E : per_vector_params
    db = QuantDatabase{B}(E, codes; dim=encoder.dim)
    write_lock(engine.lock) do
        q.encoder = encoder
        q.codes = codes
        q.params = params
        _install_index!(engine.backend, encoder.dist, db)
    end
    return engine
end

function _install_index!(backend::GraphBackend, dist, db)
    old = backend.index
    index = SearchGraph(dist, db)
    index.algo[] = old.algo[]
    append!(index.hints, old.hints)
    backend.index = index
end

function _install_index!(backend::ExactBackend, dist, db)
    backend.index = backend.index isa ParallelExhaustiveSearch ? ParallelExhaustiveSearch(dist, db) :
                                                              ExhaustiveSearch(dist, db)
end

"""
    encode_vectors(encoder, vectors) -> Vector{SQVec}

The codes of `vectors`, computed in parallel.
"""
function encode_vectors(encoder::SQEncoder, vectors::AbstractVector)
    n = length(vectors)
    first_code = encode(encoder, vectors[1])
    codes = Vector{typeof(first_code)}(undef, n)
    codes[1] = first_code
    n == 1 && return codes
    @BATCHES getminbatch(n) for i in 2:n
        codes[i] = encode(encoder, vectors[i])
    end
    return codes
end

"""
    insert_codes!(engine, codes) -> UnitRange{Int}

Appends `codes` to the index of the quantized `engine`, and the range of each code of a
per-vector quantizer to its file. Returns the ids the codes received. A graph stores them as its
backlog, which `index!` connects. An exact index searches them at once.
"""
function insert_codes!(engine::AbstractSearchEngine, codes::AbstractVector)
    q = engine.quant
    write_lock(engine.lock) do
        index = engine.backend.index
        sp = length(database(index)) + 1
        for c in codes
            if engine.backend isa GraphBackend
                push_item!(index.db, c)
            else
                push_item!(index, engine.backend.ctx, c)
            end
            q.params === nothing || push_item!(q.params, Float32[c.E.min, c.E.c])
        end
        q.npending -= length(codes)
        return sp:length(database(index))
    end
end

"""
    stage_quantized!(engine)

Counts one staged vector of the quantized `engine`. The vector itself is stored by the caller,
and `index!` encodes it.
"""
stage_quantized!(engine::AbstractSearchEngine) = write_lock(() -> (engine.quant.npending += 1), engine.lock)

"""
    prepare_query(engine, vector) -> AbstractVector

The query a quantized or sketched `engine` evaluates against its codes: for a quantizer, the
vector rotated and prepared by `encodequery`; for a sketch, its sketch. Before the encoder exists
the index is empty, and the vector is returned as it is.
"""
function prepare_query(engine::AbstractSearchEngine, vector::AbstractVector{Float32})
    enc = engine.quant.encoder
    enc === nothing && return vector
    length(vector) == enc.dim ||
        wrong_dimension(enc.dim, length(vector), "query dimension $(length(vector)) does not match the project's $(enc.dim)")
    return _prepare_query(enc, vector)
end

_prepare_query(enc::SQEncoder, vector::AbstractVector{Float32}) = encodequery(enc, vector)

"The number of staged vectors that the first `index!` samples to fit the encoder."
fit_samplesize(spec::Quantized) = spec.quantizer.range.samplesize

"""
    quantizer_info(q::QuantState) -> QuantizerInfo
"""
function quantizer_info(q::QuantState)
    spec = q.spec
    enc = q.encoder
    bits = spec.quantizer.bits
    glob = spec.quantizer.range isa GlobalRange
    lo = hi = nothing
    if enc !== nothing && glob
        lo = enc.E.min
        hi = enc.E.min + Float32(2^bits - 1) * enc.E.c
    end
    dim = enc === nothing ? nothing : enc.dim
    bpv = dim === nothing ? nothing : cld(dim * bits, 8) + (glob ? 0 : 8)
    QuantizerInfo(spec, enc !== nothing, lo, hi, dim, bpv)
end

"""
    close_quantized!(engine)

Closes the files of the codes of a quantized engine.
"""
function close_quantized!(engine::AbstractSearchEngine)
    q = engine.quant
    q === nothing && return nothing
    q.codes === nothing || close(q.codes)
    q.params === nothing || close(q.params)
    return nothing
end

"""
    flush_quantized!(engine)

Makes the codes appended so far durable.
"""
function flush_quantized!(engine::AbstractSearchEngine)
    q = engine.quant
    q.codes === nothing || SimilaritySearch.flush(q.codes)
    q.params === nothing || SimilaritySearch.flush(q.params)
    return nothing
end
