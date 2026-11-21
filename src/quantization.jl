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

export ScalarQuantizer, QuantizerInfo

const _QUANT_KINDS = (:sqgu8, :sqgu4, :sqgu2, :squ8, :squ4, :squ2)
const _QUANT_ROTATIONS = (:none, :qr, :hadamard)

"""
    ScalarQuantizer(kind=:sqgu8; rotation=:none, keep_vectors=false, minmax=nothing, samplesize=4096)

The scalar quantizer of a dense project, given to `create_project(...; quantizer=...)`. The
project stores each vector as `bits` codes per coordinate instead of a `Float32`, and searches
with the distances of `SimilaritySearch.ScalarQuant`.

`kind` names the family and the number of bits:

- `:sqgu8`, `:sqgu4`, `:sqgu2`: one range for every coordinate of every vector. The range is
  estimated from a sample of the staged vectors at the first `index!`, or given as `minmax`.
- `:squ8`, `:squ4`, `:squ2`: one range for each vector, taken from its own coordinates. It
  costs 8 more bytes per vector, and it needs no sample.

A vector of dimension 1024 takes 1024 bytes at 8 bits, 512 at 4 bits and 256 at 2 bits,
against 4096 bytes in `Float32`. The dimension must be a multiple of 2 at 4 bits and of 4 at 2
bits.

`rotation` is applied to every vector and every query before quantization:

- `:none`, the default, quantizes the coordinates as they are. It works for the output of most
  modern neural embedding models, whose coordinates have similar scales. On SISAP 2025
  `ccnews`, a QR rotation changed recall@10 by less than 0.01 at every width.
- `:qr` applies a random orthogonal matrix, which gives every coordinate the same scale. Use it
  when the coordinates have different scales or heavy tails: vectors that are not normalized,
  vectors from several models or pipelines in one project, features that were not produced by
  a neural model, or a recall clearly below the exact search on a sample of queries. It costs
  `dim²` operations per query and per inserted item, and `4 dim²` bytes of storage.
- `:hadamard` is a random rotation that costs `dim log dim` operations. The dimension must be a
  power of two.

`keep_vectors=true` keeps the original `Float32` vectors on disk next to the codes, so that
`fetch_items` returns them. With `false`, the original vectors are removed once they are
encoded, and `fetch_items` returns `nothing` as the payload.

`minmax=(lo, hi)` fixes the range of a global quantizer, and nothing is fitted. `samplesize` is
the number of staged vectors the range is estimated from.
"""
struct ScalarQuantizer
    kind::Symbol
    rotation::Symbol
    keep_vectors::Bool
    minmax::Union{Nothing,Tuple{Float32,Float32}}
    samplesize::Int

    function ScalarQuantizer(kind::Symbol, rotation::Symbol, keep_vectors::Bool, minmax, samplesize::Integer)
        kind in _QUANT_KINDS ||
            invalid_option(:quantizer, "unknown quantizer kind $(repr(kind)); expected one of $(join(_QUANT_KINDS, ", "))")
        rotation in _QUANT_ROTATIONS ||
            invalid_option(:quantizer, "unknown rotation $(repr(rotation)); expected one of $(join(_QUANT_ROTATIONS, ", "))")
        samplesize >= 1 || invalid_option(:quantizer, "samplesize must be at least 1; got $samplesize")
        mm = if minmax === nothing
            nothing
        else
            _quant_global(kind) ||
                invalid_option(:quantizer, "`minmax` applies to a global quantizer (sqgu8, sqgu4, sqgu2); " *
                      "$kind takes the range of each vector from its own coordinates")
            length(minmax) == 2 && minmax[1] < minmax[2] ||
                invalid_option(:quantizer, "`minmax` must be two numbers (lo, hi) with lo < hi; got $(repr(minmax))")
            (Float32(minmax[1]), Float32(minmax[2]))
        end
        new(kind, rotation, keep_vectors, mm, Int(samplesize))
    end
end

ScalarQuantizer(kind::Symbol=:sqgu8; rotation::Symbol=:none, keep_vectors::Bool=false,
                minmax=nothing, samplesize::Integer=4096) =
    ScalarQuantizer(kind, rotation, keep_vectors, minmax, samplesize)

Base.:(==)(a::ScalarQuantizer, b::ScalarQuantizer) =
    a.kind == b.kind && a.rotation == b.rotation && a.keep_vectors == b.keep_vectors &&
    a.minmax == b.minmax && a.samplesize == b.samplesize

_quant_global(kind::Symbol) = kind in (:sqgu8, :sqgu4, :sqgu2)
_quant_bits(kind::Symbol) = kind in (:sqgu8, :squ8) ? 8 : kind in (:sqgu4, :squ4) ? 4 : 2
_quant_module(kind::Symbol) =
    kind === :sqgu8 ? ScalarQuant.SQgu8 : kind === :sqgu4 ? ScalarQuant.SQgu4 : kind === :sqgu2 ? ScalarQuant.SQgu2 :
    kind === :squ8 ? ScalarQuant.SQu8 : kind === :squ4 ? ScalarQuant.SQu4 : ScalarQuant.SQu2

"""
    QuantizerInfo

What `quantizer_info(h)` reports about the quantizer of a dense project.

- `kind`, `rotation`, `keep_vectors`: as given to [`ScalarQuantizer`](@ref).
- `family`: `:global` (one range for the project) or `:per_vector`.
- `bits`: bits per coordinate.
- `fitted`: whether the encoder exists. A global quantizer without `minmax` is fitted at the
  first `index!`.
- `min`, `max`: the range of a fitted global quantizer, or `nothing`.
- `dimension`: the dimension of the vectors, or `nothing` before the encoder exists.
- `bytes_per_vector`: the bytes the codes of one vector take, including the 8 bytes of the
  range of a per-vector quantizer, or `nothing` before the encoder exists.
"""
struct QuantizerInfo
    kind::Symbol
    family::Symbol
    bits::Int
    rotation::Symbol
    keep_vectors::Bool
    fitted::Bool
    min::Union{Nothing,Float32}
    max::Union{Nothing,Float32}
    dimension::Union{Nothing,Int}
    bytes_per_vector::Union{Nothing,Int}
end

"""
    QuantState

The quantizer of a dense engine: the specification it was created with, the distance the caller
asked for, the encoder once it is fitted, the files that hold the codes, and how many staged
vectors are not encoded yet.

`codes` is the `MMapMatrixDatabase{UInt8}` of the codes. `params` is the
`MMapMatrixDatabase{Float32}` of the `(min, c)` pair of each vector of a per-vector quantizer,
and `nothing` for a global one. Both are `nothing` before the encoder exists.
"""
mutable struct QuantState
    spec::ScalarQuantizer
    distance::SimilaritySearch.PreMetric
    encoder::Union{Nothing,SQEncoder}
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
    enable_quantization!(engine::AbstractSearchEngine, spec::ScalarQuantizer, distance) -> DenseEngine

Returns `engine` as a quantized engine. `engine` must be empty and have no partitions yet, and
`distance` is the distance it was created with.

An exact engine is rebuilt around an `ExactBackend` whose type names the index type without its
parameters (`ExactBackend{ExhaustiveSearch}`). `install_encoder!` replaces the empty index over
`Float32` vectors with an index over codes, whose concrete type is different, and the field of a
backend whose parameter is concrete cannot hold it.
"""
function enable_quantization!(engine::AbstractSearchEngine, spec::ScalarQuantizer, distance)
    quantized_distance(distance)
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
function check_quantized_dimension(spec::ScalarQuantizer, dim::Integer)
    bits = _quant_bits(spec.kind)
    perbyte = 8 ÷ bits
    dim % perbyte == 0 ||
        invalid_option(:quantizer, "$(spec.kind) packs $perbyte coordinates per byte, so the dimension must be " *
              "a multiple of $perbyte; the vectors have dimension $dim")
    spec.rotation === :hadamard && !ispow2(dim) &&
        invalid_option(:quantizer, "rotation=:hadamard needs a dimension that is a power of two; the vectors have " *
              "dimension $dim. Use rotation=:qr")
    return nothing
end

function _rotation(spec::ScalarQuantizer, dim::Int, rng::AbstractRNG)
    spec.rotation === :none && return nothing
    spec.rotation === :qr && return Projections.qr(rng, Float32, dim, dim)
    return Projections.RandomizedHadamard(dim; rng)
end

"""
    fit_encoder(spec, distance, sample::AbstractMatrix{Float32}, rng) -> SQEncoder

The encoder of a quantized project. `sample` holds one staged vector per column. A global
quantizer estimates its range from `sample` unless `spec.minmax` gives it. A per-vector
quantizer calibrates its range policy on `sample`.
"""
function fit_encoder(spec::ScalarQuantizer, distance, sample::AbstractMatrix{Float32}, rng::AbstractRNG)
    dim = size(sample, 1)
    check_quantized_dimension(spec, dim)
    quant = _quant_module(spec.kind)
    rot = _rotation(spec, dim, rng)
    dist = quantized_distance(distance)
    if spec.minmax !== nothing
        return SQEncoder(quant, rot, dim; dist, minmax=spec.minmax)
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

The query a quantized `engine` evaluates against its codes: the vector rotated and prepared by
`encodequery`. Before the encoder exists the index is empty, and the vector is returned as it
is.
"""
function prepare_query(engine::AbstractSearchEngine, vector::AbstractVector{Float32})
    enc = engine.quant.encoder
    enc === nothing && return vector
    length(vector) == enc.dim ||
        wrong_dimension(enc.dim, length(vector), "query dimension $(length(vector)) does not match the project's $(enc.dim)")
    return encodequery(enc, vector)
end

"""
    quantizer_info(q::QuantState) -> QuantizerInfo
"""
function quantizer_info(q::QuantState)
    spec = q.spec
    enc = q.encoder
    bits = _quant_bits(spec.kind)
    glob = _quant_global(spec.kind)
    lo = hi = nothing
    if enc !== nothing && glob
        lo = enc.E.min
        hi = enc.E.min + Float32(2^bits - 1) * enc.E.c
    end
    dim = enc === nothing ? nothing : enc.dim
    bpv = dim === nothing ? nothing : cld(dim * bits, 8) + (glob ? 0 : 8)
    QuantizerInfo(spec.kind, glob ? :global : :per_vector, bits, spec.rotation, spec.keep_vectors,
                  enc !== nothing, lo, hi, dim, bpv)
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
