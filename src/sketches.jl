# Sketched dense projects (PLAN.md §5.14). Part of `IndexEngine`, next to `quantization.jl`: a
# sketched project keeps the state of a quantized one (`QuantState`), with a `SketchEncoder` in
# place of an `SQEncoder`. Its codes are sketches of `SimilaritySearch.Projections`: an
# `AnchoredDistantHyperplanes` model encoded by a `QuantSketch` at 1, 2, 4 or 8 bits per
# hyperplane. The index of the project is built over the sketches, with the distance of the
# `QuantSketch`.

using SimilaritySearch.Projections: AnchoredDistantHyperplanes, QuantSketch, quantsketch, sketchsize, outdim

export SketchInfo

"""
    SketchEncoder

The encoder of a sketched project: the fitted `QuantSketch` (its `AnchoredDistantHyperplanes`
model keeps only its anchor vectors), the mean subtracted from every vector when the project
centers them, and the dimension of the vectors.
"""
struct SketchEncoder{QS<:QuantSketch}
    qs::QS
    mean::Union{Nothing,Vector{Float32}}
    dim::Int
end

# The element type of the codes: `UInt64` words of sign bits at 1 bit, `UInt8` codes otherwise.
_sketch_codetype(bits::Integer) = bits == 1 ? UInt64 : UInt8

"""
    sketch_vector(encoder::SketchEncoder, v) -> Vector

The sketch of the vector `v`: its margins against the anchors of the encoder, after subtracting
the mean of a project that centers its vectors, quantized at the bits of the encoder.
"""
function sketch_vector(enc::SketchEncoder, v::AbstractVector)
    length(v) == enc.dim ||
        wrong_dimension(enc.dim, length(v), "vector dimension $(length(v)) does not match the project's $(enc.dim)")
    x = enc.mean === nothing ? v : Vector{Float32}(v) .- enc.mean
    return quantsketch(enc.qs, x)
end

# The model of `m` with only the anchors that its hyperplanes use. The library keeps the whole
# fitting sample as the reference set, and the project saves the encoder.
function _compact_anchors(m::AnchoredDistantHyperplanes)
    used = sort!(unique!(vcat(first.(m.H), last.(m.H))))
    position = Dict(id => i for (i, id) in enumerate(used))
    A = Matrix{Float32}(undef, length(m.C[first(used)]), length(used))
    for (j, i) in enumerate(used)
        A[:, j] .= m.C[i]
    end
    C = MatrixDatabase(A)
    H = [position[a] => position[b] for (a, b) in m.H]
    anchor = m.anchor isa AbstractVector ? Vector{Float32}(m.anchor) : m.anchor
    return AnchoredDistantHyperplanes(m.dist, H, C, anchor)
end

"""
    fit_encoder(spec::Sketched, distance, sample::AbstractMatrix{Float32}, rng) -> SketchEncoder

The encoder of a sketched project, fitted on `sample` (one staged vector per column, more than
64 of them): the anchors of `spec.hyperplanes` hyperplanes under `distance`, and, above 1 bit,
the range of the codes. The library draws its samples from the default random generator, so
`rng` selects only the rows of `sample`.
"""
function fit_encoder(spec::Sketched, distance, sample::AbstractMatrix{Float32}, rng::AbstractRNG)
    dim, n = size(sample)
    n > 64 || not_trained("a sketched project fits its encoder on more than 64 vectors; the sample has $n")
    mean = spec.center ? vec(sum(sample; dims=2)) ./ Float32(n) : nothing
    db = MatrixDatabase(mean === nothing ? sample : sample .- mean)
    henc = min(spec.henc, 64 * fld(n - 1, 64))
    model = AnchoredDistantHyperplanes(distance, db, spec.hyperplanes; anchorpolicy=spec.anchorpolicy,
                                       hsel=spec.hsel, henc, entquantile=spec.entquantile, verbose=false)
    outdim(model) == spec.hyperplanes || invalid_option(:sketch,
        "the fitting kept $(outdim(model)) of $(spec.hyperplanes) hyperplanes: too few candidate pairs had an entropy above " *
        "the cutoff. Raise `hsel` ($(spec.hsel)) or lower `entquantile` ($(spec.entquantile))")
    qs = QuantSketch(_compact_anchors(model), spec.bits, db)
    return SketchEncoder(qs, mean, dim)
end

"""
    install_encoder!(engine, encoder::SketchEncoder, codes, params, per_vector_params)

Gives the sketched `engine` its encoder and an index over the sketches in `codes`, under the
distance of the `QuantSketch`. `params` and `per_vector_params` are not used.
"""
function install_encoder!(engine::AbstractSearchEngine, encoder::SketchEncoder, codes::MMapMatrixDatabase,
                          params, per_vector_params)
    q = engine.quant
    write_lock(engine.lock) do
        q.encoder = encoder
        q.codes = codes
        q.params = nothing
        _install_index!(engine.backend, SimilaritySearch.distance(encoder.qs), codes)
    end
    return engine
end

"""
    encode_vectors(encoder::SketchEncoder, vectors) -> Vector

The sketches of `vectors`, computed in parallel.
"""
function encode_vectors(encoder::SketchEncoder, vectors::AbstractVector)
    n = length(vectors)
    codes = Vector{Vector{_sketch_codetype(sketch_bits(encoder))}}(undef, n)
    @BATCHES getminbatch(n) for i in 1:n
        codes[i] = sketch_vector(encoder, vectors[i])
    end
    return codes
end

sketch_bits(::SketchEncoder{<:QuantSketch{B}}) where {B} = B

fit_samplesize(spec::Sketched) = spec.samplesize

# The query of a sketched project is sketched as the vectors are.
_prepare_query(enc::SketchEncoder, vector::AbstractVector{Float32}) = sketch_vector(enc, vector)

"""
    issketched(engine) -> Bool

Whether `engine` is a dense engine whose vectors are stored as sketches (PLAN.md §5.14).
"""
issketched(engine::AbstractSearchEngine) = isquantized(engine) && engine.quant.spec isa Sketched

"""
    SketchInfo

What `sketch_info(h)` reports about the sketches of a dense project (PLAN.md §5.14).

- `storage`: the [`Sketched`](@ref) storage the project was created with.
- `fitted`: whether the encoder exists. It is fitted at the first `index!` with more than 64
  staged vectors.
- `dimension`: the dimension of the vectors, or `nothing` before the encoder exists.
- `min`, `max`: the range of the codes above 1 bit, or `nothing`.
- `bytes_per_vector`: the bytes of one sketch.
- `bytes_on_disk_per_vector`: the bytes of one sketch, and of its `Float32` vector when the
  project keeps them, or `nothing` before the encoder exists.
"""
struct SketchInfo
    storage::Sketched
    fitted::Bool
    dimension::Union{Nothing,Int}
    min::Union{Nothing,Float32}
    max::Union{Nothing,Float32}
    bytes_per_vector::Int
    bytes_on_disk_per_vector::Union{Nothing,Int}
end

"""
    sketch_info(q::QuantState) -> SketchInfo
"""
function sketch_info(q::QuantState)
    spec = q.spec::Sketched
    enc = q.encoder
    bytes = spec.hyperplanes * spec.bits ÷ 8
    dim = enc === nothing ? nothing : enc.dim
    lo = hi = nothing
    if enc !== nothing && spec.bits > 1
        lo, hi = enc.qs.minmax
    end
    disk = dim === nothing ? nothing : bytes + (spec.keep_vectors ? 4 * dim : 0)
    return SketchInfo(spec, enc !== nothing, dim, lo, hi, bytes, disk)
end

"""
    create_sketch_codes(path, encoder::SketchEncoder) -> MMapMatrixDatabase

The empty file of the sketches of a project: one column of `sketchsize` words per vector.
"""
create_sketch_codes(path::String, encoder::SketchEncoder) =
    MMapMatrixDatabase(path, sketchsize(encoder.qs), _sketch_codetype(sketch_bits(encoder)))
