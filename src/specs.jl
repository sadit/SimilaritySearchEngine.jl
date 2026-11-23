# The specification of a project (issue #2). Part of `IndexEngine`.
#
# A project is described by one value: what it holds, the index that holds it, and, for dense
# vectors, how the vectors are stored. Each type holds only its own options, so a combination
# that does not exist cannot be written. `create_project` takes the value, saves it in the field
# `:spec`, and `open_project` reads it back.

export ProjectSpec, DenseVectors, SparseVectors, TextDocuments
export DenseIndex, Graph, Exhaustive, TextIndex, BM25, VectorSpace
export DenseStorage, Float32Vectors, Quantized, SQ, QuantRange, GlobalRange, PerVectorRange, Sketched
export Rotation, NoRotation, QRRotation, HadamardRotation

"""
    ProjectSpec

What a project holds and how it is indexed. One of [`DenseVectors`](@ref),
[`SparseVectors`](@ref) and [`TextDocuments`](@ref).
"""
abstract type ProjectSpec end

# ---------------------------------------------------------------------------------------
# Dense indexes

"""
    DenseIndex

The index of a [`DenseVectors`](@ref) project: [`Graph`](@ref) or [`Exhaustive`](@ref).
"""
abstract type DenseIndex end

"""
    Graph(; minrecall=0.9)

A `SearchGraph`: an approximate index that finds the neighbors of a query with a beam search
over a graph of the items.

`minrecall` is the recall the beam search is tuned to reach as the project grows. It is a
target and not a floor: since SimilaritySearch 1.6 the tuning accepts a configuration slightly
below it when the cost is much lower. `calibrate!` reports the recall of each configuration.
`minrecall=nothing` disables the tuning, and the graph keeps the `BeamSearch` it has.
"""
struct Graph <: DenseIndex
    minrecall::Union{Nothing,Float32}

    function Graph(minrecall::Union{Nothing,Real})
        minrecall === nothing || 0 < minrecall <= 1 ||
            invalid_option(:minrecall, "minrecall must be in (0, 1], or nothing; got $minrecall")
        new(minrecall === nothing ? nothing : Float32(minrecall))
    end
end

Graph(; minrecall::Union{Nothing,Real}=0.9) = Graph(minrecall)

"""
    Exhaustive(; parallel=false)

An exact index: each query is compared with every item. `parallel=true` compares one query
with the items in parallel (`ParallelExhaustiveSearch`), which lowers the time of one query and
does not change the results.
"""
struct Exhaustive <: DenseIndex
    parallel::Bool
end

Exhaustive(; parallel::Bool=false) = Exhaustive(parallel)

# ---------------------------------------------------------------------------------------
# Dense storage

"""
    DenseStorage

How a [`DenseVectors`](@ref) project stores its vectors: [`Float32Vectors`](@ref),
[`Quantized`](@ref) or [`Sketched`](@ref).
"""
abstract type DenseStorage end

"""
    Float32Vectors()

Each vector is stored as `Float32` values. This is the default storage of a dense project.
"""
struct Float32Vectors <: DenseStorage end

"""
    QuantRange

How a scalar quantizer maps a coordinate to a code: [`GlobalRange`](@ref) or
[`PerVectorRange`](@ref).
"""
abstract type QuantRange end

"""
    GlobalRange(; minmax=nothing, samplesize=4096)

One range `(lo, hi)` for every coordinate of every vector. `minmax=(lo, hi)` fixes it.
Without `minmax`, the range is estimated from `samplesize` of the staged vectors at the first
`index!`, or by `fit_quantizer!`.
"""
struct GlobalRange <: QuantRange
    minmax::Union{Nothing,Tuple{Float32,Float32}}
    samplesize::Int

    function GlobalRange(minmax, samplesize::Integer)
        samplesize >= 1 || invalid_option(:quantizer, "samplesize must be at least 1; got $samplesize")
        mm = if minmax === nothing
            nothing
        else
            length(minmax) == 2 && minmax[1] < minmax[2] ||
                invalid_option(:quantizer, "`minmax` must be two numbers (lo, hi) with lo < hi; got $(repr(minmax))")
            (Float32(minmax[1]), Float32(minmax[2]))
        end
        new(mm, Int(samplesize))
    end
end

GlobalRange(; minmax=nothing, samplesize::Integer=4096) = GlobalRange(minmax, samplesize)

"""
    PerVectorRange(; samplesize=4096)

One range for each vector, taken from its own coordinates. It costs 8 more bytes per vector.
The encoder calibrates how it rounds each range on `samplesize` of the staged vectors, at the
first `index!` or by `fit_quantizer!`.
"""
struct PerVectorRange <: QuantRange
    samplesize::Int

    function PerVectorRange(samplesize::Integer)
        samplesize >= 1 || invalid_option(:quantizer, "samplesize must be at least 1; got $samplesize")
        new(Int(samplesize))
    end
end

PerVectorRange(; samplesize::Integer=4096) = PerVectorRange(samplesize)

"""
    SQ(bits=8; range=GlobalRange())

A scalar quantizer: each coordinate becomes a code of `bits` bits, 8, 4 or 2. A vector of
dimension 1024 takes 1024 bytes at 8 bits, 512 at 4 bits and 256 at 2 bits, against 4096 bytes
in `Float32`. The dimension must be a multiple of 2 at 4 bits and of 4 at 2 bits.
"""
struct SQ
    bits::Int
    range::QuantRange

    function SQ(bits::Integer, range::QuantRange)
        bits in (8, 4, 2) || invalid_option(:quantizer, "a scalar quantizer has 8, 4 or 2 bits; got $bits")
        new(Int(bits), range)
    end
end

SQ(bits::Integer=8; range::QuantRange=GlobalRange()) = SQ(bits, range)

"""
    Rotation

The rotation applied to every vector and every query before quantization:
[`NoRotation`](@ref), [`QRRotation`](@ref) or [`HadamardRotation`](@ref).
"""
abstract type Rotation end

"""
    NoRotation()

The coordinates are quantized as they are. This is the default. It works for the output of most
modern neural embedding models, whose coordinates have similar scales. On SISAP 2025 `ccnews`,
a QR rotation changed recall@10 by less than 0.01 at every width.
"""
struct NoRotation <: Rotation end

"""
    QRRotation()

A random orthogonal matrix, which gives every coordinate the same scale. Use it when the
coordinates have different scales or heavy tails: vectors that are not normalized, vectors from
several models or pipelines in one project, features that were not produced by a neural model,
or a recall clearly below the exact search on a sample of queries. It costs `dim²` operations per
query and per inserted item, and `4 dim²` bytes of storage.
"""
struct QRRotation <: Rotation end

"""
    HadamardRotation()

A random rotation that costs `dim log dim` operations per vector. The dimension must be a power
of two.
"""
struct HadamardRotation <: Rotation end

"""
    Quantized(quantizer=SQ(8); rotation=NoRotation(), keep_vectors=false)

Each vector is stored as the codes of `quantizer` instead of `Float32` values (issue #1), and
the project searches with the distances of `SimilaritySearch.ScalarQuant`. The distance of the
project must be `SqL2`, `L2`, `L1`, `Cosine` or `NormCosine`. It works with [`Graph`](@ref) and
[`Exhaustive`](@ref).

`append_items!` stages the `Float32` vectors, and `index!` encodes them. `keep_vectors=true`
keeps the `Float32` vectors on disk next to the codes, so that `fetch_items` returns them. With
`false`, they are removed once they are encoded, and `fetch_items` returns `nothing` as the
payload.
"""
struct Quantized <: DenseStorage
    quantizer::SQ
    rotation::Rotation
    keep_vectors::Bool
end

Quantized(quantizer::SQ=SQ(); rotation::Rotation=NoRotation(), keep_vectors::Bool=false) =
    Quantized(quantizer, rotation, keep_vectors)

"""
    Sketched(hyperplanes=256; bits=8, rerank=0, keep_vectors=false, center=false,
             samplesize=16_384, hsel=16hyperplanes, henc=1024, entquantile=0.5,
             anchorpolicy=:random)

Each vector is stored as a sketch of `hyperplanes` components at `bits` bits each, instead of
its `Float32` values (PLAN.md §5.14). The sketch is built by `SimilaritySearch.Projections`: an
`AnchoredDistantHyperplanes` model selects `hyperplanes` pairs of anchor vectors, and a
`QuantSketch` encodes the signed margin of a vector against each pair, `d(v, b) - d(v, a)`,
divided by `d(a, b)`. The anchors are selected under the distance of the project, so any
distance of SimilaritySearch works.

A sketch takes `hyperplanes * bits / 8` bytes: 1024 hyperplanes at 1 bit, 512 at 2, 256 at 4
and 128 at 8 all take 128 bytes, against 4,096 bytes for a `Float32` vector of dimension 1024.
At 1 bit, sketches are compared with the Hamming distance; at 2, 4 and 8 bits, with the squared
Euclidean distance between their codes.

- `hyperplanes`: a multiple of 64.
- `bits`: 1, 2, 4 or 8.
- `rerank`: `0`, or the number of candidates, as a multiple of `k`, that a second evaluation
  scores with the distance of the project and the `Float32` vectors. Without it, a result has
  the distance between sketches, which orders the results and is not the distance of the
  project. A value above 0 needs `keep_vectors=true`.
- `keep_vectors`: keep the `Float32` vectors on disk. With `false`, they are removed once they
  are encoded, which is the reduction of memory and disk that this storage exists for.
- `center`: subtract the mean of the fitting sample from every vector and every query before
  the distances are computed. The anchors depend only on distances, so centering changes
  nothing under `L2` or `SqL2`, which do not change when every vector moves by the same amount;
  under `Cosine` or `NormCosine` it changes the angles. Vectors of a language model seldom need
  it. A rotation changes none of these distances, so this storage has no rotation.

The first [`index!`](@ref) fits the encoder on a sample of at most `samplesize` staged vectors,
and later ones encode with it. The fitting evaluates `hsel` candidate pairs on `henc` vectors of
the sample, `2 * hsel * henc` distances, and keeps the `hyperplanes` most diverse of the pairs
whose sides have an entropy above the `entquantile` quantile. About `hsel * (1 - entquantile)`
pairs pass that cutoff, and they must be at least `hyperplanes`. `henc` is a multiple of 64, and
it is lowered to the largest multiple of 64 below the size of the sample, so the encoder needs
more than 64 staged vectors. `anchorpolicy` (`:random` or `:extremal`) selects the vector that
orients the pairs.

Measured on 2026-11-23 on SISAP 2025 `ccnews` (603,664 vectors of dimension 384, `NormCosine`,
1,000 queries, 32 threads). With the defaults, a sketch takes 256 bytes, the fitting took 0.1 s,
and recall@10 was 0.49; with `rerank=4` it was 0.73. With 2,048 hyperplanes at 1 bit, also 256
bytes, the fitting took 2.6 to 3.9 s and recall@10 was 0.56. On the same data, `Quantized(SQ(8))`
takes 384 bytes per vector and gave 0.989, and `Quantized(SQ(4))` takes 192 bytes and gave 0.901.
For the vectors of a language model, use [`Quantized`](@ref). A sketched project works with any
`SemiMetric` distance, which a scalar quantizer does not.
"""
struct Sketched <: DenseStorage
    hyperplanes::Int
    bits::Int
    rerank::Int
    keep_vectors::Bool
    center::Bool
    samplesize::Int
    hsel::Int
    henc::Int
    entquantile::Float64
    anchorpolicy::Symbol

    function Sketched(hyperplanes::Integer, bits::Integer, rerank::Integer, keep_vectors::Bool, center::Bool,
                      samplesize::Integer, hsel::Integer, henc::Integer, entquantile::Real, anchorpolicy::Symbol)
        hyperplanes >= 64 && hyperplanes % 64 == 0 ||
            invalid_option(:sketch, "`hyperplanes` must be a positive multiple of 64; got $hyperplanes")
        bits in (1, 2, 4, 8) || invalid_option(:sketch, "a sketch has 1, 2, 4 or 8 bits per hyperplane; got $bits")
        rerank >= 0 || invalid_option(:sketch, "`rerank` must be 0 or a positive number of candidates per result; got $rerank")
        rerank == 0 || keep_vectors || invalid_option(:sketch,
            "`rerank` = $rerank evaluates the candidates with the Float32 vectors, which `keep_vectors=false` removes; " *
            "pass keep_vectors=true, or rerank=0")
        henc >= 64 && henc % 64 == 0 || invalid_option(:sketch, "`henc` must be a positive multiple of 64; got $henc")
        hsel >= hyperplanes || invalid_option(:sketch, "`hsel` ($hsel) must be at least `hyperplanes` ($hyperplanes)")
        samplesize > 64 || invalid_option(:sketch, "`samplesize` must be above 64; got $samplesize")
        0 < entquantile < 1 || invalid_option(:sketch, "`entquantile` must be between 0 and 1; got $entquantile")
        # The fitting keeps the candidate pairs above the `entquantile` quantile of the entropies,
        # about `hsel * (1 - entquantile)` of them, and selects the hyperplanes among those. The
        # product is rounded: 1280 * (1 - 0.9) is 127.99999999999997 in Float64.
        kept = round(hsel * (1 - entquantile); digits=6)
        kept >= hyperplanes || invalid_option(:sketch,
            "`hsel` ($hsel) keeps about $(floor(Int, kept)) candidate pairs above the `entquantile` " *
            "($entquantile) quantile, fewer than `hyperplanes` ($hyperplanes); raise `hsel` to at least " *
            "$(ceil(Int, hyperplanes / (1 - entquantile))) or lower `entquantile`")
        anchorpolicy in (:random, :extremal) ||
            invalid_option(:sketch, "`anchorpolicy` must be :random or :extremal; got $(repr(anchorpolicy))")
        new(Int(hyperplanes), Int(bits), Int(rerank), keep_vectors, center, Int(samplesize), Int(hsel), Int(henc),
            Float64(entquantile), anchorpolicy)
    end
end

Sketched(hyperplanes::Integer=256; bits::Integer=8, rerank::Integer=0, keep_vectors::Bool=false, center::Bool=false,
         samplesize::Integer=16_384, hsel::Integer=16hyperplanes, henc::Integer=1024, entquantile::Real=0.5,
         anchorpolicy::Symbol=:random) =
    Sketched(hyperplanes, bits, rerank, keep_vectors, center, samplesize, hsel, henc, entquantile, anchorpolicy)

# ---------------------------------------------------------------------------------------
# What a project holds

"""
    DenseVectors(distance=SqL2(); index=Graph(), storage=Float32Vectors())

A project of dense vectors. It takes `DenseItem`s.

```julia
DenseVectors()                                              # SqL2, a SearchGraph, Float32
DenseVectors(Dist.NormCosine(); index=Exhaustive(parallel=true))
DenseVectors(Dist.NormCosine(); storage=Quantized(SQ(4); rotation=QRRotation()))
```
"""
struct DenseVectors{I<:DenseIndex,S<:DenseStorage} <: ProjectSpec
    distance::SimilaritySearch.PreMetric
    index::I
    storage::S

    function DenseVectors(distance::SimilaritySearch.PreMetric, index::I, storage::S) where {I<:DenseIndex,S<:DenseStorage}
        storage isa Quantized && quantized_distance(distance)
        storage isa Sketched && (distance isa SimilaritySearch.SemiMetric || invalid_option(:sketch,
            "a sketched project selects its anchors with a SemiMetric distance; $(nameof(typeof(distance))) is not one"))
        new{I,S}(distance, index, storage)
    end
end

DenseVectors(distance::SimilaritySearch.PreMetric=SimilaritySearch.Dist.SqL2();
             index::DenseIndex=Graph(), storage::DenseStorage=Float32Vectors()) =
    DenseVectors(distance, index, storage)

"""
    SparseVectors(dimension, distance=Dist.NormCosine())

A project of sparse vectors that the caller encodes. It takes `SparseItem`s, and indexes them in
a `SimilaritySearch.InvertedFile`. `dimension` is required: an inverted file is a fixed array of
posting lists, so it has to be sized before the first item, and every `SparseItem` appended must
agree with it. `distance` is cosine over the weights, or one of `Dist.Sets.*` over the nonzero
positions.
"""
struct SparseVectors <: ProjectSpec
    dimension::Int
    distance::SimilaritySearch.PreMetric

    function SparseVectors(dimension::Integer, distance::SimilaritySearch.PreMetric)
        dimension >= 1 || invalid_option(:dimension, "dimension must be at least 1; got $dimension")
        new(Int(dimension), distance)
    end
end

SparseVectors(dimension::Integer) = SparseVectors(dimension, SimilaritySearch.Dist.NormCosine())

"""
    TextIndex

The index of a [`TextDocuments`](@ref) project: [`BM25`](@ref) or [`VectorSpace`](@ref).
"""
abstract type TextIndex end

"""
    BM25()

An inverted file scored with BM25 (`BM25InvertedFile`). This is the default index of a text
project.
"""
struct BM25 <: TextIndex end

"""
    VectorSpace(; distance=Dist.NormCosine())

An inverted file of the weighted vectors of the text profile (`TextInvertedFile`), compared with
`distance`.
"""
struct VectorSpace <: TextIndex
    distance::SimilaritySearch.PreMetric
end

VectorSpace(; distance::SimilaritySearch.PreMetric=SimilaritySearch.Dist.NormCosine()) = VectorSpace(distance)

"""
    TextDocuments(model; index=BM25())

A project of text that this package encodes. It takes `TextItem`s. `model` says where the
vocabulary comes from, and it has no default (see [`AbstractTextModelSpec`](@ref)):

- `DefaultProfile(:es)`: the published profile for a language, refitted to the project's corpus
  at the first `index!`. This is the recommended form.
- `BaseProfile(load_profile("path/to/profile.zip"))`: a profile used as it is.
- `FitFromCorpus(TextConfig(language=:es); min_ndocs=3)`: a profile fitted at the first
  `index!` from a sample of the project's own corpus.
"""
struct TextDocuments{I<:TextIndex} <: ProjectSpec
    model::AbstractTextModelSpec
    index::I
end

TextDocuments(model::AbstractTextModelSpec; index::TextIndex=BM25()) = TextDocuments(model, index)

# ---------------------------------------------------------------------------------------
# What each spec means to the rest of the engine

spec_kind(::DenseVectors) = :dense
spec_kind(::SparseVectors) = :sparse
spec_kind(::TextDocuments) = :text

# The library index type of a spec, which `IndexEngine` builds.
index_type(::DenseVectors{Graph}) = SearchGraph
index_type(s::DenseVectors{Exhaustive}) = s.index.parallel ? ParallelExhaustiveSearch : ExhaustiveSearch
index_type(::TextDocuments{BM25}) = BM25InvertedFile
index_type(::TextDocuments{VectorSpace}) = TextInvertedFile

# The tag `snapshot_state` used to save as `:backend`, kept because `restore_engine` and the
# legacy reader of `open_project` still speak it.
backend_symbol(::DenseVectors{Graph}) = :graph
backend_symbol(s::DenseVectors{Exhaustive}) = s.index.parallel ? :parallel_exhaustive : :exhaustive
backend_symbol(::SparseVectors) = :inverted_file
backend_symbol(::TextDocuments{BM25}) = :bm25
backend_symbol(::TextDocuments{VectorSpace}) = :text_inverted_file

# A project whose vectors are stored as codes: `Quantized` or `Sketched`.
isquantized(s::ProjectSpec) = s isa DenseVectors && (s.storage isa Quantized || s.storage isa Sketched)

text_distance(s::TextDocuments{BM25}) = nothing
text_distance(s::TextDocuments{VectorSpace}) = s.index.distance

# ---------------------------------------------------------------------------------------
# The saved form

"""
    spec_to_dict(spec::ProjectSpec) -> Dict{String,Any}

The form of `spec` that a project saves in its field `:spec`. It is a dictionary with the
structure of the JSON form of the HTTP API, so that the saved format does not depend on the names
of the Julia types of this file. A distance is kept as the `SimilaritySearch` value. The text
model of a `TextDocuments` keeps the options of a `DefaultProfile`; a `BaseProfile` and a
`FitFromCorpus` are saved by the engine in the fields `:profile` and `:fitspec`, and the
dictionary only names which one it is.
"""
spec_to_dict(s::DenseVectors) =
    Dict{String,Any}("dense" => Dict{String,Any}("distance" => s.distance, "index" => _dict(s.index),
                                                 "storage" => _dict(s.storage)))
spec_to_dict(s::SparseVectors) =
    Dict{String,Any}("sparse" => Dict{String,Any}("dimension" => s.dimension, "distance" => s.distance))
spec_to_dict(s::TextDocuments) =
    Dict{String,Any}("text" => Dict{String,Any}("model" => _dict(s.model), "index" => _dict(s.index)))

_dict(g::Graph) = Dict{String,Any}("graph" => Dict{String,Any}("minrecall" => g.minrecall))
_dict(e::Exhaustive) = Dict{String,Any}("exhaustive" => Dict{String,Any}("parallel" => e.parallel))
_dict(::Float32Vectors) = Dict{String,Any}("float32" => Dict{String,Any}())
function _dict(q::Quantized)
    r = q.quantizer.range
    Dict{String,Any}("quantized" => Dict{String,Any}(
        "bits" => q.quantizer.bits,
        "range" => r isa GlobalRange ? "global" : "per_vector",
        "minmax" => r isa GlobalRange && r.minmax !== nothing ? Float32[r.minmax...] : nothing,
        "samplesize" => r.samplesize,
        "rotation" => _rotation_name(q.rotation),
        "keep_vectors" => q.keep_vectors))
end
_dict(s::Sketched) = Dict{String,Any}("sketched" => Dict{String,Any}(
    "hyperplanes" => s.hyperplanes, "bits" => s.bits, "rerank" => s.rerank, "keep_vectors" => s.keep_vectors,
    "center" => s.center, "samplesize" => s.samplesize, "hsel" => s.hsel, "henc" => s.henc,
    "entquantile" => s.entquantile, "anchorpolicy" => String(s.anchorpolicy)))
_dict(::BM25) = Dict{String,Any}("bm25" => Dict{String,Any}())
_dict(v::VectorSpace) = Dict{String,Any}("vector_space" => Dict{String,Any}("distance" => v.distance))
_dict(m::DefaultProfile) = Dict{String,Any}("default_profile" => Dict{String,Any}(
    "language" => String(m.language), "nickname" => m.nickname, "refit" => m.refit,
    "max_documents" => m.max_documents))
_dict(::BaseProfile) = Dict{String,Any}("base_profile" => Dict{String,Any}())
_dict(::FitFromCorpus) = Dict{String,Any}("fit_from_corpus" => Dict{String,Any}())

_rotation_name(::NoRotation) = "none"
_rotation_name(::QRRotation) = "qr"
_rotation_name(::HadamardRotation) = "hadamard"

# The one entry of a dictionary of the saved form, as `(name, value)`.
function _only(d::AbstractDict, what::String)
    length(d) == 1 || corrupted_storage("the saved $what has $(length(d)) entries; expected one")
    k, v = first(d)
    return String(k), v
end

"""
    spec_from_dict(d::AbstractDict; textmodel) -> ProjectSpec

The inverse of [`spec_to_dict`](@ref). `textmodel` is the text model of a text project whose
saved model is a `BaseProfile` or a `FitFromCorpus`, rebuilt by the caller from the fields
`:profile` and `:fitspec`; it is not read for the other projects.
"""
function spec_from_dict(d::AbstractDict; textmodel::Union{Nothing,AbstractTextModelSpec})
    kind, body = _only(d, "project spec")
    if kind == "dense"
        return DenseVectors(body["distance"], _dense_index(body["index"]), _dense_storage(body["storage"]))
    elseif kind == "sparse"
        return SparseVectors(body["dimension"], body["distance"])
    elseif kind == "text"
        mkind, mbody = _only(body["model"], "text model")
        model = if mkind == "default_profile"
            DefaultProfile(Symbol(mbody["language"]); nickname=mbody["nickname"], refit=mbody["refit"],
                           max_documents=mbody["max_documents"])
        else
            textmodel === nothing && corrupted_storage("the text model of this project ($mkind) is not saved")
            textmodel
        end
        return TextDocuments(model, _text_index_from_dict(body["index"]))
    end
    corrupted_storage("unknown project kind $(repr(kind)) in the saved spec; expected dense, sparse or text")
end

function _dense_index(d::AbstractDict)
    k, v = _only(d, "dense index")
    k == "graph" && return Graph(v["minrecall"])
    k == "exhaustive" && return Exhaustive(v["parallel"])
    corrupted_storage("unknown dense index $(repr(k)) in the saved spec; expected graph or exhaustive")
end

function _dense_storage(d::AbstractDict)
    k, v = _only(d, "dense storage")
    k == "float32" && return Float32Vectors()
    k == "sketched" && return Sketched(v["hyperplanes"], v["bits"], v["rerank"], v["keep_vectors"], v["center"],
                                       v["samplesize"], v["hsel"], v["henc"], v["entquantile"], Symbol(v["anchorpolicy"]))
    k == "quantized" || corrupted_storage("unknown dense storage $(repr(k)) in the saved spec; expected float32, quantized or sketched")
    range = v["range"] == "global" ? GlobalRange(v["minmax"] === nothing ? nothing : Tuple(v["minmax"]), v["samplesize"]) :
            PerVectorRange(v["samplesize"])
    rot = v["rotation"]
    rotation = rot == "none" ? NoRotation() : rot == "qr" ? QRRotation() : rot == "hadamard" ? HadamardRotation() :
               corrupted_storage("unknown rotation $(repr(rot)) in the saved spec; expected none, qr or hadamard")
    return Quantized(SQ(v["bits"], range), rotation, v["keep_vectors"])
end

function _text_index_from_dict(d::AbstractDict)
    k, v = _only(d, "text index")
    k == "bm25" && return BM25()
    k == "vector_space" && return VectorSpace(v["distance"])
    corrupted_storage("unknown text index $(repr(k)) in the saved spec; expected bm25 or vector_space")
end
