# What the test files of the engine share: the packages they use, the data path, and the
# helpers that turn the rows of the test corpus into items.

using Test
using SimilaritySearchEngine
using SimilaritySearchEngine: Schema, IndexEngine
using SimilaritySearch: SearchGraph, Dist, MatrixDatabase, SearchGraphContext, InformativeLog, BeamSearch, ExhaustiveSearch
using TextSearch: BM25InvertedFile, InvertedFile, TextInvertedFile, NormalizationConfig,
                  AppliedArtifacts, vocsize, gettrainsize, gettextconfig,
                  lineage_summary, token2id, query_tokens, QueryPipeline
using JSON
using RocksDB
using Random: Xoshiro
using SparseArrays: sparsevec, SparseVector, nonzeros, nonzeroinds

const FRANKENSTEIN_PATH = joinpath(@__DIR__, "data", "frankenstein.jsonl")

function mktempworkdir(f)
    dir = mktempdir()
    try
        f(dir)
    finally
        rm(dir; recursive=true, force=true)
    end
end

# The engine takes typed items, so a corpus read off disk has to be converted -- and doing it
# here, in the caller, is the whole point of the change: the test knows that "vector" is the
# payload and "verb_count" is metadata, and the library no longer has to guess from key names.
const PAYLOAD_KEYS = ("vector", "text", "doc_id", "keywords", "ref")
_meta_of(row) = Dict{String,Any}(k => v for (k, v) in row if !(k in PAYLOAD_KEYS))
dense_items(rows) = [DenseItem(r["vector"]; doc_id=r["doc_id"], keywords=r["keywords"],
                               refs=r["ref"], meta=_meta_of(r)) for r in rows]
text_items(rows) = [TextItem(r["text"]; doc_id=r["doc_id"], keywords=r["keywords"],
                             refs=r["ref"], meta=_meta_of(r)) for r in rows]

# A sparse project's items are vectors the *caller* encoded -- the engine has no vocabulary and
# never sees a token. Hashing words into a fixed dimension is the smallest honest stand-in for
# whatever a caller's own encoder does (a learned sparse retriever, a feature table, a set of ids),
# and it is deliberately not TextSearch's encoding: that path is what `FullTextEngine` is for.
const SPARSE_DIM = 4096
function sparse_items(rows)
    map(rows) do r
        counts = Dict{Int32,Float32}()
        for w in split(lowercase(r["text"]), r"[^\p{L}\p{N}]+"; keepempty=false)
            k = Int32(mod(hash(w), SPARSE_DIM) + 1)
            counts[k] = get(counts, k, 0f0) + 1f0
        end
        ind = sort!(collect(keys(counts)))
        val = Float32[counts[i] for i in ind]
        val ./= sqrt(sum(abs2, val))          # NormCosine reads its inputs as already normalized
        SparseItem(sparsevec(ind, val, SPARSE_DIM); doc_id=r["doc_id"], keywords=r["keywords"],
                   refs=r["ref"], meta=_meta_of(r))
    end
end

# A text policy that keeps case and diacritics, which is what makes orthographic bridging
# possible at all: `derive_variants` has nothing to do under the default `TextConfig()`, whose
# normalization already folds both.
const CASED_ES = TextConfig(normalization=NormalizationConfig(lc=false, del_diac=false), language=:es)

# 60 documents spelling it "musica" with an accent against a single one without: far enough
# apart for `QueryPolicy`'s default `negligible_ratio=50` to read the bare spelling as wrong.
const ACCENT_ITEMS = vcat(
    [TextItem("la m\u00fasica cl\u00e1sica de la ciudad $i"; doc_id="acc_$i") for i in 1:60],
    [TextItem("una musica rara sin acento"; doc_id="bare")])
