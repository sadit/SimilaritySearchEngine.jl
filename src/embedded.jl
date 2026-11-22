# Friendly, function-per-operation embedded API (PLAN.md §8.5, chunk 22): a script running
# on the same machine against the same --workdir a similarity-search/similarity-search-serve
# process already uses, with no HTTP, no CLI subprocess, no running server required. Every
# function here mirrors an existing cli_handlers.jl/server.jl code path exactly (same
# on-disk layout, same insertion/search/persistence semantics) so a project touched through
# this API stays fully interoperable with the CLI and HTTP server -- this is a second way to
# reach the same shared engine, not a separate one.
#
# Scope is deliberately bounded to project lifecycle + core CRUD + calibrate + one
# representative heavy op (allknn). fft/neardup/hsp/rebuild!/dump_dataset/load_dataset are
# not built here -- a documented follow-up (PLAN.md §8.5), not an oversight.

"""
    SearchResult

One hit: the internal `_id` the engine ranked, the caller's own `doc_id` for it (`nothing`
when the item was appended without one, and when the hit is soft-deleted), the `distance`, and
whether the item is soft-deleted.

The field names match [`Schema.MetadataRecord`](@ref) on purpose. The named tuple this
replaced called the caller's external id `id` and the internal one `doc_id` -- exactly
backwards from the record, so `result.doc_id` and `record.doc_id` were different things and one
of them was always the wrong guess.

A soft-deleted hit is reported, not hidden (see [`IndexEngine.search_live`](@ref)), and carries
no `doc_id`: its metadata is not accessible through search.
"""
struct SearchResult
    _id::Int32
    doc_id::Union{String,Nothing}
    distance::Float32
    deleted::Bool
end

"""
    SearchStats

A mutable, caller-supplied box for the cost accounting of one search. Pass it as
`search(h, q, k; stats=SearchStats())` (or to [`ftsearch`](@ref)) and read
`stats.distance_evaluations` afterwards: the number of distance computations *that call*
performed, which is what a frontend logs per request and what a recall/cost experiment
measures.

It exists because the count is produced inside [`IndexEngine.search_live`](@ref), on a
context borrowed from the engine's pool for the duration of the call, and is therefore
not recoverable from outside afterwards -- the pool hands the same context to the next
concurrent search, and `ctx.costdists` is a running total, never reset. A caller that
snapshots a context around a search attributes another thread's work to its own request.

`stats.candidates` is the number of results the index returned for that call, before a
`filter` was applied. With a filter, a value equal to the `candidates` that was asked for
means the budget may have stopped the search early: asking again with a larger budget can
return more results. A smaller value means the index had no more candidates, and a larger
budget returns the same results.

Reused across calls if you want a running total of your own; each search *overwrites*
the fields rather than accumulating, so add them up yourself if that is what you want.
"""
mutable struct SearchStats
    distance_evaluations::Int
    candidates::Int
end

SearchStats() = SearchStats(0, 0)

"""
    ExistsResult

What [`exists`](@ref) reports per queried id: the id as asked for, whether a record was found,
and whether it is soft-deleted.
"""
struct ExistsResult
    id::String
    exists::Bool
    deleted::Bool
end

"""
    KnnRow

One row of [`allknn`](@ref): item `_id` and its `k` nearest neighbours, ascending by distance.
`_id` and `neighbors` are internal ids -- 1-based positions in insertion order -- not `doc_id`s.
"""
struct KnnRow
    _id::Int32
    neighbors::Vector{Int32}
    dists::Vector{Float32}
end

"""
    CenterSelectionResult

What [`fft`](@ref) and [`dnet`](@ref) return: the chosen `centers`, which center each item was
assigned to (`assign`) and its distance to it (`assigndist`), the two radii describing the
selection (`covering` and `separation`), and the two cost counters.

A faithful, typed restatement of `SimilaritySearch.CenterSelection`, with ids as `Int32` to
match every other id this package hands back.

- `centers` are internal `_id`s (1-based position in the index), not hydrated with metadata.
- `assign[i]` is a **position in `centers`**, from `1` to `length(centers)` -- not an `_id`.
  The `_id` of item `i`'s center is `centers[assign[i]]`.
- `covering` is the largest `assigndist`: the radius the centers need to cover everything.
- `separation` is the smallest distance between two centers. These two are different numbers
  and used to be conflated under one name; see `SimilaritySearch.CenterSelection`.
"""
struct CenterSelectionResult
    centers::Vector{Int32}
    assign::Vector{Int32}
    assigndist::Vector{Float32}
    covering::Float32
    separation::Float32
    costdists::Int
    costblocks::Int
end

const FFTResult = CenterSelectionResult

"""
    NearDupResult

What [`neardup`](@ref) returns: the selected `centers` (the ϵ-net of surviving non-duplicate
objects), which center covers each object (`assign`) and its distance to it (`assigndist`),
the threshold radius `epsilon` and the maximum covering radius `covering`, plus the cost counters.

A faithful, typed restatement of `SimilaritySearch.NearDupSelection`, with ids as `Int32` to
match every other id this package hands back.
"""
struct NearDupResult
    centers::Vector{Int32}
    assign::Vector{Int32}
    assigndist::Vector{Float32}
    covering::Float32
    epsilon::Float32
    costdists::Int
    costblocks::Int
end

"""
    OOVCounters

How many of the tokens this project has been asked for were not in its vocabulary.

A text project fits its vocabulary once, from the documents staged when `index!` first ran.
Everything appended afterwards is encoded against that vocabulary, and a word it does not
contain cannot be searched for: it is dropped, silently, at both ends. These counters make
that visible, so that the decision to run a `rebuild` -- which refits the vocabulary over
what the project holds now -- rests on a measurement.

`queries` counts the queries that were resolved, `tokens` the tokens they contained, and
`oov` the ones the vocabulary did not have as typed. Correction may still find a variant for
some of those (`ftexplain` reports which), so this is an upper bound on what a query lost.

The counts are kept in memory and written to the project when it is closed, or every
`OOV_FLUSH_EVERY` queries. A process that is killed loses the queries since the last write;
this is a signal about vocabulary drift, not accounting. `rebuild` starts them again, since
after it the vocabulary is a different one and the old counts describe something that no
longer exists.
"""
mutable struct OOVCounters
    queries::Int
    tokens::Int
    oov::Int
    unflushed::Int
end

OOVCounters() = OOVCounters(0, 0, 0, 0)

"How many queries pass before the counters are written to the project."
const OOV_FLUSH_EVERY = 256

"""
    DEFAULT_OOV_SCAN_SAMPLE

How many documents [`vocabulary_report`](@ref) reads by default when asked to measure the
out-of-vocabulary rate over the project's own documents.

Reading every document costs time linear in the project: 11.6 s over 12.3 million tokens,
measured 2026-09-14. That is an answer to a question an operator asks while waiting for it,
so it is bounded instead: a rate measured over twenty thousand documents, spread over the
whole project, decides whether to rebuild with room to spare, and costs about a second
whatever the project's size. `sample=0` reads everything, for a caller that is offline and
wants the exact figure.
"""
const DEFAULT_OOV_SCAN_SAMPLE = 20_000

"""
    EmbeddedEngine

Bundles everything a script needs to keep working against one open project: its directory
layout plus the live `Project.ProjectManager`/`IndexEngine.AbstractSearchEngine` pair
`cli_handlers.jl`'s `_load_dataset_and_engine` already threads through separately for a
single CLI command, plus the `Persistence.EngineStore` (a RocksDB column family shared
with `project.db`, see `Persistence.open_engine_store`) that engine mutations persist
into field-by-field -- worth bundling here since an embedded-API caller makes many calls
against the same open project instead of running once and exiting.
"""
mutable struct EmbeddedEngine
    workdir::String
    dataset::String
    dir::String
    project::Project.ProjectManager
    engine::IndexEngine.AbstractSearchEngine
    store::Persistence.EngineStore
    schema_version::Int
    # What this project declares about `meta` (PLAN.md §4.5). Empty for a project that
    # declares nothing, which is every project written before this existed and every one
    # created without the keyword.
    meta_schema::Schema.MetaSchema
    # How much of what this project is asked for, it has words for (PLAN.md §5). See
    # [`OOVCounters`](@ref) and [`vocabulary_report`](@ref).
    oov::OOVCounters
    dense_vectors::Base.RefValue{Union{Nothing,MMapMatrixDatabase}}
    read_only::Bool
    wrote::Base.RefValue{Bool}
    # The project's partitions (PLAN.md §5.9), or `nothing` when it declares no partition
    # field. The same reference is held by the text index's `Persistence.LazyPostings`, which
    # writes the partition lists of every block it persists.
    partitions::Base.RefValue{Union{Nothing,Persistence.PartitionStore}}
    # `postings_cache_max` and `postings_cache_base`, kept for a partition store opened after
    # the project (`declare_partition!`).
    postings_cache::Tuple{Int,Int}
    # What the project holds and how it is indexed (issue #2), as saved in `:spec`.
    spec::IndexEngine.ProjectSpec
    # Set by `close_project!`. Every other function that takes a handle checks it first and
    # raises `ProjectClosed` (issue #4): the stores behind a closed handle are closed, and an
    # operation on them returned wrong results instead of an error.
    closed::Base.RefValue{Bool}
    # The followers opened on this handle, by name (PLAN.md §5.11). `index!` and
    # `append_items!` process them according to their policies, and `close_project!` closes
    # them. The values are `Follower`s, a type defined after this one.
    followers::Dict{String,Any}
end

"""
    _check_open(handle::EmbeddedEngine)

Raises [`ProjectClosed`](@ref) when `close_project!` already closed `handle`.
"""
_check_open(handle::EmbeddedEngine) = handle.closed[] && project_closed(handle.dataset,
    "project '", handle.dataset, "' was closed by close_project!; open it again with open_project")

"""
    _searchgraph_on_change(store::Persistence.EngineStore, adj_store::Persistence.AdjacencyStore) -> Function

The `on_change` callback for a `DenseEngine{GraphBackend}`: fires from *inside*
`IndexEngine.index!(engine::IndexEngine.DenseEngine{GraphBackend})` (never from `add_item!`/
`append_items!` anymore -- see that function's docstring for why staging and indexing are
now two separate steps), once per `sp:ep` range `SimilaritySearch.index!`'s own internal
batching processes. Saves each object `i` in that range's own direct-links-only neighbor
list under its own key in `adj_store` (`Persistence.save_neighbors!`,
`IndexEngine.direct_neighbors`) -- adjacency lives in its own RocksDB column family, keyed
by object id (see `Persistence.AdjacencyStore`) -- never the whole (growing) graph as one
value -- and persists `ep` as the new `:graph_len` engine field (`Persistence`'s small
per-field store), so a restart mid-catch-up resumes exactly where this range left off
rather than redoing (or worse, silently forgetting) it.

Raw vectors are *not* handled here -- unlike before, they're persisted directly by
[`append_items!`](@ref) at stage time (see its own docstring), since they're durable and
meaningful the moment they're staged, well before any graph-linking work happens; only
adjacency and `:graph_len` genuinely need to wait for `SimilaritySearch.index!` itself to
compute them, which is exactly what makes it safe to do this synchronously, right inside
the callback: this is exactly the direct-links-only slice
`SimilaritySearch.CallbackLog` hands over at that point, and reconstruction
(`IndexEngine.build_searchgraph`, used by [`open_project`](@ref)) reconnects every reverse
link itself, once, up to the restored `:graph_len`, after replaying every staged vector
and looking up every graph-indexed object's saved adjacency.
"""
function _searchgraph_on_change(store::Persistence.EngineStore, adj_store::Persistence.AdjacencyStore)
    return (index, sp, ep) -> begin
        for i in sp:ep
            Persistence.save_neighbors!(adj_store, i, IndexEngine.direct_neighbors(index, i))
        end
        Persistence.save_field!(store, :graph_len, ep)
    end
end

"""
    _invertedfile_on_change(obj_store::Persistence.InvertedFileObjectStore) -> Function

The `on_change` callback for a `FullTextEngine`: on every
`push_item!`/`append_items!` report for range `sp:ep`, saves *only* that range's raw
indexed objects (bags-of-words or `SparseVector`s -- see `IndexEngine.invertedfile_objects`)
as a new, never-again-rewritten block in `obj_store` (`Persistence.append_objects!`) --
never the whole (growing) index as one value. Safe to do synchronously, right inside the
callback: an inverted file's `LOG` only fires after a call's
mutation is fully done, so there's no partial-state hazard here the way there is for a
`SearchGraph` (see `SimilaritySearch.CallbackLog`'s docstring). Reconstruction
(`IndexEngine.build_sparseinvertedfile`, used by [`open_project`](@ref)) rebuilds the whole
index by replaying every saved object back through the library's own insertion -- see
`Persistence.InvertedFileObjectStore`'s docstring for the scaling trade-off that implies.

Only a sparse project is wired to this now: a text project persists its index
(`Persistence.InvertedIndexStore`) and has no observer of its own.
"""
_invertedfile_on_change(obj_store::Persistence.InvertedFileObjectStore) =
    (index, sp, ep) -> Persistence.append_objects!(obj_store, sp, IndexEngine.invertedfile_objects(index, sp, ep))

"""
    _invfile_store(project) -> Persistence.InvertedIndexStore

The project's persisted inverted index (posting lists and per-document vectors). See
`Persistence.INVFILE_POSTINGS_CF`.
"""
_invfile_store(project::Project.ProjectManager) = Persistence.open_inverted_index_store(project.db)

"""
    _invfile_adj_factory(store, partitions; maxlists, baselists) -> Function

The `vocsize -> AbstractAdjList` an inverted file is built around when its posting lists live
in `store` instead of in memory. `IndexEngine` calls it when it constructs the index and never
looks inside the result -- see `IndexEngine._text_index`.
"""
_invfile_adj_factory(store::Persistence.InvertedIndexStore,
                     partitions::Base.RefValue{Union{Nothing,Persistence.PartitionStore}};
                     maxlists::Int, baselists::Int) =
    vocsize -> Persistence.LazyPostings(store, vocsize, maxlists, baselists, partitions)

"""
    _assemble_text_index(profile, kind, distance, store, adj_factory, edit_correction) -> AbstractInvertedFile

Rebuilds a queryable text index from what was persisted, without reindexing anything.

The cheap, small half is reconstructed from the profile -- vocabulary, scorer or weighting
model, and the query pipeline all come from the library's own profile-taking constructor, which
is also what derives the spelling-variant map once instead of per query. The large half comes
from storage: document vectors read into memory, posting lists left on disk behind
`adj_factory`'s adjacency list, and the per-document bookkeeping each backend keeps
(`doclens` for BM25, `sizes` for the weighted one) recomputed from those vectors rather than
stored a second time where the two could drift.

Measured on 265k Gutenberg paragraphs, 2026-09-11: 1.1s against 20.4s to recompute a BM25
index from the raw objects, and that is *with* the document vectors loaded eagerly.
"""
function _assemble_text_index(profile::TextProfile, kind::Type, distance,
                              store::Persistence.InvertedIndexStore, adj_factory::Function,
                              edit_correction::Bool)
    docvecs = Persistence.load_invfile_docvecs(store)
    if kind === BM25InvertedFile
        template = BM25InvertedFile(profile)
        doclens = Int32[Int32(sum(v.nzval; init=UInt32(0))) for v in docvecs]
        return BM25InvertedFile(template.voc, template.bm25, adj_factory(length(template.adj)),
                                doclens, VectorDatabase(docvecs), Ref(Int64(length(docvecs))),
                                IndexEngine._query_pipeline(template.query, template.voc, edit_correction))
    end
    template = TextInvertedFile(profile; dist=distance)
    inner = template.invfile
    sizes = UInt32[UInt32(Persistence.docvec_nnz(v)) for v in docvecs]
    rebuilt = InvertedFile(inner.dist, adj_factory(length(inner.adj)), sizes,
                                            VectorDatabase(docvecs), Ref(Int64(length(docvecs))))
    TextInvertedFile(template.model, rebuilt,
                     IndexEngine._query_pipeline(template.query, template.model.voc, edit_correction))
end

# Every column family this package adds to a project, opened on every open (see
# `Project.open_project`). The partition ones exist in every project, partitioned or not.
const _PROJECT_CFS = [Persistence.ENGINE_CF, Persistence.ADJACENCY_CF, Persistence.INVFILE_DB_CF,
                      Persistence.STAGED_TEXT_CF, Persistence.INVFILE_POSTINGS_CF, Persistence.INVFILE_DOCVECS_CF,
                      Persistence.PARTITION_TABLE_CF, Persistence.PARTITION_MEMBERS_CF,
                      Persistence.PARTITION_POSTINGS_CF, Persistence.PARTITION_GRAPH_CF]

"""
    PartitionBy(field; graph_min=nothing)

Partitions a text or a dense project by the `meta` field `field` (PLAN.md §5.9, §5.10). Given to
`create_project(...; partitions=PartitionBy("repo"))` or to [`declare_partition!`](@ref).

The value of the field is a non-empty string or a list of them; a list puts the item in several
partitions. An item without the field belongs to no partition. If the field is declared in
`meta_schema`, it must be declared `:string`, which also restricts it to one value per item. A
project has at most one partition field, and it cannot be changed. A sparse project cannot be
partitioned yet.

- On a text project, the engine keeps one set of posting lists per value of the field, next to
  the global ones. `ftsearch(h, q, k; partition="unicach")` searches only the items whose
  field is `"unicach"`, and returns the ids, order and scores of the global search restricted to
  those items.
- On a dense project, `search(h, v, k; partition="unicach")` searches only the items of that
  partition. A partition shares the vectors of the project and copies none. A partition with
  fewer than `graph_min` indexed members is searched with an exact scan of its members. A larger
  one, on a [`Graph`](@ref) project, has its own graph, which `index!` builds and extends. On an
  exact project every partition is scanned. `graph_min` is
  [`DEFAULT_PARTITION_GRAPH_MIN`](@ref) when it is `nothing`, and it is saved with the project.

`graph_min` is an error on a text project.
"""
struct PartitionBy
    field::String
    graph_min::Union{Nothing,Int}

    function PartitionBy(field::AbstractString, graph_min::Union{Nothing,Integer})
        isempty(field) && invalid_option(:partitions, "the partition field must be a non-empty name")
        graph_min === nothing || graph_min >= 1 ||
            invalid_option(:partitions, "graph_min must be at least 1; got $graph_min")
        new(String(field), graph_min === nothing ? nothing : Int(graph_min))
    end
end

PartitionBy(field::AbstractString; graph_min::Union{Nothing,Integer}=nothing) = PartitionBy(field, graph_min)

"""
    Runtime(; maxbatches=nothing, postings_cache_max=4096, postings_cache_base=2048, edit_correction=false)

The options of an open project that are not saved with it. [`create_project`](@ref) and
[`open_project`](@ref) take them, and each open decides them again.

- `maxbatches` caps the number of batches of a parallel insertion and of a batch of searches.
  `nothing` lets the engine choose (see `IndexEngine.SEARCH_CONTEXT_MAXBATCHES`). A server that
  runs several projects in one process sets it to share the threads.
- `postings_cache_max` and `postings_cache_base` bound the number of posting lists of a text
  project that stay in memory (see `Persistence.LazyPostings`).
- `edit_correction=true` lets the queries of a text project correct a token that is absent from
  the vocabulary to the only vocabulary token at Damerau-Levenshtein distance 1 from it, for
  example `guerar` -> `guerra`. [`ftexplain`](@ref) reports such a correction with the reason
  `:edit`. The correction runs only when `policy.correction` is not `:off`, only for a token the
  vocabulary does not hold, and only when the other corrections found nothing for it. It is
  `false` by default because of its cost, which `IndexEngine._query_pipeline` gives with
  measurements. It is an error on a dense or sparse project.
"""
struct Runtime
    maxbatches::Union{Nothing,Int}
    postings_cache_max::Int
    postings_cache_base::Int
    edit_correction::Bool

    function Runtime(maxbatches::Union{Nothing,Integer}, postings_cache_max::Integer,
                     postings_cache_base::Integer, edit_correction::Bool)
        maxbatches === nothing || maxbatches >= 1 ||
            invalid_option(:maxbatches, "maxbatches must be at least 1, or nothing; got $maxbatches")
        new(maxbatches === nothing ? nothing : Int(maxbatches), Int(postings_cache_max),
            Int(postings_cache_base), edit_correction)
    end
end

Runtime(; maxbatches::Union{Nothing,Integer}=nothing, postings_cache_max::Integer=4096,
        postings_cache_base::Integer=2048, edit_correction::Bool=false) =
    Runtime(maxbatches, postings_cache_max, postings_cache_base, edit_correction)

"""
    create_project(workdir, dataset, spec=DenseVectors(); meta_schema=nothing, partitions=nothing,
                   runtime=Runtime(), schema_version=1) -> EmbeddedEngine

Creates a new project at `<workdir>/<dataset>`. The directory has the layout that
`similarity-search build` produces, so the CLI can inspect, rebuild or dump a project created
this way, and [`open_project`](@ref) can reopen it.

`spec` says what the project holds, the index that holds it and, for dense vectors, how they are
stored (see [`ProjectSpec`](@ref)):

```julia
create_project(w, "emb")                                              # SqL2, a SearchGraph, Float32
create_project(w, "emb", DenseVectors(Dist.NormCosine(); index=Exhaustive()))
create_project(w, "emb", DenseVectors(Dist.NormCosine(); index=Graph(minrecall=0.95),
                                      storage=Quantized(SQ(8); rotation=QRRotation())))
create_project(w, "sp",  SparseVectors(50_000))
create_project(w, "txt", TextDocuments(DefaultProfile(:es); index=BM25()))
```

The spec is saved in the field `:spec`, and [`project_spec`](@ref) returns it. It fixes the item
type [`append_items!`](@ref) accepts: `DenseItem`, `SparseItem` or `TextItem`.

The keywords apply to every kind of project:

- `meta_schema` declares the fields of `meta` (see [`declare_meta_schema!`](@ref)).
- `partitions=PartitionBy("repo")` partitions the project by a field of `meta` (see
  [`PartitionBy`](@ref) and [`partitions`](@ref)).
- `runtime` holds the options that are not saved (see [`Runtime`](@ref)).
- `schema_version` is written in every `Schema.MetadataRecord` that [`append_items!`](@ref)
  writes. It is not saved, and it is not checked against `meta`: a caller that reopens the
  project passes it again.

The index is saved as it grows. A `SearchGraph` saves the links of each block that
[`index!`](@ref index!(::EmbeddedEngine)) builds (see [`_searchgraph_on_change`](@ref)). An
inverted file saves the objects of each insertion (see [`_invertedfile_on_change`](@ref)). An
exact index of `Float32` vectors saves nothing besides its vectors, which
[`append_items!`](@ref) writes to the dense-vector file of the project (issue #3).
"""
function create_project(workdir::String, dataset::String, spec::ProjectSpec=DenseVectors();
                        meta_schema=nothing, partitions::Union{Nothing,PartitionBy}=nothing,
                        runtime::Runtime=Runtime(), schema_version::Int=1)
    # Everything checkable is checked before anything exists on disk: a call that fails after the
    # project directory is made and RocksDB's write lock is taken leaves both behind.
    kind = IndexEngine.spec_kind(spec)
    _check_edit_correction(kind, runtime)
    schema = _as_meta_schema(meta_schema)
    partition_by = partitions === nothing ? nothing : partitions.field
    if partitions !== nothing
        kind === :sparse &&
            invalid_option(:partitions, "a text or a dense project can be partitioned; a sparse project cannot " *
                  "be partitioned yet (PLAN.md §5.10)")
        _check_partition_field(schema, partition_by)
    end
    graph_min = _check_partition_graph_min(kind, partitions)

    dir = joinpath(workdir, dataset)
    mkpath(dir)
    project = Project.open_project(dir, dataset; extra_cf_names=_PROJECT_CFS)
    store = Persistence.open_engine_store(project.db)
    dense_vectors = Ref{Union{Nothing,MMapMatrixDatabase}}(nothing)
    partitions_ref = Ref{Union{Nothing,Persistence.PartitionStore}}(nothing)
    if partition_by !== nothing
        partitions_ref[] = Persistence.open_partition_store(project.db, partition_by,
                                                            runtime.postings_cache_max, runtime.postings_cache_base)
    end
    adj_factory = kind === :text ?
        _invfile_adj_factory(_invfile_store(project), partitions_ref; maxlists=runtime.postings_cache_max,
                             baselists=runtime.postings_cache_base) :
        nothing
    eng = IndexEngine.create_engine(spec; on_change=_on_change(spec, store, project), log_io=nothing,
                                    maxbatches=runtime.maxbatches, adj_factory, edit_correction=runtime.edit_correction)
    Persistence.save_field!(store, :spec, IndexEngine.spec_to_dict(spec))
    # The numbering of the items, which a follower checks (PLAN.md §5.11).
    Persistence.save_field!(store, :generation, Project.generate_id())
    Persistence.save_fields!(store, IndexEngine.snapshot_state(eng))
    isempty(schema) || Persistence.save_field!(store, :meta_schema, schema)
    partition_by === nothing || Persistence.save_field!(store, :partition_by, partition_by)
    handle = EmbeddedEngine(workdir, dataset, dir, project, eng, store, schema_version,
                            schema, OOVCounters(), dense_vectors, false, Ref(true), partitions_ref,
                            (runtime.postings_cache_max, runtime.postings_cache_base), spec, Ref(false),
                            Dict{String,Any}())
    if kind === :dense && partition_by !== nothing
        Persistence.save_field!(store, :partition_graph_min, graph_min)
        _enable_dense_partitions!(handle, graph_min, Dict{UInt32,Int}())
    end
    return handle
end

# What a project saves as its index grows, by what its index reports per insertion: a graph
# reports link blocks, and an inverted file of sparse vectors reports encoded objects. An exact
# index has no structure besides its vectors, which `append_items!` writes to the dense-vector
# file (issue #3). A quantized exact index keeps its codes in their own file (see
# `_index_quantized!`), and a text index is written a block at a time by `IndexEngine.index!`
# itself (see `Persistence.InvertedIndexStore`).
_on_change(::DenseVectors{Graph}, store, project) =
    _searchgraph_on_change(store, Persistence.open_adjacency_store(project.db))
_on_change(::DenseVectors{Exhaustive}, store, project) = nothing
_on_change(::SparseVectors, store, project) =
    _invertedfile_on_change(Persistence.open_invertedfile_object_store(project.db))
_on_change(::TextDocuments, store, project) = nothing

function _check_edit_correction(kind::Symbol, runtime::Runtime)
    kind === :text || !runtime.edit_correction ||
        invalid_option(:edit_correction, "`edit_correction` only applies to a text project; this one holds " *
              "$kind vectors, and its queries have no tokens to correct")
    return nothing
end

"""
    project_spec(handle::EmbeddedEngine) -> ProjectSpec

The specification the project was created with (see [`create_project`](@ref)).
"""
project_spec(handle::EmbeddedEngine) = (_check_open(handle); handle.spec)

"""
    _load_spec(store, read_only) -> Union{Nothing,ProjectSpec}

The spec a project saved in `:spec`, or `nothing` for a project that was never saved. A project
saved before issue #2 has no `:spec`, and its spec is built from the fields it saved then
(`:kind`, `:backend`, `:distance`, `:minrecall`, `:dimension`, or the exact index itself). A
writable open saves the result in `:spec`.
"""
function _load_spec(store, read_only::Bool)
    saved = Persistence.load_field(store, :spec, nothing)
    saved === nothing || return IndexEngine.spec_from_dict(saved; textmodel=_saved_textmodel(store))
    spec = _legacy_spec(store)
    spec === nothing || read_only || Persistence.save_field!(store, :spec, IndexEngine.spec_to_dict(spec))
    return spec
end

# The text model of a text project whose spec names a `BaseProfile` or a `FitFromCorpus`: the
# deferred fit when there is one, and the profile otherwise.
function _saved_textmodel(store)
    fitspec = Persistence.load_field(store, :fitspec, nothing)
    fitspec === nothing || return fitspec
    profile = Persistence.load_field(store, :profile, nothing)
    return profile === nothing ? nothing : BaseProfile(profile)
end

function _legacy_spec(store)
    kind = Persistence.load_field(store, :kind, nothing)
    kind === nothing && return nothing
    backend = Persistence.load_field(store, :backend, nothing)
    if kind === :dense
        Persistence.has_field(store, :quantizer) &&
            corrupted_storage("this quantized project was saved before project specs existed (issue #2); create it again")
        backend === :graph &&
            return DenseVectors(Persistence.load_field(store, :distance, nothing),
                                Graph(Persistence.load_field(store, :minrecall, nothing)), Float32Vectors())
        backend in (:exhaustive, :parallel_exhaustive) ||
            corrupted_storage("unknown dense backend $(repr(backend)); expected :graph, :exhaustive or :parallel_exhaustive")
        index = Persistence.load_field(store, :index, nothing)
        index === nothing && corrupted_storage("this exact dense project has no saved index")
        return DenseVectors(SimilaritySearch.distance(index), Exhaustive(backend === :parallel_exhaustive), Float32Vectors())
    elseif kind === :sparse
        return SparseVectors(Persistence.load_field(store, :dimension, nothing), Persistence.load_field(store, :distance, nothing))
    elseif kind === :text
        model = _saved_textmodel(store)
        model === nothing && corrupted_storage("this text project has neither a profile nor a deferred fit")
        index = backend === :bm25 ? BM25() :
                backend === :text_inverted_file ? VectorSpace(Persistence.load_field(store, :distance, nothing)) :
                corrupted_storage("unknown text backend $(repr(backend)); expected :bm25 or :text_inverted_file")
        return TextDocuments(model, index)
    end
    corrupted_storage("""
        this project records kind $(repr(kind)), which this version does not know how to \
        restore -- it reads :dense, :sparse and :text. A project written before the engine \
        kinds were restructured stored a Julia type there instead of a symbol, and has to be \
        rebuilt rather than reopened.""")
end

"""
    DEFAULT_PARTITION_GRAPH_MIN

The number of indexed members from which a partition of a dense `SearchGraph` project has its
own graph (PLAN.md §5.10). A smaller partition is searched with an exact scan of its members.

Measured on 2026-11-18 over 200,000 synthetic vectors of 128 dimensions, 16 threads: an exact
scan took 0.12 ms for 2,000 members and 3.7 ms for 20,000, and a graph search 0.016 ms and
0.039 ms. The value is to be set again with a measurement on real embeddings.
"""
const DEFAULT_PARTITION_GRAPH_MIN = 4096

# The threshold of a dense partitioned project, or `nothing` for any other. Raises
# `InvalidOption` when `graph_min` is given to a project that is not dense.
function _check_partition_graph_min(kind::Symbol, partitions::Union{Nothing,PartitionBy})
    partitions === nothing && return nothing
    if partitions.graph_min !== nothing && kind !== :dense
        invalid_option(:partitions, "`graph_min` only applies to a dense project; this one indexes $kind " *
              "items, whose partitions are posting lists")
    end
    kind === :dense || return nothing
    return something(partitions.graph_min, DEFAULT_PARTITION_GRAPH_MIN)
end

"""
    _enable_dense_partitions!(handle, graph_min, graph_len) -> IndexEngine.DensePartitions

Gives the dense engine of `handle` its partitions, with the observer that saves each range of a
partition graph: the direct neighbors of the range in `PARTITION_GRAPH_CF`, then the number of
members the graph holds, in the engine field `:partition_graph_len`. `graph_len` is that field
as last saved.
"""
function _enable_dense_partitions!(handle::EmbeddedEngine, graph_min::Integer, graph_len::Dict{UInt32,Int})
    parts = handle.partitions[]
    store = handle.store
    on_change = number -> (index, sp, ep) -> begin
        Persistence.save_partition_neighbors!(parts, number, sp, [IndexEngine.direct_neighbors(index, i) for i in sp:ep])
        graph_len[number] = ep
        Persistence.save_field!(store, :partition_graph_len, graph_len)
    end
    return IndexEngine.enable_partitions!(handle.engine, graph_min, on_change)
end

"""
    _restore_dense_partitions!(handle) -> Nothing

Rebuilds the partitions of a dense project when it is opened (PLAN.md §5.10). The members come
from `PARTITION_MEMBERS_CF`, limited to the indexed items. Each partition graph is rebuilt from
`PARTITION_GRAPH_CF` up to its saved length, with its saved `BeamSearch`, hints and calibrated
levels.
"""
function _restore_dense_partitions!(handle::EmbeddedEngine)
    parts = handle.partitions[]
    store = handle.store
    engine = handle.engine
    graph_len = Persistence.load_field(store, :partition_graph_len, Dict{UInt32,Int}())
    search_state = Persistence.load_field(store, :partition_search_state, Dict{UInt32,NamedTuple}())
    opt = Persistence.load_field(store, :partition_opt_beamsearch, Dict{UInt32,IndexEngine.OptBeamSearch}())
    _enable_dense_partitions!(handle, Persistence.load_field(store, :partition_graph_min, DEFAULT_PARTITION_GRAPH_MIN), graph_len)
    membership = Persistence.load_partition_membership(parts, length(engine.backend.index))
    for number in sort!(collect(keys(membership)))
        st = get(search_state, number, nothing)
        IndexEngine.restore_partition!(engine, number, membership[number];
            load_neighbors=(i -> Persistence.load_partition_neighbors(parts, number, i)),
            graph_len=get(graph_len, number, 0),
            beamsearch=st === nothing ? nothing : st.beamsearch,
            hints=st === nothing ? nothing : st.hints,
            opt_beamsearch=get(opt, number, nothing))
    end
    return nothing
end

"""
    _save_partition_state!(handle) -> Nothing

Saves the `BeamSearch` and hints of each partition graph (`:partition_search_state`) and the
calibrated levels of each partition (`:partition_opt_beamsearch`), as
[`_save_search_state!`](@ref) does for the global graph. Does nothing for a project without
dense partitions.
"""
function _save_partition_state!(handle::EmbeddedEngine)
    engine = handle.engine
    (engine isa IndexEngine.DenseEngine && engine.partitions !== nothing) || return nothing
    Persistence.save_fields!(handle.store, (partition_search_state=IndexEngine.partition_search_state(engine),
                                            partition_opt_beamsearch=IndexEngine.partition_opt_beamsearch(engine)))
    return nothing
end

"""
    open_project(workdir, dataset; read_only=false, runtime=Runtime(), schema_version=1) -> EmbeddedEngine

Reopens a project created by [`create_project`](@ref). The engine is rebuilt from the
[`ProjectSpec`](@ref) saved in `:spec` and from the state saved as the project grew: the
`BeamSearch`, the hints and the calibrated levels of a graph, the graph links (replayed by
`IndexEngine.build_searchgraph`), the codes of a quantized project, the posting lists of an
inverted file. A directory with no saved project gets the default project of
`create_project(workdir, dataset)`, which is saved.

`read_only=true` opens a project that another process, such as a running
`similarity-search-serve`, has open for writing. A writable open of such a directory raises the
lock error of RocksDB.

`runtime` holds the options that are not saved with the project (see [`Runtime`](@ref)).
`schema_version` is as in [`create_project`](@ref).
"""
function open_project(workdir::String, dataset::String; read_only::Bool=false, schema_version::Int=1,
                      runtime::Runtime=Runtime())
    dir = joinpath(workdir, dataset)
    project = Project.open_project(dir, dataset; read_only, extra_cf_names=_PROJECT_CFS)
    store = Persistence.open_engine_store(project.db)
    dense_vectors = Ref{Union{Nothing,MMapMatrixDatabase}}(nothing)
    partitions_ref = Ref{Union{Nothing,Persistence.PartitionStore}}(nothing)
    maxbatches = runtime.maxbatches
    spec = try
        s = _load_spec(store, read_only)
        _check_edit_correction(s === nothing ? :dense : IndexEngine.spec_kind(s), runtime)
        s
    catch
        Project.close_project(project)
        rethrow()
    end
    partition_by = Persistence.load_field(store, :partition_by, nothing)
    if partition_by !== nothing
        partitions_ref[] = Persistence.open_partition_store(project.db, partition_by,
                                                            runtime.postings_cache_max, runtime.postings_cache_base)
    end

    engine = if spec === nothing
        # Never saved: create the default project `create_project` would have, and save it.
        spec = DenseVectors()
        engine = IndexEngine.create_engine(spec; on_change=_on_change(spec, store, project),
                                           log_io=nothing, maxbatches, adj_factory=nothing, edit_correction=false)
        Persistence.save_field!(store, :spec, IndexEngine.spec_to_dict(spec))
        Persistence.save_fields!(store, IndexEngine.snapshot_state(engine))
        engine
    elseif IndexEngine.isquantized(spec)
        dense_vectors[] = Persistence.open_dense_vectors(dir; read_only)
        _open_quantized_engine(dir, store, project, spec, dense_vectors; read_only, maxbatches)
    elseif spec isa DenseVectors{Graph}
        dense_vectors[] = Persistence.open_dense_vectors(dir; read_only)
        adj_store = Persistence.open_adjacency_store(project.db)
        state = (
            kind=:dense, backend=:graph,
            distance=spec.distance,
            vector_blocks=Persistence.load_dense_vector_blocks(dense_vectors[]),
            load_neighbors=(i -> Persistence.load_neighbors(adj_store, i)),
            graph_len=Persistence.load_field(store, :graph_len, 0),
            minrecall=spec.index.minrecall,
            opt_beamsearch=Persistence.load_field(store, :opt_beamsearch, nothing),
            beamsearch=Persistence.load_field(store, :beamsearch, nothing),
            hints=Persistence.load_field(store, :hints, nothing),
            deleted_ids=Persistence.load_field(store, :deleted_ids, nothing),
        )
        IndexEngine.restore_engine(state; on_change=_searchgraph_on_change(store, adj_store), log_io=nothing, maxbatches)
    elseif spec isa DenseVectors
        dense_vectors[] = _open_exact_vectors(dir, store, read_only)
        index = dense_vectors[] === nothing && Persistence.has_field(store, :index) ?
            Persistence.load_field(store, :index, nothing) :
            IndexEngine.build_exact_index(spec, Persistence.load_dense_vector_blocks(dense_vectors[]))
        state = (
            kind=:dense, backend=IndexEngine.backend_symbol(spec),
            index=index,
            deleted_ids=Persistence.load_field(store, :deleted_ids, nothing),
        )
        IndexEngine.restore_engine(state; on_change=_on_change(spec, store, project), log_io=nothing, maxbatches)
    elseif spec isa SparseVectors
        obj_store = Persistence.open_invertedfile_object_store(project.db)
        state = (
            kind=:sparse, backend=:inverted_file,
            distance=spec.distance,
            dimension=spec.dimension,
            object_blocks=Persistence.load_object_blocks(obj_store),
            deleted_ids=Persistence.load_field(store, :deleted_ids, nothing),
        )
        IndexEngine.restore_engine(state; on_change=_invertedfile_on_change(obj_store), log_io=nothing, maxbatches)
    else
        # One branch for both text indexes: which inverted file to assemble is the spec's to say.
        staged_store = Persistence.open_staged_text_store(project.db)
        profile = Persistence.load_field(store, :profile, nothing)
        invfile_store = _invfile_store(project)
        distance = IndexEngine.text_distance(spec)
        adj_factory = _invfile_adj_factory(invfile_store, partitions_ref; maxlists=runtime.postings_cache_max,
                                           baselists=runtime.postings_cache_base)
        # No index on disk means no index: a text project is assembled from what was persisted or
        # it has none, and `index!` is what builds one out of the staged text.
        prebuilt = profile !== nothing && Persistence.has_inverted_index(invfile_store) ?
            _assemble_text_index(profile, IndexEngine.index_type(spec), distance, invfile_store, adj_factory,
                                 runtime.edit_correction) : nothing
        state = (
            kind=:text, backend=IndexEngine.backend_symbol(spec),
            profile=profile,
            fitspec=Persistence.load_field(store, :fitspec, nothing),
            distance=distance,
            prebuilt_index=prebuilt,
            adj_factory=adj_factory,
            edit_correction=runtime.edit_correction,
            staged=vcat(String[], Persistence.load_staged_text_blocks(staged_store)...),
            deleted_ids=Persistence.load_field(store, :deleted_ids, nothing),
        )
        IndexEngine.restore_engine(state; on_change=nothing, log_io=nothing, maxbatches)
    end
    handle = EmbeddedEngine(workdir, dataset, dir, project, engine, store, schema_version,
                            Persistence.load_field(store, :meta_schema, Schema.MetaSchema()),
                            Persistence.load_field(store, :oov_counters, OOVCounters()),
                            dense_vectors, read_only, Ref(false), partitions_ref,
                            (runtime.postings_cache_max, runtime.postings_cache_base), spec, Ref(false),
                            Dict{String,Any}())
    spec isa DenseVectors && partition_by !== nothing && _restore_dense_partitions!(handle)
    return handle
end

"""
    _open_exact_vectors(dir, store, read_only) -> Union{Nothing,MMapMatrixDatabase}

Opens the dense-vector file of an exact project of `Float32` vectors. A project saved before
2026-11-22 has no file, and its whole index is in the field `:index` (issue #3). A writable
open moves those vectors to the file, and then removes `:index`. The file is written under a
temporary name and renamed when it is complete, so an interrupted move leaves `:index` as the
only copy. A read-only open returns `nothing` for such a project, and the caller reads `:index`.
"""
function _open_exact_vectors(dir::String, store, read_only::Bool)
    raw = Persistence.open_dense_vectors(dir; read_only)
    read_only && return raw
    if raw === nothing && Persistence.has_field(store, :index)
        db = SimilaritySearch.database(Persistence.load_field(store, :index, nothing))
        if length(db) > 0
            path = Persistence.dense_vectors_path(dir)
            tmp = path * ".tmp"
            rm(tmp; force=true)
            out = MMapMatrixDatabase(tmp, length(db[1]), Float32)
            SimilaritySearch.append_items!(out, [Vector{Float32}(db[i]) for i in 1:length(db)])
            close(out)
            mv(tmp, path)
            raw = Persistence.open_dense_vectors(dir; read_only)
        end
    end
    # The file is complete here, or the project holds no vectors: `:index` is not needed.
    Persistence.delete_field!(store, :index)
    return raw
end

"""
    _open_quantized_engine(dir, store, project, spec, dense_vectors; read_only, maxbatches) -> DenseEngine

Rebuilds the engine of a quantized dense project (issue #1): an empty engine of `spec` with its
quantizer, and, once the encoder exists, the index over the codes file with the saved graph. The
staged vectors not yet encoded are counted from the dense-vector file and `:raw_offset`. A file
that holds only encoded vectors of a project that does not keep them is what an `index!`
interrupted before removing it leaves, and it is removed here.
"""
function _open_quantized_engine(dir::String, store, project, spec::DenseVectors, dense_vectors;
                                read_only::Bool, maxbatches)
    is_graph = spec.index isa Graph
    adj_store = is_graph ? Persistence.open_adjacency_store(project.db) : nothing
    on_change = is_graph ? _searchgraph_on_change(store, adj_store) : nothing
    engine = IndexEngine.create_engine(spec; on_change, log_io=nothing, maxbatches, adj_factory=nothing,
                                       edit_correction=false)
    engine.deleted_ids = something(Persistence.load_field(store, :deleted_ids, nothing), Set{UInt32}())
    if is_graph
        opt = Persistence.load_field(store, :opt_beamsearch, nothing)
        opt === nothing || (engine.backend.opt_beamsearch = opt)
        bs = Persistence.load_field(store, :beamsearch, nothing)
        bs === nothing || (engine.backend.index.algo[] = bs)
        hints = Persistence.load_field(store, :hints, nothing)
        hints === nothing || append!(engine.backend.index.hints, hints)
    end
    q = engine.quant
    encoder = Persistence.load_field(store, :encoder, nothing)
    if encoder !== nothing
        codes, ranges = Persistence.open_dense_codes(dir; read_only)
        if codes === nothing
            read_only && corrupted_storage("the codes file of this quantized project is missing; open it for writing once to create it")
            codes, ranges = _create_code_files(dir, encoder)
        end
        per_vector = ranges === nothing ? ScalarQuant.SQMinC[] :
            [ScalarQuant.SQMinC(r[1], r[2]) for r in (ranges[i] for i in 1:length(ranges))]
        IndexEngine.install_encoder!(engine, encoder, codes, ranges, per_vector)
        if is_graph
            graph_len = min(Persistence.load_field(store, :graph_len, 0), length(codes))
            IndexEngine.restore_adjacency!(engine.backend.index, i -> Persistence.load_neighbors(adj_store, i), graph_len)
        end
    end
    raw = dense_vectors[]
    if raw !== nothing
        pending = Persistence.load_field(store, :raw_offset, 0) + length(raw) - IndexEngine.quantized_length(q)
        pending >= 0 || corrupted_storage("the staged vectors of this quantized project end before its codes do")
        q.npending = pending
        if pending == 0 && !spec.storage.keep_vectors && !read_only
            close(raw)
            dense_vectors[] = nothing
            rm(Persistence.dense_vectors_path(dir); force=true)
        end
    end
    return engine
end

"""
    close_project!(handle::EmbeddedEngine; compact::Bool=true)

Closes the project's RocksDB connection. A dense project that has vectors also has its
`MMapMatrixDatabase` file, and this closes that file too.

No index is saved here. Each block of an index was saved when it was built. Every other field
(`deleted_ids`, `opt_beamsearch`, ...) is saved by the call that changed it (`delete_item!`,
`calibrate!`, ...). Before it closes the project, this function saves one value: the count of
query words not found in the vocabulary (see [`vocabulary_report`](@ref)).

Then, unless `compact=false`, it compacts the project's storage through
[`compact_project!`](@ref) -- a no-op for a read-only handle or a session that only read. That
is what keeps the *next* open fast (2.97s versus 0.28s on a 50k-vector project, measured
2026-09-10); pass `compact=false` when closing is on a latency path and the next open is not.

After it, every other function that takes `handle` raises [`ProjectClosed`](@ref). A second
`close_project!` on the same handle does nothing.
"""
function close_project!(handle::EmbeddedEngine; compact::Bool=true)
    handle.closed[] && return nothing
    # A follower reads the index of its leader, so it is closed first.
    for f in collect(values(handle.followers))
        close_follower!(f)
    end
    # The queries counted since the last write (see `OOVCounters`). A read-only handle counted
    # them for its own report and has nowhere to put them.
    if !handle.read_only && handle.oov.unflushed > 0
        handle.oov.unflushed = 0
        Persistence.save_field!(handle.store, :oov_counters, handle.oov)
    end
    handle.dense_vectors[] === nothing || close(handle.dense_vectors[])
    IndexEngine.isquantized(handle.engine) && IndexEngine.close_quantized!(handle.engine)
    compact && compact_project!(handle)
    handle.closed[] = true
    Project.close_project(handle.project)
    return nothing
end

"""
    compact_project!(handle::EmbeddedEngine; force::Bool=false) -> Bool

Compacts the project's RocksDB storage now, and reports whether it actually ran.

Skipped, and `false` returned, when the handle is read-only (it cannot write) or when this
session has not written anything -- opening a project to run queries and closing it again
leaves nothing to compact, and compaction is not free. `force=true` runs it regardless, for
a caller that knows another process did the writing.

Worth understanding rather than treating as housekeeping: what a bulk write session leaves
behind is a write-ahead log the *next* open has to replay, and that shows up as open
latency, not write latency. Measured 2026-09-10 on a 50k-vector dense project: 2.97s to
reopen without this, 0.28s with it, for 0.03s of compaction. It runs automatically from
[`close_project!`](@ref) (pass `compact=false` there to skip it), so an explicit call is for
compacting mid-session -- after a big ingest that the process will keep serving from, say.
"""
function compact_project!(handle::EmbeddedEngine; force::Bool=false)
    _check_open(handle)
    (handle.read_only || !(force || handle.wrote[])) && return false
    Project.compact_all!(handle.project)
    handle.wrote[] = false
    return true
end

"""
    _save_search_state!(handle::EmbeddedEngine) -> Nothing

Saves what a search of a graph project starts from, so that [`open_project`](@ref) sets it on
the rebuilt graph (PLAN.md §0, item 13): the `BeamSearch` (`index.algo[]`) as the engine field
`:beamsearch`, and the starting points of the search (`index.hints`) as `:hints`. Two calls
change them: [`index!`](@ref index!(::EmbeddedEngine)), whose context tunes the beam and picks
the hints while the graph grows, and [`calibrate!`](@ref), which leaves the last level it tuned.
Does nothing for any other project.
"""
function _save_search_state!(handle::EmbeddedEngine)
    engine = handle.engine
    engine isa IndexEngine.DenseEngine{IndexEngine.GraphBackend} || return nothing
    state = IndexEngine.read_lock(engine.lock) do
        index = engine.backend.index
        (beamsearch=index.algo[], hints=collect(UInt32, index.hints))
    end
    Persistence.save_fields!(handle.store, state)
    return nothing
end

"""
    index!(handle::EmbeddedEngine)

Catches up encoding/indexing over whatever's been staged via [`append_items!`](@ref)
since the last call (or since creation) -- one uniform entry point for every project kind:
idempotent, safe to call repeatedly, only ever processes the backlog (see
`IndexEngine.index!(engine::IndexEngine.DenseEngine{IndexEngine.GraphBackend})`'s docstring for
the exact contract, shared verbatim by the text engines). Nothing staged via `append_items!` becomes
visible to [`search`](@ref)/[`ftsearch`](@ref)/[`allknn`](@ref)/[`fft`](@ref)/
[`closestpairs`](@ref)/[`bichromatic_kclosestpairs`](@ref) until this runs at least once.

For a text project created with a `FitFromCorpus`, the first call additionally runs that fit
(producing a `TextSearch.TextProfile`: vocabulary, weights and lineage) over every item staged
so far and builds the real index against it -- there is no separate training entry point;
whatever you've `append_items!`-ed before the first `index!` call *is* the training corpus,
and tokens absent from it are out-of-vocabulary forever after. That one-time transition
(`:profile`) is persisted immediately after it happens; every later call only ever touches
the encoded posting-list blocks already covered by the engine's own incremental `on_change`,
so this does not re-save `:deleted_ids` or anything else on every catch-up call.

A project created with a `BaseProfile` (see [`create_project`](@ref)) was already trained
before its first item was staged, so there is no such transition and every call here is a
pure catch-up.

A no-op for the backends that have no staging split -- a sparse project's `InvertedFile` and a
dense project's `ExhaustiveSearch`/`ParallelExhaustiveSearch` both index on insertion, so there
is never a backlog here to catch up. It is deliberately a no-op and not an error: a caller
writing `create_project` / `append_items!` / `index!` / `search` gets four lines that mean the
same thing for every kind of project, and does not have to know which backends need the third
one. The cost of that is a call that does nothing, which is the cheaper mistake -- the
alternative made switching a project from a graph to a brute-force scan a change to every script
that fed it.
"""
function index!(handle::EmbeddedEngine)
    _check_open(handle)
    handle.wrote[] = true
    engine = handle.engine
    if IndexEngine.isquantized(engine)
        _index_quantized!(handle, Random.default_rng())
    elseif engine isa IndexEngine.DenseEngine{IndexEngine.GraphBackend}
        if engine.partitions === nothing
            IndexEngine.index!(engine)
        else
            parts = handle.partitions[]
            IndexEngine.index!(engine, ids -> Persistence._load_members(parts, ids))
            _save_partition_state!(handle)
        end
        _save_search_state!(handle)
    elseif IndexEngine.payload_kind(engine) === :text
        # `:fitspec`/`:distance` were already written by create_project's full snapshot_state
        # save, and neither ever changes afterwards -- only `:profile` can go from nothing to a
        # fitted model, and only once, so that is the only field this has to write back.
        just_fitted = IndexEngine.text_profile(engine) === nothing
        IndexEngine.index!(engine)
        just_fitted && Persistence.save_field!(handle.store, :profile, IndexEngine.text_profile(engine))
    else
        IndexEngine.index!(engine)   # sparse and exact: already indexed on append, a no-op
    end
    _sync_followers!(handle, OnIndex)
    return handle
end

"""
    QUANT_ENCODE_BLOCK

The number of staged vectors [`index!`](@ref index!(::EmbeddedEngine)) reads, encodes and inserts
at a time in a quantized project. It bounds the `Float32` vectors held in memory during an
`index!` to `QUANT_ENCODE_BLOCK × dim × 4` bytes: 64 MB at dimension 1024.
"""
const QUANT_ENCODE_BLOCK = 16_384

# `index!` of a quantized project (issue #1): fit the encoder if there is none, encode the staged
# vectors a block at a time, insert the codes, connect them on a graph, add the new ids to their
# partitions, and remove the staged vectors that `keep_vectors=false` does not keep.
function _index_quantized!(handle::EmbeddedEngine, rng::AbstractRNG)
    engine = handle.engine
    q = engine.quant
    _fit_quantizer!(handle, rng)
    raw = handle.dense_vectors[]
    parts = handle.partitions[]
    is_graph = engine isa IndexEngine.DenseEngine{IndexEngine.GraphBackend}
    if raw !== nothing && q.npending > 0
        offset = Persistence.load_field(handle.store, :raw_offset, 0)
        first_row = IndexEngine.quantized_length(q) - offset + 1
        for rows in Iterators.partition(first_row:length(raw), QUANT_ENCODE_BLOCK)
            codes = IndexEngine.encode_vectors(q.encoder, [raw[i] for i in rows])
            ids = IndexEngine.insert_codes!(engine, codes)
            # A graph adds the members of a partition when it connects them, below.
            if !is_graph && parts !== nothing
                IndexEngine.add_partition_members!(engine, first(ids), Persistence._load_members(parts, ids))
            end
        end
        IndexEngine.flush_quantized!(engine)
    end
    if is_graph
        if engine.partitions === nothing
            IndexEngine.index!(engine)
        else
            IndexEngine.index!(engine, ids -> Persistence._load_members(parts, ids))
            _save_partition_state!(handle)
        end
        _save_search_state!(handle)
    end
    _drop_encoded_vectors!(handle)
    return handle
end

# Fits the encoder of a quantized project on a sample of its staged vectors, saves it, creates
# the files of the codes, and installs the index over them. Does nothing when the encoder exists.
function _fit_quantizer!(handle::EmbeddedEngine, rng::AbstractRNG)
    engine = handle.engine
    q = engine.quant
    q.encoder === nothing || return nothing
    raw = handle.dense_vectors[]
    (raw === nothing || length(raw) == 0) &&
        nothing_staged("this quantized project has nothing staged yet -- append_items! at least one vector " *
                       "before index!; the quantizer is fitted on a sample of the staged vectors")
    n = length(raw)
    s = min(q.spec.quantizer.range.samplesize, n)
    rows = s == n ? (1:n) : sort!(Random.shuffle(rng, collect(1:n))[1:s])
    sample = reduce(hcat, (Vector{Float32}(raw[i]) for i in rows))
    encoder = IndexEngine.fit_encoder(q.spec, q.distance, sample, rng)
    # The encoder is saved before the files exist: a project reopened after an interruption
    # between the two creates empty files, and never holds codes without their encoder.
    Persistence.save_field!(handle.store, :encoder, encoder)
    codes, ranges = _create_code_files(handle.dir, encoder)
    IndexEngine.install_encoder!(engine, encoder, codes, ranges, ScalarQuant.SQMinC[])
    return nothing
end

function _create_code_files(dir::String, encoder)
    B = ScalarQuant.codewidth(encoder)
    cpath = Persistence.dense_codes_path(dir)
    rpath = Persistence.dense_code_ranges_path(dir)
    rm(cpath; force=true)
    rm(rpath; force=true)
    codes = MMapMatrixDatabase(cpath, cld(encoder.dim * B, 8), UInt8)
    ranges = ScalarQuant.isglobal(encoder) ? nothing : MMapMatrixDatabase(rpath, 2, Float32)
    return codes, ranges
end

# Removes the staged vectors of a quantized project once every one of them is encoded, unless
# the project keeps them. The next append starts a new file, whose first row is the next item
# (`:raw_offset`).
function _drop_encoded_vectors!(handle::EmbeddedEngine)
    q = handle.engine.quant
    raw = handle.dense_vectors[]
    (raw === nothing || q.spec.keep_vectors || q.npending > 0 || handle.read_only) && return nothing
    close(raw)
    handle.dense_vectors[] = nothing
    rm(Persistence.dense_vectors_path(handle.dir); force=true)
    return nothing
end

"""
    fit_quantizer!(handle::EmbeddedEngine; rng=Random.default_rng()) -> EmbeddedEngine

Fits the encoder of a quantized project on a sample of its staged vectors (issue #1), without
encoding them. [`index!`](@ref index!(::EmbeddedEngine)) fits it on its first call, so this is
needed only to choose the `rng` of the sample and of a random rotation, or to fit before the
whole collection is staged. Does nothing when the encoder exists. Raises `NothingStaged` when
nothing is staged, and `InvalidOption` for a project that is not quantized.
"""
function fit_quantizer!(handle::EmbeddedEngine; rng::AbstractRNG=Random.default_rng())
    _check_open(handle)
    IndexEngine.isquantized(handle.engine) ||
        invalid_option(:quantizer, "fit_quantizer! applies to a project with Quantized storage; this one stores Float32 vectors")
    handle.wrote[] = true
    _fit_quantizer!(handle, rng)
    return handle
end

"""
    quantizer_info(handle::EmbeddedEngine) -> Union{QuantizerInfo,Nothing}

The quantizer of a dense project (issue #1), or `nothing` for a project that stores `Float32`
vectors. See [`QuantizerInfo`](@ref) for the fields.
"""
quantizer_info(handle::EmbeddedEngine) = (_check_open(handle);
    IndexEngine.isquantized(handle.engine) ? IndexEngine.quantizer_info(handle.engine.quant) : nothing)

"""
    append_items!(handle::EmbeddedEngine, items) -> Int

Appends a batch of typed items -- [`DenseItem`](@ref)s for a dense project,
[`TextItem`](@ref)s for a text one, each already carrying its own `doc_id`,
`keywords`, `refs` and `meta`. Returns the number inserted, which is always `length(items)`:
every item in the batch is inserted, and an item of the wrong kind for this project raises
rather than being skipped.

That is the change from the dictionary form this replaced, and it is worth being explicit
about. Appending `Dict("txt" => "...")` to a text project used to insert nothing and report
zero, because the item had no `"text"` key and the loop skipped it -- a typo in a key name
produced a silent no-op and a count the caller had to think to check. There is no key to
misspell now, and nothing is skipped.

For a dense (`SearchGraph`) project or a text (`BM25InvertedFile`/`InvertedFile`) project
alike, this only *stages* items -- durably (see "Performance" below) but without any
graph-linking/encoding work -- so freshly appended items are *not* yet visible to
[`search`](@ref)/[`ftsearch`](@ref)/[`allknn`](@ref)/etc. until an explicit
[`index!`](@ref index!(::EmbeddedEngine)) call catches up the backlog (for a text
project's very first `index!` call, that also trains its `Vocabulary` -- there's no
separate training entry point or "must be trained first" error here anymore).
The backends with no staging split -- the exact dense ones
(`ExhaustiveSearch`/`ParallelExhaustiveSearch`) and a sparse project's `InvertedFile` -- index
synchronously on every `add_item!` instead (see
`IndexEngine.index!(engine::IndexEngine.DenseEngine{GraphBackend})`'s docstring for why
`SimilaritySearch.jl`/`TextSearch.jl` support the split where they do).

Persistence: for a dense project, every batch's raw vectors are appended directly to the
project's `MMapMatrixDatabase` (see [`index!`](@ref index!(::EmbeddedEngine))'s docstring
and the "Performance" note below); for a text project, every batch's raw text is appended
directly to its own `Persistence.StagedTextStore` -- both happen right here, at stage
time, not via `on_change`, since an item is already durable-worthy the moment it's staged,
long before any graph-linking/encoding happens. Once an [`index!`](@ref index!(::EmbeddedEngine))
call actually encodes/indexes a text project's backlog, `SimilaritySearch.CallbackLog` persists
each newly-encoded object incrementally (see [`_invertedfile_on_change`](@ref)).
An exact dense project of `Float32` vectors indexes each vector at once, and it writes the
vectors to the same file as a graph project. Its index is rebuilt from that file when the
project is opened (issue #3).

# Performance: batch size matters a lot for a `SearchGraph` (dense) project

Every call here does *one* `SimilaritySearch.append_items!` call into the project's
`MMapMatrixDatabase` for its whole batch of staged vectors, followed by *one*
`SimilaritySearch.flush` of it (durability is opt-in on that type now -- see its docstring's
"Durability" section -- and staying durable right away, as documented above, is this
function's job to arrange, not the database's). So the number of calls to *this* function a
caller makes (not the total vector count) is what drives wall-clock time, exactly as before:
confirmed empirically, 200,000 128-dim vectors written as one project's
dense store: one 200,000-item call (1 fsync) took ~0.10s; the same total split into calls
of 1,000 items (200 fsyncs) took ~0.14s; split into calls of 100 items (2,000 fsyncs) took
~0.31s; split into calls of 10 items (20,000 fsyncs) took ~2.1s. Calling this function once
per single item (the worst case, one fsync per item) is dramatically worse still --
measured independently at roughly 1000x slower than an all-at-once call for the same total
count. Prefer fewer, larger calls to this function over many small ones whenever a caller
controls the batching.
"""
# A project indexes one kind of thing, and the item type says which. Checked per item rather
# than once per batch so a heterogeneous `Vector{AbstractItem}` is caught on the offending
# element instead of on whatever happened to be first.
# Which item a project takes is `payload_kind` spelled the other way round, so it is derived
# rather than restated: a fourth engine kind would otherwise need remembering here too.
const _ITEM_KIND = Dict(Schema.DenseItem => :dense, Schema.SparseItem => :sparse,
                        Schema.TextItem => :text)
const _KIND_ITEM = Dict(v => k for (k, v) in _ITEM_KIND)

_check_item(engine, item::Schema.AbstractItem) =
    _ITEM_KIND[typeof(item)] === IndexEngine.payload_kind(engine) || payload_mismatch(
        "this project indexes $(IndexEngine.payload_kind(engine)), so it takes " *
        "$(nameof(_KIND_ITEM[IndexEngine.payload_kind(engine)]))s; got a $(typeof(item))" *
        (item.doc_id === nothing ? "" : " (doc_id $(item.doc_id))"))

function append_items!(handle::EmbeddedEngine, items::AbstractVector{<:Schema.AbstractItem})
    _check_open(handle)
    handle.wrote[] = true
    engine = handle.engine
    project = handle.project
    is_text = IndexEngine.payload_kind(engine) === :text
    is_dense_graph = engine isa IndexEngine.DenseEngine{IndexEngine.GraphBackend}
    # A graph project and a quantized project of any backend stage the vectors in the
    # dense-vector file, and `index!` indexes them. An exact project of `Float32` vectors indexes
    # them here, and it writes them to the same file, from which its index is rebuilt (issue #3).
    stages_vectors = is_dense_graph || IndexEngine.isquantized(engine)
    writes_vectors = IndexEngine.payload_kind(engine) === :dense
    staged_vectors = Vector{Float32}[]
    staged_texts = String[]
    text_sp = is_text ? _current_size(engine) + 1 : 0
    # Checked for the whole batch before anything is written, so that a bad value in the last
    # item does not leave the first ones stored without their partitions.
    parts = handle.partitions[]
    part_values = parts === nothing ? nothing :
        [_partition_values(parts.field, item.meta) for item in items]

    # A dense project records the partitions of the batch before its items: an interrupted call
    # then leaves entries for ids the next append overwrites, never items without partitions.
    dense_sp = _current_size(engine) + 1
    dense_members = parts !== nothing && IndexEngine.payload_kind(engine) === :dense ?
        (foreach(item -> _check_item(engine, item), items); Persistence.assign_partitions!(parts, dense_sp, part_values)) :
        nothing

    for item in items
        _check_item(engine, item)
        p = Schema.payload(item)
        IndexEngine.add_item!(engine, p)
        if is_text
            push!(staged_texts, p)
        elseif writes_vectors
            push!(staged_vectors, p)
        end

        _id = Int32(_current_size(engine))
        # An empty `meta` is stored as nothing rather than as an empty JSON object: there is
        # no difference to read back (`get_meta` answers `nothing` either way) and one of them
        # is a write that never had to happen.
        meta = isempty(item.meta) ? nothing : Schema.coerce_meta(handle.meta_schema, item.meta)
        put_metadata!(project, Schema.metadata_record(item, _id, handle.schema_version), meta)
    end

    # An exact project indexes on append, so its partitions grow here. A graph project adds the
    # members of a partition when `index!` indexes them.
    if dense_members !== nothing && !stages_vectors
        IndexEngine.add_partition_members!(engine, dense_sp, dense_members)
    end

    if writes_vectors && !isempty(staged_vectors)
        if handle.dense_vectors[] === nothing
            # The first row of a new file of a quantized project is the item after the ones
            # already encoded (see `_drop_encoded_vectors!`). Saved before the file exists, so
            # the file never exists without it.
            IndexEngine.isquantized(engine) && Persistence.save_field!(handle.store, :raw_offset, dense_sp - 1)
            handle.dense_vectors[] = MMapMatrixDatabase(Persistence.dense_vectors_path(handle.dir), length(staged_vectors[1]), Float32)
        end
        SimilaritySearch.append_items!(handle.dense_vectors[], staged_vectors)
        # `MMapMatrixDatabase` no longer msyncs/fsyncs on its own (see its docstring's
        # "Durability" section) -- making a staged batch durable right away, which is the
        # contract this function documents, is this module's job now.
        SimilaritySearch.flush(handle.dense_vectors[])
    end

    if is_text && !isempty(staged_texts)
        # Before the staged text, which is what makes the items exist on the next open: an
        # interrupted call then leaves partition entries for ids the next append overwrites,
        # never items without their partitions.
        parts === nothing || Persistence.assign_partitions!(parts, text_sp, part_values)
        Persistence.append_staged_texts!(Persistence.open_staged_text_store(project.db), text_sp, staged_texts)
    end

    _sync_followers!(handle, OnWrite)
    return length(items)
end

"""
    append_items!(handle::EmbeddedEngine, item::Schema.AbstractItem) -> Int

One item, for a caller that has one. Note the performance warning above: this is the
single-item call, and for a dense project it costs one `fsync` for one vector.
"""
append_items!(handle::EmbeddedEngine, item::Schema.AbstractItem) = append_items!(handle, [item])

"""
    _current_size(engine::IndexEngine.AbstractSearchEngine) -> Int

The count that determines the next item's `_id`, which has to be the *staged* count wherever
staging and indexing are separate: for a `DenseEngine{GraphBackend}`, the number of staged vectors
(`length(database(engine.backend.index))`, i.e. `engine.backend.index.db`'s own count), and for a
`FullTextEngine`, the number of staged texts (`length(engine.staged)`). Since
[`append_items!`](@ref) only stages on those two, `length(engine.backend.index)` (the
encoded/indexed count) would lag behind and hand out `_id`s already taken.

The fallback method is `length(engine.backend.index)`, which is right for every backend that
indexes on insertion -- the exact dense ones and a sparse project's `InvertedFile` -- because
there the index already reflects the item just added.
"""
_current_size(engine::IndexEngine.DenseEngine{IndexEngine.GraphBackend}) =
    length(SimilaritySearch.database(engine.backend.index)) + _quant_pending(engine)
_current_size(engine::IndexEngine.DenseEngine) = length(engine.backend.index) + _quant_pending(engine)
_quant_pending(engine::IndexEngine.DenseEngine) = IndexEngine.isquantized(engine) ? engine.quant.npending : 0
_current_size(engine::IndexEngine.FullTextEngine) = length(engine.staged)
_current_size(engine::IndexEngine.AbstractSearchEngine) = length(engine.backend.index)

"""
    _search_with_filter(engine, project, query, k, predicate; minrecall, candidates)

Top-`k` under a metadata `predicate`: asks the index for `candidates` hits, keeps the ones
the predicate accepts, stops at `k`.

One pass, not a widening loop. The candidate budget is what the caller pays for and what
the caller therefore names: every candidate costs two RocksDB reads and two decodes (the
record and its `meta`), so a budget that grows itself until `k` is satisfied would let a
selective predicate turn one query into a full scan without anybody asking for it. See
[`search`](@ref) for what to set it to.
"""
function _search_with_filter(engine::IndexEngine.AbstractSearchEngine, project::Project.ProjectManager, query, k::Int, predicate; minrecall, candidates::Int)
    raw = IndexEngine.search_live(engine, query, max(candidates, k); bs_override=nothing, minrecall, policy=nothing)
    return _filter_hits(project, raw, k, predicate)
end

"""
    _filter_hits(project, raw, k, predicate) -> (id=..., dist=..., deleted=..., distance_evaluations=..., candidates=...)

The first `k` hits of `raw`, in the order of `raw`, whose record and `meta` satisfy
`predicate`. A deleted hit never satisfies it. `candidates` is the length of `raw`. Shared by
[`search`](@ref) and [`ftsearch`](@ref).
"""
function _filter_hits(project::Project.ProjectManager, raw, k::Int, predicate)
    ids = Int32[]
    dists = Float32[]
    for (_id, dist, deleted) in zip(raw.id, raw.dist, raw.deleted)
        # A soft-deleted id has no metadata a caller is allowed to see, so it can never
        # satisfy a metadata predicate -- skip it without even attempting the lookup.
        deleted && continue
        record = get_metadata(project, _id)
        record === nothing && continue
        meta = get_meta(project, _id)
        predicate(record, meta) || continue
        push!(ids, _id)
        push!(dists, dist)
        length(ids) == k && break
    end
    return (id=ids, dist=dists, deleted=falses(length(ids)), distance_evaluations=raw.distance_evaluations,
            candidates=length(raw.id))
end

"""
    _record_stats!(stats, res_knn) -> Nothing

Writes the cost of one search into `stats` (see [`SearchStats`](@ref)), if a `SearchStats` was
given. Without a filter, the number of candidates is the number of results.
"""
function _record_stats!(stats::Union{Nothing,SearchStats}, res_knn)
    stats === nothing && return nothing
    stats.distance_evaluations = res_knn.distance_evaluations
    stats.candidates = hasproperty(res_knn, :candidates) ? res_knn.candidates : length(res_knn.id)
    return nothing
end

function _hydrate_results(project::Project.ProjectManager, res_knn)
    results = SearchResult[]
    for (_id, dist, deleted) in zip(res_knn.id, res_knn.dist, res_knn.deleted)
        if deleted
            # Metadata for a soft-deleted _id is not accessible through search, so there is no
            # doc_id to report -- `nothing`, rather than the internal id stringified into the
            # external id's field, which is what the named-tuple version did and which a caller
            # had no way to tell apart from a real doc_id that happened to look like a number.
            push!(results, SearchResult(Int32(_id), nothing, Float32(dist), true))
            continue
        end
        record = get_metadata(project, _id)
        push!(results, SearchResult(Int32(_id), record === nothing ? nothing : record.doc_id,
                                    Float32(dist), false))
    end
    return results
end

"""
    _resolve_records(project, raw_id) -> Vector{Schema.MetadataRecord}

The records `raw_id` names, whether it is an internal `_id` or a caller-supplied `doc_id`.

Tries the numeric reading first, because it is a direct key lookup, and it can only ever
name one record. A numeric-looking `doc_id` therefore resolves as an `_id` if one exists
with that value -- an ambiguity inherent to accepting both in one argument, and the reason
[`fetch_items`](@ref) and [`exists`](@ref) share this function rather than each inventing its
own precedence.

Everything else is a `doc_id` lookup through `Project.find_all_by_doc_id`, which returns
however many records carry exactly that `doc_id` -- zero, one, or several. A vector, not a
record, because `doc_id` is a caller-chosen external id that nothing here requires to be
unique: appending two items under the same one is allowed, and the scan this replaced
answered such a lookup with whichever of them it reached first.
"""
function _resolve_records(project::Project.ProjectManager, raw_id)
    id_str = string(raw_id)
    maybe_int = tryparse(Int, id_str)
    if maybe_int !== nothing
        record = get_metadata(project, maybe_int)
        record === nothing || return [record]
    end
    find_all_by_doc_id(project, id_str)
end

"""
    _search_query(engine, vector) -> Vector{Float32} | SparseVector{Float32,Int32}

`vector` in the form this project's backend searches with.

A dense project's index evaluates a distance against a contiguous vector; a sparse project's
`InvertedFile` walks the query's nonzero positions to pick posting lists, and reads its
`nzind`/`nzval` directly. So the conversion is the project's, not the caller's -- and the sparse
case cannot fall back on the dense one: `convert(Vector{Float32}, sparse_query)` type-checks,
densifies, and then dies inside `select_posting_lists` on a `Float32` used as an index.

A dense vector handed to a sparse project is rejected rather than sparsified. It would usually
be a mistake worth naming (a project's items are `SparseItem`s, so its queries are sparse too),
and sparsifying silently would turn a dimension mismatch into posting lists selected from
whatever the values happened to be.
"""
function _search_query(engine::IndexEngine.AbstractSearchEngine, vector)
    kind = IndexEngine.payload_kind(engine)
    kind === :dense && IndexEngine.isquantized(engine) &&
        return IndexEngine.prepare_query(engine, convert(Vector{Float32}, vector))
    kind === :dense && return convert(Vector{Float32}, vector)
    kind === :sparse || payload_mismatch("search(handle, vector) needs a dense or sparse project; this one holds $kind")
    vector isa AbstractSparseVector ||
        payload_mismatch("a sparse project searches with a SparseVector{Float32,Int32}, not a $(typeof(vector)); " *
              "build one with `sparsevec(indices, values, dimension)` over the same dimension the " *
              "project was created with ($(engine.backend.dimension))")
    length(vector) == engine.backend.dimension ||
        wrong_dimension(engine.backend.dimension, length(vector),
              "query dimension $(length(vector)) does not match the project's $(engine.backend.dimension)")
    convert(SparseVector{Float32,Int32}, vector)
end

"""
    search(handle::EmbeddedEngine, vector, k::Int=10; filter=nothing) -> Vector{SearchResult}

Dense vector search, hydrated with each hit's original id (mirrors `Server.handle_search`
minus the HTTP/telemetry/pagination machinery). `filter`, if given, is a
`(record::Schema.MetadataRecord, meta) -> Bool` predicate function called on each
candidate's record and (lazily decoded, see `Schema.decode_meta`) `meta` -- e.g.
`(record, meta) -> meta !== nothing && meta["year"] >= 2020`. A caller here already has real
Julia functions to work with, not a JSON wire format to encode a filter into, so `filter`
is just that function, not a name-keyed spec for some interpreter to replay. `meta` is
fetched (one extra `Project.get_meta` read per candidate) even if `record` alone would've
been enough for a given predicate -- the common case (filtering on a `meta` field) needs it,
and there's no way to know a predicate won't from here.

`candidates` is how many hits the index is asked for before filtering, and it matters
whenever `filter` is given: **the search returns fewer than `k` results when fewer than `k`
of those candidates satisfy the predicate**, quietly, because the alternative -- growing the
budget until `k` is met -- turns a selective predicate into a full scan of the collection at
two RocksDB reads per item. The default of `2k` suits a predicate that most items pass (a
language, a type, a recent date). Match it to how selective yours is: a predicate that
roughly one item in `n` satisfies needs about `n * k` candidates to fill a page of `k`, so
`search(h, q, 10; filter=f, candidates=8192)` is the shape of a query filtering on something
rare, and the cost of that is ~8k candidate reads. Without `filter`, `candidates` is unused.

Without `filter`, this is a plain top-`k` search: soft-deleted candidates are not hidden
or backfilled -- they're returned with `deleted=true` and no hydrated metadata (see
[`IndexEngine.search_live`](@ref)), so `doc_id` is `nothing` for them.
Getting `k` *live* results back is a paging concern for a layer above this one (e.g. a
server walking successive windows via a cursor), not something this function does itself.

`minrecall`, for a `DenseEngine{GraphBackend}`, searches at (approximately) that target recall
using a calibrated `BeamSearch` from `engine.backend.opt_beamsearch` instead of its current
default (see [`IndexEngine.search_live`](@ref)) -- if that table is still empty, this
triggers a one-off `calibrate!` over `IndexEngine.DEFAULT_MINRECALL_LEVELS` and persists
the resulting `opt_beamsearch` so that calibration isn't silently repeated on every future
search. Ignored for any other engine kind.

Returns a `Vector{`[`SearchResult`](@ref)`}` -- `_id` is the internal id, `doc_id` the
caller's own, and note that those two field names sit the opposite way round from the named
tuple this replaced.

`stats`, if given a [`SearchStats`](@ref), receives this call's distance-evaluation count --
the only way to get it, since the context that counted them goes back to the engine's pool
when the call returns.

`partition`, on a partitioned dense project, searches only the items of that partition
(PLAN.md §5.10). A partition without its own graph is scanned, and the result is exact. A
partition with a graph is searched with the `BeamSearch` its construction tuned. With
`minrecall`, it uses the levels of [`calibrate!`](@ref)`(handle; partition)`, and it raises
`NotTrained` if the partition was never calibrated: a search does not calibrate a partition.
`filter` and `candidates` apply to the results of the partition. A name the project does not
hold raises `UnknownPartition`.
"""
function search(handle::EmbeddedEngine, vector, k::Int=10; filter=nothing, minrecall=nothing, candidates::Int=2k,
                stats::Union{Nothing,SearchStats}=nothing, partition::Union{Nothing,AbstractString}=nothing)
    _check_open(handle)
    query = _search_query(handle.engine, vector)
    partition === nothing || return _search_partition(handle, query, k, partition; filter, minrecall, candidates, stats)
    needs_save = minrecall !== nothing && handle.engine isa IndexEngine.DenseEngine{IndexEngine.GraphBackend} && isempty(handle.engine.backend.opt_beamsearch)
    res_knn = filter === nothing ?
        IndexEngine.search_live(handle.engine, query, k; bs_override=nothing, minrecall, policy=nothing) :
        _search_with_filter(handle.engine, handle.project, query, k, filter; minrecall, candidates)
    if needs_save
        Persistence.save_field!(handle.store, :opt_beamsearch, handle.engine.backend.opt_beamsearch)
        _save_search_state!(handle)
    end
    _record_stats!(stats, res_knn)
    return _hydrate_results(handle.project, res_knn)
end

"""
    _search_partition(handle, query, k, partition; filter, minrecall, candidates, stats) -> Vector{SearchResult}

[`search`](@ref) in one partition of a dense project (PLAN.md §5.10). `filter` and
`candidates` apply to the results of the partition.
"""
function _search_partition(handle::EmbeddedEngine, query, k::Int, partition::AbstractString; filter, minrecall,
                           candidates::Int, stats::Union{Nothing,SearchStats})
    engine = handle.engine
    engine isa IndexEngine.DenseEngine ||
        invalid_option(:partition, "search with a partition needs a dense project; this one holds " *
              "$(IndexEngine.payload_kind(engine)) items")
    number = _partition_number(handle, partition)
    n = filter === nothing ? k : max(candidates, k)
    raw = IndexEngine.search_partition_live(engine, number, query, n, minrecall)
    res_knn = filter === nothing ? raw : _filter_hits(handle.project, raw, k, filter)
    _record_stats!(stats, res_knn)
    return _hydrate_results(handle.project, res_knn)
end

"""
    _batch_query_database(engine, queries) -> SimilaritySearch.AbstractDatabase

`queries` in the form this project's index searches a whole batch with: a `MatrixDatabase`
of `Float32` columns for a dense project, a `VectorDatabase` of `SparseVector{Float32,Int32}`
for a sparse one.

Each query goes through [`_search_query`](@ref) first, so a batch rejects exactly what a
single query rejects (wrong payload kind, wrong dimension) and says the same thing when it
does -- the batch entry point is not a second, laxer door into the same index.
"""
function _batch_query_database(engine::IndexEngine.AbstractSearchEngine, queries)
    isempty(queries) && invalid_option(:queries, "searchbatch needs at least one query")
    kind = IndexEngine.payload_kind(engine)
    kind === :text &&
        unsupported_operation(:searchbatch, "searchbatch is for dense and sparse projects; a text project answers strings, " *
              "so batch it with `[ftsearch(h, q, k) for q in queries]` (one query at a time)")
    prepared = [_search_query(engine, q) for q in queries]
    if kind === :dense
        dim = length(first(prepared))
        all(q -> length(q) == dim, prepared) ||
            wrong_dimension(dim, length(first(prepared)),
                  "every query in a batch must have the project's dimension ($dim)")
        M = Matrix{Float32}(undef, dim, length(prepared))
        for (j, q) in enumerate(prepared)
            M[:, j] = q
        end
        return MatrixDatabase(M)
    end
    return VectorDatabase(prepared)
end

"""
    searchbatch(handle::EmbeddedEngine, queries, k::Int=10) -> (ids, dists)

`k` nearest neighbours for every query at once: two `(k, length(queries))` matrices, of
`UInt32` internal ids and `Float32` distances, exactly as the library returns them. A column
with fewer than `k` hits is padded with `0` ids and `typemax(Float32)` distances.

This is the raw, unhydrated form -- no `doc_id`, no soft-delete flags, no RocksDB reads at
all -- and it exists because that is what the operations built on a batch of queries
(evaluation loops, offline scoring, feeding another index) actually want.
`search(handle, queries, k)` is the hydrated form.

**A whole-dataset operation, in the same family as [`allknn`](@ref)/[`fft`](@ref)**, not a
concurrent search: it takes the project's write lock (full exclusivity, including from other
batch calls and from `index!`) and refuses to run while items are staged but not yet indexed.
See `IndexEngine.searchbatch_live` for why that is not a choice.

Worth it whenever there are many queries: measured 2026-09-10 over 50k dense vectors on 8
threads, 39k queries/s here against 3.6k/s calling [`search`](@ref) one query at a time.

A text project has no batch entry point here -- its queries are strings the library resolves
one at a time -- so call [`ftsearch`](@ref) in a loop.
"""
function searchbatch(handle::EmbeddedEngine, queries, k::Int=10)
    _check_open(handle)
    Q = _batch_query_database(handle.engine, queries)
    IndexEngine.searchbatch_live(handle.engine, Q, k)
end

"""
    search(handle::EmbeddedEngine, queries::AbstractVector{<:AbstractVector{<:Real}}, k::Int=10) -> Vector{Vector{SearchResult}}

Hydrated batch search: one inner vector of [`SearchResult`](@ref) per query, in the order
`queries` gave, each ordered by ascending distance.

The same results [`search`](@ref) returns for one query, computed for all of them in one
parallel pass ([`searchbatch`](@ref)) and hydrated the same way -- one `doc_id` lookup and one
soft-delete check per hit. It carries `searchbatch`'s constraints with it (write lock, no
staged backlog, dense or sparse only), because it is that call plus hydration.

No `filter` keyword here: filtering is a per-query candidate budget (see [`search`](@ref)'s
`candidates`), and mixing that with a batch whose whole point is one uniform `k` for every
query would make the cost of a call impossible to read off it. Ask for a larger `k` and apply
the predicate yourself, or loop over the single-query [`search`](@ref) when the predicate is
what matters.
"""
function search(handle::EmbeddedEngine, queries::AbstractVector{<:AbstractVector{<:Real}}, k::Int=10)
    _check_open(handle)
    ids, dists = searchbatch(handle, queries, k)
    engine = handle.engine
    out = Vector{Vector{SearchResult}}(undef, size(ids, 2))
    for j in axes(ids, 2)
        hits = SearchResult[]
        for i in axes(ids, 1)
            id = ids[i, j]
            id == 0 && continue
            _id = Int32(id)
            if _id in engine.deleted_ids
                push!(hits, SearchResult(_id, nothing, Float32(dists[i, j]), true))
            else
                record = get_metadata(handle.project, _id)
                push!(hits, SearchResult(_id, record === nothing ? nothing : record.doc_id,
                                         Float32(dists[i, j]), false))
            end
        end
        out[j] = hits
    end
    return out
end

"""
    ftsearch(handle::EmbeddedEngine, text, k::Int=10; policy=QueryPolicy(), partition=nothing,
             filter=nothing, candidates=2k, stats=nothing) -> Vector{SearchResult}
    ftsearch(handle::EmbeddedEngine, tokens::AbstractVector{<:AbstractString}, k::Int=10; ...)

Text search against a bm25/weighted-inverted-file project (mirrors `Server.handle_ftsearch`).
Same soft-delete marker behavior as [`search`](@ref); `minrecall` is accepted only for
parity with `search` and is a no-op here since no text engine has a `BeamSearch` to
calibrate.

`policy::TextSearch.QueryPolicy` says how to treat the query: whether to correct its
spelling against the project's vocabulary (`correction=:auto` by default, `:off` to search
exactly what was typed, `:always` to bridge every token) and whether to widen it with the
profile's query-expansion network (`expansion`, `expansion_k`). Both are guesses about
intent rather than properties of the index, which is why they travel with the query instead
of being fixed at project creation. The default is inert for a project fitted under the
default `TextConfig()`, whose policy already folds case and diacritics; it starts mattering
for one built on a profile that preserves them. Use [`ftexplain`](@ref) to see what a query
was actually searched as.

`stats` behaves exactly as in [`search`](@ref).

`partition` searches one partition of a project created with `partition_by` (see
[`create_project`](@ref) and [`declare_partition!`](@ref)): only the items of that partition are
evaluated, and each gets the score the global search gives it. A name the project's partition
table does not hold raises [`UnknownPartition`](@ref), whose message lists the closest names. A
partition that exists but has no indexed item yet returns no results. `partition` on a project
without a partition field raises `InvalidOption`.

`filter` and `candidates` work as in [`search`](@ref): `filter` is a `(record, meta) -> Bool`
predicate, and the search examines the best `candidates` results in order of score and returns
the first `k` that satisfy it. It returns fewer than `k` results when fewer than `k` of those
candidates satisfy it. `stats.candidates` then tells whether a larger budget can find more (see
[`SearchStats`](@ref)). With `partition`, the candidates are those of the partition. The
predicate reads the record and the `meta` of each candidate it examines, two RocksDB reads per
candidate. Without `filter`, `candidates` is unused.

A query given as a list of tokens is searched with those tokens, without tokenizing them again.
Each token is a token of the project's vocabulary, as `TextSearch.query_tokens` or
`text_profile(handle).model.voc` gives them. `policy` still applies correction and expansion to
them; pass `QueryPolicy(correction=:off, expansion=false)` to search exactly those tokens. A
caller that selects the tokens of a long text, for example the ones with the highest idf, can
search exactly its selection this way. Tokenizing a token again can change it: a tokenizer that
replaces digits turns `A2` into `A0`.
"""
function ftsearch(handle::EmbeddedEngine, query::Union{AbstractString,AbstractVector{<:AbstractString}}, k::Int=10;
                  minrecall=nothing, policy::QueryPolicy=QueryPolicy(),
                  stats::Union{Nothing,SearchStats}=nothing, partition::Union{Nothing,AbstractString}=nothing,
                  filter=nothing, candidates::Int=2k)
    _check_open(handle)
    n = filter === nothing ? k : max(candidates, k)
    raw = partition === nothing ?
        IndexEngine.search_live(handle.engine, query, n; bs_override=nothing, minrecall, policy) :
        IndexEngine.search_partition_live(handle.engine, _partition_adj(handle, partition), query, n; policy)
    res_knn = filter === nothing ? raw : _filter_hits(handle.project, raw, k, filter)
    _count_oov!(handle, query)
    _record_stats!(stats, res_knn)
    return _hydrate_results(handle.project, res_knn)
end

"""
    partition_field(handle::EmbeddedEngine) -> Union{Nothing,String}

The `meta` field this project is partitioned by, or `nothing` when it declares none (PLAN.md
§5.9). See [`create_project`](@ref)'s `partition_by` and [`declare_partition!`](@ref).
"""
function partition_field(handle::EmbeddedEngine)
    _check_open(handle)
    parts = handle.partitions[]
    parts === nothing ? nothing : parts.field
end

"""
    partition_graph_min(handle::EmbeddedEngine) -> Union{Nothing,Int}

The number of indexed members from which a partition of this dense project has its own graph
(PLAN.md §5.10), or `nothing` for a project that is not a partitioned dense project. See
`graph_min` of [`PartitionBy`](@ref).
"""
function partition_graph_min(handle::EmbeddedEngine)
    _check_open(handle)
    (handle.engine isa IndexEngine.DenseEngine && handle.partitions[] !== nothing) || return nothing
    return Persistence.load_field(handle.store, :partition_graph_min, DEFAULT_PARTITION_GRAPH_MIN)
end

"""
    partitions(handle::EmbeddedEngine) -> Vector{@NamedTuple{name::String, items::Int, graph::Bool}}

The partitions of this project, sorted by name, each with the number of items appended to it.
That number counts items that are not indexed yet and items deleted since, because both are
still members of the partition. An item in several partitions is counted in each. Raises
`InvalidOption` on a project without a partition field.

`graph` says whether the partition has its own graph (PLAN.md §5.10). It is `true` only for a
partition of a dense `SearchGraph` project with at least `graph_min` indexed members (see
[`PartitionBy`](@ref)).
A text partition has posting lists, and `graph` is `false`.
"""
function partitions(handle::EmbeddedEngine)
    _check_open(handle)
    parts = _partition_store(handle, :partitions)
    names = Persistence.partition_names(parts)
    counts = Persistence.partition_counts(parts)
    engine = handle.engine
    graphs = engine isa IndexEngine.DenseEngine ? IndexEngine.partition_graphs(engine) : Dict{UInt32,Bool}()
    sort!([(name=n, items=c, graph=get(graphs, UInt32(i), false)) for (i, (n, c)) in enumerate(zip(names, counts))];
          by=p -> p.name)
end

"""
    declare_partition!(handle::EmbeddedEngine, partitions::PartitionBy)
    declare_partition!(handle::EmbeddedEngine, field::AbstractString)
        -> Vector{@NamedTuple{name::String, items::Int, graph::Bool}}

Partitions an existing text or dense project by a `meta` field, as `create_project`'s
`partitions` does for a new one (PLAN.md §5.9 and §5.10), and returns [`partitions`](@ref).
A field name is `PartitionBy(field)`.

On a text project, it reads the `meta` of every item once, and builds the partition lists by
splitting the global lists already on disk. It does not read or tokenize any text. Items
appended but not indexed yet get their partition lists from the next
[`index!`](@ref index!(::EmbeddedEngine)) call. It holds the engine's write lock while it runs,
so searches wait for it.

On a dense project, it reads the `meta` of every item once, and builds the graph of each
partition with at least `graph_min` indexed members (by default
[`DEFAULT_PARTITION_GRAPH_MIN`](@ref)). Building a graph inserts its members again, so this
costs about as much as indexing those members once. Items appended but not indexed yet join
their partitions in the next `index!`.

Every value of the field is checked as [`append_items!`](@ref) checks it, and the first bad one
raises `PayloadMismatch` before the declaration is recorded. The declaration is recorded last,
so an interrupted call leaves the project unpartitioned, and calling it again starts over.

Raises `InvalidOption` on a sparse project, on a project that already has a partition field,
and for a `graph_min` on a project that is not dense. Raises `UnsupportedOperation` on a read-only
handle.
"""
declare_partition!(handle::EmbeddedEngine, field::AbstractString) = declare_partition!(handle, PartitionBy(field))

function declare_partition!(handle::EmbeddedEngine, by::PartitionBy)
    _check_open(handle)
    field = by.field
    handle.read_only && unsupported_operation(:declare_partition!, "this project is open read-only")
    engine = handle.engine
    kind = IndexEngine.payload_kind(engine)
    kind === :sparse &&
        invalid_option(:partitions, "a text or a dense project can be partitioned; a sparse project cannot " *
              "be partitioned yet (PLAN.md §5.10)")
    current = partition_field(handle)
    current === nothing ||
        invalid_option(:partitions, "this project is already partitioned by $(repr(current)); a project has " *
              "one partition field, and it cannot be changed")
    _check_partition_field(handle.meta_schema, field)
    graph_min = _check_partition_graph_min(kind, by)
    handle.wrote[] = true
    parts = Persistence.open_partition_store(handle.project.db, String(field), handle.postings_cache...)
    kind === :dense && return _declare_dense_partition!(handle, parts, String(field), graph_min)
    IndexEngine.write_lock(engine.lock) do
        n = length(engine.staged)
        values = Vector{Vector{String}}(undef, n)
        for _id in 1:n
            meta = get_meta(handle.project, _id; lazy=false)
            values[_id] = meta === nothing ? String[] : _partition_values(String(field), meta)
        end
        Persistence.clear_partitions!(parts)
        members = Persistence.assign_partitions!(parts, 1, values)
        if engine.backend.index !== nothing && length(engine.backend.index) > 0
            Persistence.split_partition_postings!(parts, _invfile_store(handle.project), members,
                                                  DECLARE_PARTITION_FLUSH_IDS)
        end
        Persistence.save_field!(handle.store, :partition_by, String(field))
        handle.partitions[] = parts
    end
    return partitions(handle)
end

"""
    DECLARE_PARTITION_FLUSH_IDS

How many posting ids [`declare_partition!`](@ref) accumulates in one `WriteBatch` before writing
it. Bounds the memory of the batch; the value only changes how many writes the split takes.
"""
const DECLARE_PARTITION_FLUSH_IDS = 4_000_000

"""
    _declare_dense_partition!(handle, parts, field, graph_min) -> Vector{NamedTuple}

The dense branch of [`declare_partition!`](@ref): records the partitions of every item, adds
the indexed ones to their partitions, and builds the graphs of the partitions that reach
`graph_min`. The field is recorded last, so an interrupted call leaves the project
unpartitioned on the next open.
"""
function _declare_dense_partition!(handle::EmbeddedEngine, parts::Persistence.PartitionStore, field::String, graph_min::Int)
    engine = handle.engine
    n = _current_size(engine)
    values = Vector{Vector{String}}(undef, n)
    for _id in 1:n
        meta = get_meta(handle.project, _id; lazy=false)
        values[_id] = meta === nothing ? String[] : _partition_values(field, meta)
    end
    Persistence.clear_partitions!(parts)
    Persistence.save_fields!(handle.store, (partition_graph_len=Dict{UInt32,Int}(),
                                            partition_search_state=Dict{UInt32,NamedTuple}(),
                                            partition_opt_beamsearch=Dict{UInt32,IndexEngine.OptBeamSearch}(),
                                            partition_graph_min=graph_min))
    members = Persistence.assign_partitions!(parts, 1, values)
    handle.partitions[] = parts
    _enable_dense_partitions!(handle, graph_min, Dict{UInt32,Int}())
    IndexEngine.add_partition_members!(engine, 1, members[1:length(engine.backend.index)])
    if engine isa IndexEngine.DenseEngine{IndexEngine.GraphBackend}
        IndexEngine.index_partition_graphs!(engine)
        _save_partition_state!(handle)
    end
    Persistence.save_field!(handle.store, :partition_by, field)
    return partitions(handle)
end

"""
    _partition_store(handle, operation::Symbol) -> Persistence.PartitionStore

The project's partitions, or `InvalidOption` naming `operation` when it declares no partition
field.
"""
function _partition_store(handle::EmbeddedEngine, operation::Symbol)
    parts = handle.partitions[]
    parts === nothing &&
        invalid_option(operation === :partitions ? :partition_by : :partition,
              "this project has no partition field, so `$operation` does not apply. Create it with " *
              "`partition_by`, or call declare_partition!(handle, field).")
    parts
end

"""
    _partition_adj(handle, name) -> Function

The `vocsize -> PartitionPostings` that [`IndexEngine.search_partition_live`](@ref) builds the
partition's index around. Raises `UnknownPartition` for a name the table does not hold.
"""
function _partition_adj(handle::EmbeddedEngine, name::AbstractString)
    parts = _partition_store(handle, :partition)
    number = _partition_number(handle, name)
    return vocsize -> Persistence.PartitionPostings(parts, number, vocsize)
end

"""
    _partition_number(handle, name) -> UInt32

The number of partition `name`. Raises `UnknownPartition` for a name the table does not hold,
with the closest names, and `InvalidOption` on a project without a partition field.
"""
function _partition_number(handle::EmbeddedEngine, name::AbstractString)
    parts = _partition_store(handle, :partition)
    number = Persistence.partition_number(parts, name)
    if number === nothing
        names = Persistence.partition_names(parts)
        nearest = _closest_names(names, String(name), 5)
        unknown_partition(name, "this project has no partition $(repr(String(name))) in its field " *
              "$(repr(parts.field)) ($(length(names)) partitions)" *
              (isempty(nearest) ? "" : "; the closest names are $(join(repr.(nearest), ", "))"))
    end
    return number
end

"""
    _check_partition_field(schema::Schema.MetaSchema, field)

Refuses an empty field name, and a field the schema declares with a type other than
`:string`, with `InvalidOption`.
"""
function _check_partition_field(schema::Schema.MetaSchema, field::AbstractString)
    isempty(field) && invalid_option(:partition_by, "the partition field needs a name")
    t = Schema.declared_type(schema, String(field))
    t === nothing || t === :string ||
        invalid_option(:partition_by, "the meta field $(repr(String(field))) is declared $(repr(t)); a partition " *
              "field holds partition names, so it must be undeclared or declared :string")
    return nothing
end

"""
    _partition_values(field::String, meta) -> Vector{String}

The partition names an item's `meta` gives it: none when `field` is absent or `nothing`, one for
a string, and each distinct one for a list of strings. Anything else, and an empty string,
raises `PayloadMismatch`: an item whose partition cannot be read is refused, not stored in no
partition.
"""
function _partition_values(field::String, meta)
    value = get(meta, field, nothing)
    value === nothing && return String[]
    value isa AbstractString && return [_partition_name(field, value)]
    value isa AbstractVector && return unique!(String[_partition_name(field, v) for v in value])
    _partition_name(field, value)
end

function _partition_name(field::String, value)
    value isa AbstractString && !isempty(value) && return String(value)
    payload_mismatch("meta field $(repr(field)) is this project's partition field: its value is a " *
                     "non-empty string or a list of them; got $(repr(value))::$(typeof(value))")
end

"""
    _closest_names(names, name::String, n::Int) -> Vector{String}

Up to `n` of `names`, the closest to `name` by edit distance, for the message of an unknown
partition. Ties are broken by name.
"""
function _closest_names(names::AbstractVector{String}, name::String, n::Int)
    scored = sort!([(_edit_distance(lowercase(c), lowercase(name)), c) for c in names])
    String[c for (_, c) in first(scored, n)]
end

"Levenshtein distance between two strings, counted in characters."
function _edit_distance(a::AbstractString, b::AbstractString)
    x, y = collect(a), collect(b)
    prev = collect(0:length(y))
    for i in eachindex(x)
        cur = similar(prev)
        cur[1] = i
        for j in eachindex(y)
            cur[j+1] = min(prev[j+1] + 1, cur[j] + 1, prev[j] + (x[i] == y[j] ? 0 : 1))
        end
        prev = cur
    end
    prev[end]
end

"""
    _as_meta_schema(spec) -> Schema.MetaSchema

The declaration a caller wrote, as the type the project stores. A `MetaSchema` passes
through; `nothing` is the empty declaration; anything else is read as a list of fields, each
one a `name => type` pair or a `(name, type)` tuple:

```julia
create_project(w, ds; meta_schema = ["year" => :int64, "lang" => :string])
```
"""
_as_meta_schema(schema::Schema.MetaSchema) = schema
_as_meta_schema(::Nothing) = Schema.MetaSchema()

function _as_meta_schema(fields)
    parsed = Schema.MetaField[]
    for f in fields
        if f isa Schema.MetaField
            push!(parsed, f)
        elseif f isa Pair
            push!(parsed, Schema.MetaField(String(first(f)), Symbol(last(f))))
        elseif f isa Tuple && length(f) == 2
            push!(parsed, Schema.MetaField(String(f[1]), Symbol(f[2])))
        else
            invalid_option(:meta_schema, "a declared field is `name => type` or `(name, type)`; got $(repr(f))")
        end
    end
    names = [f.name for f in parsed]
    length(unique(names)) == length(names) ||
        invalid_option(:meta_schema, "a field is declared more than once: $(join(unique([n for n in names if count(==(n), names) > 1]), ", "))")
    return Schema.MetaSchema(parsed)
end

"""
    meta_schema(handle::EmbeddedEngine) -> Schema.MetaSchema

What this project declares about the `meta` of its items (PLAN.md §4.5). Empty for a project
that declares nothing, which is the default and the behavior every project had before this
existed: `meta` is stored as it arrives.

A declared field is checked and converted when an item is appended, so its value is the type
it claims. Declaring is additive: an undeclared field is still accepted and stored as it
arrives. See [`declare_meta_schema!`](@ref) to declare on a project that already exists, and
[`Schema.MetaSchema`](@ref) for what a declaration is.
"""
meta_schema(handle::EmbeddedEngine) = (_check_open(handle); handle.meta_schema)

"""
    declare_meta_schema!(handle::EmbeddedEngine, fields) -> Schema.MetaSchema

Declares what `meta` holds from now on, on a project that already exists, and returns the
declaration. `fields` takes the same forms as `create_project`'s `meta_schema` keyword.

**Records already written are not read and not checked.** The declaration governs what is
appended after it, which is what makes declaring cheap on a project of any size. A field
declared now may therefore hold values of another type in older records; a `rebuild` is what
rewrites them.

The version of the declaration increases with each call, so `MetadataRecord.schema_version`
of a record says which declaration was in force when it was written.
"""
function declare_meta_schema!(handle::EmbeddedEngine, fields)
    _check_open(handle)
    handle.read_only && unsupported_operation(:declare_meta_schema!, "this project is open read-only")
    declared = _as_meta_schema(fields)
    schema = Schema.MetaSchema(handle.meta_schema.version + 1, declared.fields)
    Persistence.save_field!(handle.store, :meta_schema, schema)
    handle.meta_schema = schema
    handle.wrote[] = true
    return schema
end

"""
    text_profile(handle::EmbeddedEngine) -> Union{Nothing, TextSearch.TextProfile}

The text model this project searches with: the one handed to [`create_project`](@ref), or the
one its first [`index!`](@ref index!(::EmbeddedEngine)) call fitted from the staged corpus.
`nothing` for a dense project, and for a text project that has not been indexed yet.

Worth reaching for after a `FitFromCorpus`: `save_profile(dir, text_profile(h))` writes the
fitted vocabulary and weights out as a portable profile, so a sibling project can be created
against the same model (`TextDocuments(BaseProfile(load_profile(dir)))`) instead of fitting its own
and ending up with a different vocabulary over the same language.

Written as an explicit extension of `IndexEngine.text_profile` rather than as a bare
`text_profile(handle::EmbeddedEngine) = ...`. The bare form compiles -- `using .IndexEngine`
makes the name available but not yet resolved in this scope, so Julia quietly creates a
*second*, unrelated function shadowing the first -- and that is the hazard, not a convenience:
the two would then be one name with two disjoint method tables, and which one a call site got
would depend on whether it had already touched the imported binding. Extending keeps a single
generic function whose methods cover an engine and a handle alike (the same reasoning as the
`open_project` note in `SimilaritySearchEngine.jl`).
"""
IndexEngine.text_profile(handle::EmbeddedEngine) = (_check_open(handle); IndexEngine.text_profile(handle.engine))

"""
    _count_oov!(handle::EmbeddedEngine, text)

Adds one query to [`OOVCounters`](@ref): its tokens, and how many of them the vocabulary
does not contain as typed. `text` is a query string or a list of tokens; a list is counted
without tokenizing it.

The query is tokenized once more here, with the vocabulary's own `TextConfig`, rather than
read out of the search that just ran: the library resolves and encodes a query inside
`search`, and none of that resolution comes back. Tokenizing a query string costs a few
microseconds against a search that walks posting lists, and the alternative is for this
module to assemble the query pipeline itself again, which is the arrangement
[`IndexEngine.resolve_query`](@ref)'s docstring records as removed.

Counts nothing for a project whose vocabulary has not been fitted, and never raises: a
failure to measure must not fail the search that was measured.
"""
function _count_oov!(handle::EmbeddedEngine, text::Union{AbstractString,AbstractVector{<:AbstractString}})
    voc = IndexEngine.text_vocabulary(handle.engine)
    voc === nothing && return nothing
    counters = handle.oov
    try
        tokens = 0
        oov = 0
        # A list of tokens is counted as it is, the way `ftsearch` searches it.
        for tok in (text isa AbstractString ? TextSearch.tokenize(voc.textconfig, text) : text)
            tokens += 1
            TextSearch.token2id(voc, tok) == 0 && (oov += 1)
        end
        counters.queries += 1
        counters.tokens += tokens
        counters.oov += oov
        counters.unflushed += 1
    catch
        return nothing
    end
    if counters.unflushed >= OOV_FLUSH_EVERY && !handle.read_only
        counters.unflushed = 0
        Persistence.save_field!(handle.store, :oov_counters, counters)
    end
    return nothing
end

"""
    vocabulary_report(handle::EmbeddedEngine; scan::Bool=false) -> Union{Nothing, Dict{String,Any}}

What this project's vocabulary covers, and what it does not (PLAN.md §5).

Reports the size of the vocabulary, how many documents it was fitted from, its ten most
frequent tokens, and the running counters of [`OOVCounters`](@ref): how many queries were
resolved against it, how many tokens they carried, and what fraction of those the vocabulary
did not have.

That fraction is the drift signal. A vocabulary is fitted once, from the documents staged
when `index!` first ran, and a project whose subject matter moves ends up unable to search
for words it now holds. `rebuild` refits the vocabulary over the documents that are there
now, and these counters are what say whether that is worth doing.

`scan=true` adds `live_oov_rate`, the same fraction measured over the stored documents
instead of over the queries: a document is read and tokenized again, and its tokens are
looked up. The two answer different questions. The query rate says the words being asked for
are missing, which a rebuild fixes only if those words are in the documents; the document
rate says the words already stored are missing from the vocabulary, which is exactly what a
rebuild fixes.

`sample` bounds that pass. The cost of reading every document is linear in the size of the
project -- 11.6 s over 12.3 million tokens, measured 2026-09-14 -- and unbounded, so at most
`sample` documents are read, spread evenly over the project by taking every nth one. Both
ends of the project are therefore represented, which matters because drift accumulates at the
end: the documents appended after the vocabulary was fitted are the ones it does not cover.
The report says how many documents were read (`live_documents_scanned`) out of how many exist
(`live_documents_total`) and whether it read all of them (`live_sampled`). `sample <= 0`
reads every document, which is what an offline command such as `describe` asks for.

Runs under the project's read lock, like a search: the documents it reads are the ones an
append is writing.

Returns `nothing` for a project that does not hold text, and for one whose vocabulary has
not been fitted yet.
"""
function vocabulary_report(handle::EmbeddedEngine; scan::Bool=false, sample::Int=DEFAULT_OOV_SCAN_SAMPLE)
    _check_open(handle)
    engine = handle.engine
    IndexEngine.payload_kind(engine) === :text || return nothing
    voc = IndexEngine.text_vocabulary(engine)
    voc === nothing && return nothing

    top_n = min(10, length(voc.token))
    order = top_n == 0 ? Int[] : partialsortperm(voc.ndocs, 1:top_n, rev=true)
    counters = handle.oov

    report = Dict{String,Any}(
        "vocsize" => Int(TextSearch.vocsize(voc)),
        "trainsize" => Int(TextSearch.gettrainsize(voc)),
        "numtokens" => Int(TextSearch.getnumtokens(voc)),
        "avgdoclen" => Float64(TextSearch.avgdoclen(voc)),
        "top_tokens" => [Dict("token" => voc.token[i], "ndocs" => Int(voc.ndocs[i]),
                              "occs" => Int(voc.occs[i])) for i in order],
        "queries" => counters.queries,
        "query_tokens" => counters.tokens,
        "query_oov_tokens" => counters.oov,
        "query_oov_rate" => counters.tokens == 0 ? 0.0 : counters.oov / counters.tokens,
    )
    scan || return report

    oov = 0
    total = 0
    scanned = 0
    live = 0
    IndexEngine.read_lock(engine.lock) do
        ids = [id for id in 1:length(engine.staged) if !(id in engine.deleted_ids)]
        live = length(ids)
        step = sample > 0 && live > sample ? cld(live, sample) : 1
        for id in ids[1:step:end]
            text = IndexEngine.stored_payload(engine, id)
            text === nothing && continue
            scanned += 1
            for tok in TextSearch.tokenize(voc.textconfig, text)
                total += 1
                TextSearch.token2id(voc, tok) == 0 && (oov += 1)
            end
        end
    end
    report["live_tokens"] = total
    report["live_oov_tokens"] = oov
    report["live_oov_rate"] = total == 0 ? 0.0 : oov / total
    report["live_documents_scanned"] = scanned
    report["live_documents_total"] = live
    report["live_sampled"] = scanned < live
    return report
end

"""
    ftexplain(handle::EmbeddedEngine, text; policy=QueryPolicy()) -> Vector{String}

One human-readable line per query token that [`ftsearch`](@ref) would search as something
other than what was typed -- `"musica appears in only 9 documents, searched as música
instead"` -- and an empty vector when the query is searched verbatim.

A search that silently substitutes what was asked for owes the person a way to see it: this
is the "showing results for ..." half, and `QueryPolicy(correction=:off)` is the "search
instead for ..." half. Resolves the query the same way `ftsearch` does under the same
`policy`, so the two never disagree; it does not run the search itself.

Errors for a project whose text engine has not been trained yet, and for a dense project
(which has no query text to resolve).
"""
function ftexplain(handle::EmbeddedEngine, text::AbstractString; policy::QueryPolicy=QueryPolicy())
    _check_open(handle)
    engine = handle.engine
    IndexEngine.payload_kind(engine) === :text ||
        unsupported_operation(:ftexplain, "ftexplain is only meaningful for a text project (BM25InvertedFile/InvertedFile); this one is $(typeof(engine))")
    return TextSearch.explain(IndexEngine.resolve_query(engine, text, policy).resolution)
end

"""
    delete_item!(handle::EmbeddedEngine, _id::Integer)

Soft-deletes `_id` (future searches exclude it, the underlying index is untouched) and
immediately persists just the `deleted_ids` field -- mirrors `Server.handle_delete_item`.
"""
function delete_item!(handle::EmbeddedEngine, _id::Integer)
    _check_open(handle)
    handle.wrote[] = true
    IndexEngine.mark_deleted!(handle.engine, _id)
    Persistence.save_field!(handle.store, :deleted_ids, handle.engine.deleted_ids)
    return nothing
end

"""
    fetch_items(handle::EmbeddedEngine, ids) -> Vector{Schema.StoredItem}

Batch retrieval by id -- each element of `ids` may be the internal `_id` or a `doc_id` (see
[`_resolve_records`](@ref)). An id that resolves to nothing is skipped and a `doc_id` shared
by several items brings back all of them, so the result can be shorter *or longer* than
`ids`: it is a flat vector of whatever matched, in the order `ids` gave, ties broken by
ascending `_id`.

Each item comes back whole: the record's fixed fields, its `meta` as a plain
`Dict{String,Any}`, and `payload` -- the very text or vector that was indexed, read from the
engine via [`IndexEngine.stored_payload`](@ref).

Both halves of that are changes from what this used to return. It used to hand back a
`Dict{String,Any}` per item with the record's fields merged into the decoded `meta`, which
meant a `meta` key called `"doc_id"` silently overwrote the record's own, and it could not
return the payload at all: `"text"` was a reserved key stripped on the way in, so a text
project's fetch answered with everything about a paragraph except the paragraph.
"""
function fetch_items(handle::EmbeddedEngine, ids)
    _check_open(handle)
    project = handle.project
    results = Schema.StoredItem[]
    for raw_id in ids
        for record in _resolve_records(project, raw_id)
            meta = something(get_meta(project, record._id; lazy=false), Dict{String,Any}())
            push!(results, Schema.StoredItem(record, _stored_payload(handle, record._id), meta))
        end
    end
    return results
end

# The payload `fetch_items` returns. A quantized project returns the vector it kept in its
# dense-vector file, or `nothing`: its index holds codes, not the vectors that were appended.
function _stored_payload(handle::EmbeddedEngine, id::Integer)
    IndexEngine.isquantized(handle.engine) || return IndexEngine.stored_payload(handle.engine, id)
    raw = handle.dense_vectors[]
    raw === nothing && return nothing
    row = id - Persistence.load_field(handle.store, :raw_offset, 0)
    1 <= row <= length(raw) || return nothing
    return Vector{Float32}(raw[row])
end

"""
    exists(handle::EmbeddedEngine, ids) -> Vector{ExistsResult}

For each id in `ids` (internal `_id` or a `doc_id`, see [`_resolve_records`](@ref)), reports
whether a record exists and, if so, whether it's been soft-deleted. One result per *queried
id*, in order, including the ones that were not found -- unlike [`fetch_items`](@ref), which
drops those and returns one entry per matching record.

Since a `doc_id` may be carried by several items, the two flags summarize them: `exists` is
true when at least one record matched, and `deleted` is true only when *every* match is
soft-deleted, i.e. when asking for that id leaves nothing live to retrieve.
"""
function exists(handle::EmbeddedEngine, ids)
    _check_open(handle)
    project = handle.project
    engine = handle.engine
    results = ExistsResult[]
    for raw_id in ids
        records = _resolve_records(project, raw_id)
        found = !isempty(records)
        deleted = found && all(r -> r._id in engine.deleted_ids, records)
        push!(results, ExistsResult(string(raw_id), found, deleted))
    end
    return results
end

"""
    calibrate!(handle::EmbeddedEngine; levels=IndexEngine.DEFAULT_MINRECALL_LEVELS, numqueries=64,
               ksearch=10, queries=nothing, tradeoff=1.5, transition=(-1, 1), nboot=1000,
               confidence=0.95, rng=Random.default_rng(), partition=nothing) -> Vector{CalibrationLevel}

Tunes the project's `BeamSearch` for each recall level in `levels`, and measures the recall
each chosen configuration reaches, with an interval (see `IndexEngine.calibrate!` for the goal,
the query samples and each keyword). Returns one [`CalibrationLevel`](@ref) per level.

For example, over the 724 vectors of 128 dimensions in `test/data/frankenstein.jsonl`, measured
on 2026-11-07 with 64 threads and SimilaritySearch 1.6.11:

```julia
julia> calibrate!(h; levels=[0.9, 0.95], rng=Xoshiro(1))
2-element Vector{CalibrationLevel}:
 CalibrationLevel(minrecall=0.9, recall=0.9323, 95.0% [0.9062, 0.9549], 64 held-out queries, 317.2 evaluations/query)
 CalibrationLevel(minrecall=0.95, recall=0.9687, 95.0% [0.9531, 0.9827], 64 held-out queries, 379.2 evaluations/query)
```

The goal accepts a recall a little below the target when it saves enough cost (see
`IndexEngine.calibrate!`), so a level can end below its target. The construction of the graph is
not deterministic, so the result changes from one build to the next. Over 7 builds of the same
collection, the first level reached a recall from 0.920 to 0.950. The second level reached a
recall from 0.951 to 0.993. None of the 7 builds ended below its target.

The table of configurations, one per level, is persisted at once, so that `search`'s
`minrecall` keyword finds it after a `close_project!` and `open_project`. The measurements are
returned and not persisted.

`partition="name"` calibrates the graph of one partition of a dense project (PLAN.md §5.10),
and keeps its levels apart from the global ones. A partition is calibrated only by this call:
a search with `minrecall` on an uncalibrated partition raises `NotTrained`. A partition without
its own graph raises `InvalidOption`, because an exact scan has nothing to calibrate.
"""
function calibrate!(handle::EmbeddedEngine; levels=IndexEngine.DEFAULT_MINRECALL_LEVELS, numqueries::Int=64,
                    ksearch::Int=10, queries=nothing, tradeoff::Real=1.5, transition=(-1, 1),
                    nboot::Int=1000, confidence::Real=0.95, rng::AbstractRNG=Random.default_rng(),
                    partition::Union{Nothing,AbstractString}=nothing)
    _check_open(handle)
    handle.wrote[] = true
    # The queries of a quantized project are evaluated against codes, as in `search`.
    if queries !== nothing && IndexEngine.isquantized(handle.engine)
        queries = [_search_query(handle.engine, q) for q in queries]
    end
    if partition !== nothing
        handle.engine isa IndexEngine.DenseEngine{IndexEngine.GraphBackend} ||
            invalid_option(:partition, "calibrate! with a partition needs a dense project on a SearchGraph backend")
        number = _partition_number(handle, partition)
        report = IndexEngine.calibrate!(handle.engine; levels, numqueries, ksearch, queries, tradeoff,
                                        transition, nboot, confidence, rng, partition=number)
        _save_partition_state!(handle)
        return report
    end
    report = IndexEngine.calibrate!(handle.engine; levels, numqueries, ksearch, queries, tradeoff,
                                    transition, nboot, confidence, rng)
    Persistence.save_field!(handle.store, :opt_beamsearch, handle.engine.backend.opt_beamsearch)
    _save_search_state!(handle)
    return report
end

"""
    allknn(handle::EmbeddedEngine; k=10) -> Vector{KnnRow}

Runs `SimilaritySearch.allknn` synchronously against the project's dense index -- no
Job/spool machinery at all, unlike the HTTP API's `POST /api/v1/jobs/allknn` (PLAN.md
§8.5's "runs in-process" framing for the embedded surface). Errors if the project is a text
index or empty (mirrors `cli_handlers.jl`'s `_require_dense`). Returns one [`KnnRow`](@ref)
per item, its `_id` being the item's 1-based position in insertion order (matching
`execute_allknn`'s own output convention), not its `doc_id`.

A row can be shorter than `k`: the library pads unfilled neighbour slots with id `0`, and this
stops at the first one rather than reporting a neighbour that does not exist.
"""
function allknn(handle::EmbeddedEngine; k::Int=10)
    _check_open(handle)
    engine = handle.engine
    IndexEngine.payload_kind(engine) === :dense ||
        unsupported_operation(:allknn, "allknn requires a dense (vector) project; this one indexes $(IndexEngine.payload_kind(engine))")
    length(engine.backend.index) == 0 && empty_project(:allknn, "allknn requires a non-empty dense index")

    ids, dists = IndexEngine.allknn_live(engine, k)
    n = size(ids, 2)
    results = KnnRow[]
    for i in 1:n
        neighbor_ids = Int32[]
        neighbor_dists = Float32[]
        for j in 1:k
            ids[j, i] == 0 && break
            push!(neighbor_ids, Int32(ids[j, i]))
            push!(neighbor_dists, Float32(dists[j, i]))
        end
        push!(results, KnnRow(Int32(i), neighbor_ids, neighbor_dists))
    end
    return results
end

"""
    fft(handle::EmbeddedEngine, k::Integer; start::Int=0, verbose::Bool=false) -> FFTResult

Runs `SimilaritySearch.fft`'s Farthest-First-Traversal synchronously against the
project's dense index -- picks `k` well-separated items (e.g. as diverse candidate
cluster centers), the same "runs in-process, no job queue" framing as [`allknn`](@ref).
Errors if the project is a text index or empty (same guard as `allknn`). Unlike
`allknn`/`closestpairs`/`bichromatic_kclosestpairs`, this is a brute-force sweep over the
raw vectors (`SimilaritySearch.distance`/`database` of the engine's index) rather than a
graph search, so it works the same regardless of how well-tuned (or untuned) the
project's `BeamSearch` is.

Returns an `FFTResult` (an alias of [`CenterSelectionResult`](@ref)) -- every field `SimilaritySearch.fft` itself reports (see its
docstring and `SimilaritySearch.CenterSelection`), with ids as `Int32` to match every other id
this package hands back. `centers` are internal `_id`s (1-based position in the index), not
hydrated with metadata -- look them up yourself (e.g. via [`fetch_items`](@ref)) if you need
it. `assign[i]` is a position in `centers`, so item `i`'s center is `centers[assign[i]]`.
"""
function fft(handle::EmbeddedEngine, k::Integer; start::Int=0, verbose::Bool=false)
    _check_open(handle)
    engine = handle.engine
    IndexEngine.payload_kind(engine) === :dense ||
        unsupported_operation(:fft, "fft requires a dense (vector) project; this one indexes $(IndexEngine.payload_kind(engine))")
    length(engine.backend.index) == 0 && empty_project(:fft, "fft requires a non-empty dense index")
    r = IndexEngine.fft_live(engine, k, start, verbose)
    CenterSelectionResult(Int32.(r.centers), Int32.(r.assign), Float32.(r.assigndist),
                          Float32(r.covering), Float32(r.separation),
                          Int(r.costdists), Int(r.costblocks))
end

"""
    dnet(handle::EmbeddedEngine, k::Integer; verbose::Bool=false) -> CenterSelectionResult

Runs `SimilaritySearch.dnet` (density-based ball partitioning) synchronously against the
project's dense index -- selects representative centers by carving the database into balls of
approximately `length(db) ÷ k` elements. Errors if the project is not a dense index or is empty.

Returns a [`CenterSelectionResult`](@ref).
"""
function dnet(handle::EmbeddedEngine, k::Integer; verbose::Bool=false)
    _check_open(handle)
    engine = handle.engine
    IndexEngine.payload_kind(engine) === :dense ||
        unsupported_operation(:dnet, "dnet requires a dense (vector) project; this one indexes $(IndexEngine.payload_kind(engine))")
    length(engine.backend.index) == 0 && empty_project(:dnet, "dnet requires a non-empty dense index")
    r = IndexEngine.dnet_live(engine, k, verbose)
    CenterSelectionResult(Int32.(r.centers), Int32.(r.assign), Float32.(r.assigndist),
                          Float32(r.covering), Float32(r.separation),
                          Int(r.costdists), Int(r.costblocks))
end

"""
    neardup(handle::EmbeddedEngine, epsilon::Real; verbose::Bool=false, recall::Real=1.0) -> NearDupResult

Runs `SimilaritySearch.neardup` synchronously against the project's dense index --
finds the ``ϵ``-net of non-duplicate objects whose mutual distances are at least `epsilon`.
Objects closer than `epsilon` to an existing center are considered near-duplicates and assigned to it.
Errors if the project is not a dense index or is empty.

`epsilon` is a distance, on the scale of the project's distance function, and it must be zero
or positive. A negative or `NaN` value raises `InvalidOption`. Since SimilaritySearch 1.6, a
value below `1f-5` is raised to `1f-5` (`SimilaritySearch.NEARDUP_NUMERICAL_ZERO`): the distance
between two identical vectors is often not exactly `0f0` after rounding, and `epsilon=0` would
then miss some exact duplicates. So `epsilon=0` finds exact duplicates, including the ones
whose distance rounds to a small positive value.

Returns a [`NearDupResult`](@ref).
"""
function neardup(handle::EmbeddedEngine, epsilon::Real; verbose::Bool=false, recall::Real=1.0)
    _check_open(handle)
    engine = handle.engine
    epsilon >= 0 ||
        invalid_option(:epsilon, "neardup needs epsilon >= 0, a distance on the scale of the project's distance function; got $epsilon")
    IndexEngine.payload_kind(engine) === :dense ||
        unsupported_operation(:neardup, "neardup requires a dense (vector) project; this one indexes $(IndexEngine.payload_kind(engine))")
    length(engine.backend.index) == 0 && empty_project(:neardup, "neardup requires a non-empty dense index")
    r = IndexEngine.neardup_live(engine, Float32(epsilon), verbose, Float32(recall))
    NearDupResult(Int32.(r.centers), Int32.(r.assign), Float32.(r.assigndist),
                  Float32(r.covering), Float32(r.epsilon),
                  Int(r.costdists), Int(r.costblocks))
end

"""
    closestpairs(handle::EmbeddedEngine; k::Int=1, min_k::Int=max(k, 8)) -> Vector{Tuple{Int32,Int32,Float32}}

Runs `SimilaritySearch.closestpairs` synchronously against the project's dense index --
the `k` globally closest pairs among every item currently indexed (approximate if the
index itself is, e.g. an untuned `SearchGraph`), same in-process framing as
[`allknn`](@ref). Errors if the project is a text index or empty (same guard as
`allknn`).

Returns up to `k` `(i, j, dist)` tuples, ascending by distance -- `i`/`j` are internal
`_id`s, not hydrated with metadata.
"""
function closestpairs(handle::EmbeddedEngine; k::Int=1, min_k::Int=max(k, 8))
    _check_open(handle)
    engine = handle.engine
    IndexEngine.payload_kind(engine) === :dense ||
        unsupported_operation(:closestpairs, "closestpairs requires a dense (vector) project; this one indexes $(IndexEngine.payload_kind(engine))")
    length(engine.backend.index) == 0 && empty_project(:closestpairs, "closestpairs requires a non-empty dense index")
    IndexEngine.closestpairs_live(engine, k, min_k)
end

"""
    bichromatic_kclosestpairs(handle::EmbeddedEngine, B; k::Int=1, min_k::Int=max(k, 8)) -> Vector{Tuple{Int32,Int32,Float32}}

Runs `SimilaritySearch.bichromatic_kclosestpairs` synchronously between the project's own
dense index (playing the role of `idxA`, already indexed) and `B` -- a second, separate
collection of vectors (any `SimilaritySearch.AbstractDatabase`, e.g. a plain
`Vector{Vector{Float32}}` or `MatrixDatabase`) that is *not* itself indexed by this
project, or by anything else -- e.g. the same LSI space's vectors for a different half of
a corpus than the one this project indexes. Errors if the project is a text index or
empty (same guard as `allknn`).

There is deliberately no way to pass two `EmbeddedEngine` projects here: this embedded API
manages one project (one index) at a time, and `B` never needs its own project/persistence
of its own to take part in this -- it's queried directly, in memory, exactly as
`SimilaritySearch.bichromatic_kclosestpairs` itself expects.

Returns up to `k` `(i, j, dist)` tuples, ascending by distance -- `i` is this project's
own internal `_id` (in `idxA`), `j` is `B`'s own position (1-based, in `B`'s own order,
*not* related to any project's `_id` space) -- neither is hydrated with metadata.
"""
function bichromatic_kclosestpairs(handle::EmbeddedEngine, B; k::Int=1, min_k::Int=max(k, 8))
    _check_open(handle)
    engine = handle.engine
    IndexEngine.payload_kind(engine) === :dense ||
        unsupported_operation(:bichromatic_kclosestpairs, "bichromatic_kclosestpairs requires a dense (vector) project; this one indexes $(IndexEngine.payload_kind(engine))")
    length(engine.backend.index) == 0 && empty_project(:bichromatic_kclosestpairs, "bichromatic_kclosestpairs requires a non-empty dense index")
    IndexEngine.bichromatic_kclosestpairs_live(engine, B, k, min_k)
end
