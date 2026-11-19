"""
    SimilaritySearchClient

A client of the HTTP API of `similarity-search-serve` (PLAN.md §8.3).

The functions have the names, the arguments and the result types of the engine's own functions
(`create_project`, `append_items!`, `search`, `ftsearch`, `fetch_items`, `calibrate!`, ...).
The handle is a [`RemoteProject`](@ref) instead of an `EmbeddedEngine`, and it is opened with a
[`Connection`](@ref) instead of a directory. A script written for the engine runs against a
remote server when it opens its project differently, within the limits that each docstring
states.

The package depends on HTTP.jl and JSON3 only. It does not load the engine, SimilaritySearch,
TextSearch or RocksDB.
"""
module SimilaritySearchClient

using HTTP
using JSON3
using SparseArrays
using SparseArrays: SparseVector

export Connection, RemoteProject
export create_project, open_project, close_project!, delete_project!, list_projects, project_info
export append_items!, index!, search, ftsearch, fetch_items, exists, delete_item!, calibrate!, hybrid_search
export meta_schema, partition_field, partition_graph_min, partitions, vocabulary_report
export AbstractItem, DenseItem, SparseItem, TextItem
export SearchResult, StoredItem, ExistsResult, CalibrationLevel
export RemoteError, EngineError, InvalidRequest, NotFound, ConflictingState, StorageFailure, AccessDenied
export PayloadMismatch, WrongDimension, UnknownBackend, InvalidOption, UnsupportedOperation, RequestRejected
export ProfileNotInstalled, UnknownPartition, ResourceNotFound
export PendingBacklog, NothingStaged, NoIndex, NotTrained, EmptyProject, StateConflict
export CorruptedStorage, Unauthenticated, Forbidden, ServerFailure, UnexpectedResponse

include("errors.jl")
include("types.jl")
include("connection.jl")
include("projects.jl")
include("operations.jl")

end
