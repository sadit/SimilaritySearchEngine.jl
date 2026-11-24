"""
    SimilaritySearchClient

A client of the HTTP API of `similarity-search-serve` (CHANGELOG.md §8.3).

The functions have the names, the arguments and the result types of the engine's own functions
(`create_project`, `append_items!`, `search`, `ftsearch`, `fetch_items`, `calibrate!`, ...).
The handle is a [`RemoteProject`](@ref) instead of an `EmbeddedEngine`, and it is opened with a
[`Connection`](@ref) instead of a directory. A script written for the engine runs against a
remote server when it opens its project differently, within the limits that each docstring
states.

The package depends on HTTP.jl and JSON.jl only. It does not load the engine, SimilaritySearch,
TextSearch or RocksDB.
"""
module SimilaritySearchClient

using HTTP
using JSON
using SparseArrays
using SparseArrays: SparseVector

export Connection, RemoteProject
export create_project, open_project, close_project!, delete_project!, list_projects, project_info
export append_items!, index!, search, ftsearch, fetch_items, exists, delete_item!, calibrate!, hybrid_search
export meta_schema, partition_field, partition_graph_min, partitions, vocabulary_report
export allknn, fft, neardup, dump_project, load_project
export submit_job, job_info, wait_job, job_result, list_jobs, cancel_job!, kill_job!, block_job!, resume_job!
export RemoteFollower, RemoteFollowerView, FollowerStatus
export create_follower, open_follower, close_follower!, followers, follower_status, sync!
export rebuild_follower!, delete_follower!, migrate_partitions!, posting_list, plugins
export KnnRow, CenterSelectionResult, FFTResult, NearDupResult, JobFailed, JobTimeout
export AbstractItem, DenseItem, SparseItem, TextItem
export SearchResult, StoredItem, ExistsResult, CalibrationLevel
export RemoteError, EngineError, InvalidRequest, NotFound, ConflictingState, StorageFailure, AccessDenied
export PayloadMismatch, WrongDimension, UnknownBackend, InvalidOption, UnsupportedOperation, RequestRejected
export ProfileNotInstalled, UnknownPartition, UnknownFollower, PluginNotLoaded, ResourceNotFound
export PendingBacklog, NothingStaged, NoIndex, NotTrained, EmptyProject, ProjectClosed, StaleFollower, HasFollowers, StateConflict
export CorruptedStorage, Unauthenticated, Forbidden, ServerFailure, UnexpectedResponse

include("errors.jl")
include("types.jl")
include("connection.jl")
include("projects.jl")
include("operations.jl")
include("jobs.jl")
include("followers.jl")

end
