```@meta
CurrentModule = SimilaritySearchEngine
DocTestSetup = quote
    using SimilaritySearchEngine
end
```

# SimilaritySearchEngine API

## Embedded API

The friendly, no-HTTP-required surface — project lifecycle, CRUD, calibration, `allknn`.

```@autodocs
Modules = [SimilaritySearchEngine]
Order   = [:type, :function, :constant]
```

## `Errors`

The typed failures every other module raises: four categories (`InvalidRequest`, `NotFound`,
`ConflictingState`, `StorageFailure`) and the concrete types beneath them. A consumer maps
categories to its own vocabulary -- status codes, exit codes -- rather than reading messages.

```@autodocs
Modules = [SimilaritySearchEngine.Errors]
Order   = [:type, :function, :constant]
```

## `IndexEngine`

The `AbstractSearchEngine` type hierarchy, insertion/search, calibration, and the
concurrency (`ReadWriteLock`/`ContextPool`) and logging (reporters and observers, see
`SimilaritySearch.jl`'s `log.jl`) machinery underneath the embedded API.

```@autodocs
Modules = [SimilaritySearchEngine.IndexEngine]
Order   = [:type, :function, :constant]
```

## `Persistence`

The RocksDB column-family-backed stores (`EngineStore`, `AdjacencyStore`,
`InvertedFileObjectStore`) each engine's `on_change` callback writes into.

```@autodocs
Modules = [SimilaritySearchEngine.Persistence]
Order   = [:type, :function, :constant]
```

## `Project`

Project-level metadata storage (`ProjectManager`), independent of any search engine.

```@autodocs
Modules = [SimilaritySearchEngine.Project]
Order   = [:type, :function, :constant]
```

## `Schema`

Typed, optionally-indexed metadata schema declarations.

```@autodocs
Modules = [SimilaritySearchEngine.Schema]
Order   = [:type, :function, :constant]
```
