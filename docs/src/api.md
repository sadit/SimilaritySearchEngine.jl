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
Order   = [:type, :function]
```

## `IndexEngine`

The `AbstractSearchEngine` type hierarchy, insertion/search, calibration, and the
concurrency (`ReadWriteLock`/`ContextPool`) and logging (`CallbackLog`/`FileLog`)
machinery underneath the embedded API.

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
Order   = [:type, :function]
```

## `Schema`

Typed, optionally-indexed metadata schema declarations.

```@autodocs
Modules = [SimilaritySearchEngine.Schema]
Order   = [:type, :function]
```
