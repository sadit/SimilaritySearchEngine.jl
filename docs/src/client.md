```@meta
CurrentModule = SimilaritySearchClient
```

# SimilaritySearchClient API

The Julia client of the HTTP API, in `server/julia/` of this repository. Its functions have the
names, the arguments and the result types of the engine's functions, and they send each call to
a remote `similarity-search-serve` process. The
[Julia client page of the manual](https://sadit.github.io/SimilaritySearchEngine.jl/server/julia-client.html)
lists the differences with the engine.

```@autodocs
Modules = [SimilaritySearchClient]
Order   = [:module, :type, :function, :constant]
```
