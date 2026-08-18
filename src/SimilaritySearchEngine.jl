module SimilaritySearchEngine

using Avro
using JSON3
using SimilaritySearch
using TextSearch
using RocksDB

# Extracted verbatim from SimilaritySearchServer.jl (PLAN.md §8.5, chunk 22) -- these four
# modules were already the real shared engine both `similarity-search` (CLI) and
# `similarity-search-serve` (HTTP) call into; they never depended on anything else in that
# package (Jobs/Cursors/Tokens/Telemetry/Executors/Server are HTTP/CLI-specific glue that
# stayed behind). Schema before Project (Project needs it); IndexEngine/Persistence are
# independent of both and of each other.
include("schema.jl")
include("project.jl")
include("index_engine.jl")
include("persistence.jl")

using .Schema
using .Persistence
# open_project/close_project deliberately excluded: embedded.jl below defines its own
# EmbeddedEngine-returning open_project (identical (String, String) positional signature to
# Project.open_project's 2-arg form) and a distinctly-named close_project!. Bringing
# Project's raw open_project bare into this same scope would make embedded.jl's unqualified
# `function open_project(...)` silently extend/overwrite Project.open_project itself --
# corrupting the one every other qualified call site (`Project.open_project(...)` in
# cli_handlers.jl/server.jl) actually depends on, since Julia methods are owned by the
# generic function they extend, not by the module doing the extending. embedded.jl
# references `Project.open_project`/`Project.close_project` qualified instead.
using .Project: ProjectManager, put_metadata!, get_metadata, find_by_original_id, generate_id
using .IndexEngine

# The friendly, no-HTTP-required embedded API (PLAN.md §8.5) built on top of the four
# modules above -- a script does `using SimilaritySearchEngine` and calls these directly,
# with no Job/token/HTTP surface at all (the trust boundary here is OS file permissions on
# workdir, same as the CLI).
include("embedded.jl")

export EmbeddedEngine, create_project, open_project, close_project!, append_items!,
       search, ftsearch, delete_item!, fetch_items, exists, calibrate!, allknn

end # module
