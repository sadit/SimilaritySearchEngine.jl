# CLI argument parsing

"""
    CLI_CHOICES

Enumerated choice lists for arguments whose valid values are a closed set.
`ArgParse.jl` has no first-class "choices" field on `ArgParseField` -- only an opaque
`range_tester` validation closure -- so this is the one place those lists are declared,
used both to build each argument's `range_tester` below (so the non-interactive CLI now
rejects an invalid value immediately, rather than failing later inside the relevant
`execute_*` function or a library call) and by the interactive layer (`interactive.jl`) to
render a selection menu instead of free text for these fields, per PLAN.md's Interactive
Mode "single source of truth" principle -- applied here to the one piece of metadata
`ArgParseField` genuinely can't carry on its own.
"""
const CLI_CHOICES = Dict(
    "index-kind" => ["searchgraph", "exhaustive_search", "parallel_exhaustive_search", "invfile", "bm25_invfile"],
    "distance" => ["L2", "Cosine", "Angle", "NormalizedCosine"],
)

_choice_tester(name::String) = x -> x in CLI_CHOICES[name]

"""
    build_settings() -> ArgParse.ArgParseSettings

The `similarity-search` command line: every data-plane subcommand (`build`, `search`,
`searchbatch`, `allknn`, `fft`, `neardup`, `hsp`, `closestpair`, `describe`, `rebuild`,
`dump`, `load`) with its options. Built fresh on each call rather than held in a constant,
because `interactive.jl` introspects the same object to generate its guided forms.
"""
function build_settings()
    s = ArgParseSettings(description = "SimilaritySearchServer CLI")

    @add_arg_table! s begin
        "--config", "-c"
            help = "Path to the TOML configuration file"
            default = "config.toml"
        "interactive"
            help = "Start the interactive guided CLI"
            action = :command
        "build"
            help = "Build an index"
            action = :command
        "searchbatch"
            help = "Execute heavy offline batch queries"
            action = :command
        "allknn"
            help = "Compute all-vs-all k nearest neighbors for a dataset (requires a dense/vector index)"
            action = :command
        "fft"
            help = "Select k well-separated centers via Farthest First Traversal (index-free, operates on the raw dense database)"
            action = :command
        "neardup"
            help = "Find near-duplicate items (within epsilon of each other) in a dense dataset"
            action = :command
        "hsp"
            help = "Compute the Half-Space Proximal neighborhood of a query set against a dense dataset"
            action = :command
        "closestpair"
            help = "Find the closest pair of items in a dense dataset (requires a pre-built index)"
            action = :command
        "describe"
            help = "Describe a dataset and its index: size, distance stats, tombstone ratio, calibrated beamsearch baseline (dense), vocabulary/OOV stats (text)"
            action = :command
        "rebuild"
            help = "Rebuild a dataset's index from its current live (non-tombstoned) documents -- purges soft-deletes and, for text indices, retrains the vocabulary from scratch"
            action = :command
        "dump"
            help = "Export a dataset (index snapshot + metadata) as a portable bundle directory for backup/migration"
            action = :command
        "load"
            help = "Import a dataset from a bundle directory produced by dump"
            action = :command
    end

    @add_arg_table! s["build"] begin
        "--dataset"
            help = "Dataset ID/name"
            required = true
        "--input"
            help = "Input JSONL file"
            required = true
        "--index-kind"
            help = "Type of index to build (searchgraph, exhaustive_search, parallel_exhaustive_search, invfile, bm25_invfile)"
            default = "searchgraph"
            range_tester = _choice_tester("index-kind")
        "--distance"
            help = "Distance function (L2, Cosine, Angle, NormalizedCosine)"
            default = "L2"
            range_tester = _choice_tester("distance")
        "--workdir"
            help = "Working directory for the database"
            default = "data"
    end

    @add_arg_table! s["searchbatch"] begin
        "--dataset"
            help = "Dataset ID/name"
            required = true
        "--queries"
            help = "Queries file"
            required = true
        "--k"
            help = "Number of neighbors"
            arg_type = Int
            default = 10
        "--output"
            help = "Output file"
            required = true
        "--workdir"
            help = "Working directory for the database"
            default = "data"
    end

    @add_arg_table! s["allknn"] begin
        "--dataset"
            help = "Dataset ID/name"
            required = true
        "--k"
            help = "Number of neighbors per item"
            arg_type = Int
            default = 10
        "--output"
            help = "Output file (JSONL, one line per item)"
            required = true
        "--workdir"
            help = "Working directory for the database"
            default = "data"
    end

    @add_arg_table! s["fft"] begin
        "--dataset"
            help = "Dataset ID/name"
            required = true
        "--k"
            help = "Number of centers to select"
            arg_type = Int
            required = true
        "--output"
            help = "Output file (single JSON object)"
            required = true
        "--workdir"
            help = "Working directory for the database"
            default = "data"
    end

    @add_arg_table! s["neardup"] begin
        "--dataset"
            help = "Dataset ID/name"
            required = true
        "--epsilon"
            help = "Distance threshold below which an item is considered a duplicate of an already-kept one"
            arg_type = Float64
            required = true
        "--output"
            help = "Output file (single JSON object)"
            required = true
        "--workdir"
            help = "Working directory for the database"
            default = "data"
    end

    @add_arg_table! s["hsp"] begin
        "--dataset"
            help = "Dataset ID/name"
            required = true
        "--queries"
            help = "Queries file (JSONL, one 'vector' per line)"
            required = true
        "--k"
            help = "Number of candidate neighbors to fetch per query before HSP filtering"
            arg_type = Int
            default = 10
        "--output"
            help = "Output file (JSONL, one line per query)"
            required = true
        "--workdir"
            help = "Working directory for the database"
            default = "data"
    end

    @add_arg_table! s["closestpair"] begin
        "--dataset"
            help = "Dataset ID/name"
            required = true
        "--min-k"
            help = "Candidate neighborhood size used while searching for the closest pair (larger is more accurate for an approximate index, slower to compute)"
            arg_type = Int
            default = 8
        "--output"
            help = "Output file (single JSON object)"
            required = true
        "--workdir"
            help = "Working directory for the database"
            default = "data"
    end

    @add_arg_table! s["describe"] begin
        "--dataset"
            help = "Dataset ID/name"
            required = true
        "--output"
            help = "Output file (single JSON object). Prints to stdout if omitted"
            default = nothing
        "--workdir"
            help = "Working directory for the database"
            default = "data"
    end

    @add_arg_table! s["rebuild"] begin
        "--dataset"
            help = "Dataset ID/name"
            required = true
        "--workdir"
            help = "Working directory for the database"
            default = "data"
    end

    @add_arg_table! s["dump"] begin
        "--dataset"
            help = "Dataset ID/name"
            required = true
        "--output"
            help = "Output bundle directory (created fresh; must not already exist)"
            required = true
        "--workdir"
            help = "Working directory for the database"
            default = "data"
    end

    @add_arg_table! s["load"] begin
        "--bundle"
            help = "Bundle directory previously produced by dump"
            required = true
        "--dataset"
            help = "Target dataset ID/name (must not already exist under --workdir)"
            required = true
        "--workdir"
            help = "Working directory for the database"
            default = "data"
    end

    return s
end

"""
    parse_commandline(args::Vector{String}=ARGS) -> Dict

Parses `args` against a freshly built [`build_settings`](@ref) object. Split out from
`build_settings` so `interactive.jl` can introspect the same settings *structure* (same
subcommands, same per-subcommand fields/defaults/choices) via its own fresh call to
`build_settings()`, without needing the already-parsed result this function returns.
"""
function parse_commandline(args::Vector{String}=ARGS)
    return parse_args(args, build_settings())
end

"""
    build_serve_settings() -> ArgParseSettings

`similarity-search-serve`'s own settings object (PLAN.md §1, chunk 21) -- a genuinely
different, separate `ArgParseSettings` from `build_settings()`'s data-operations one, not
a shared parser both binaries happen to use. `serve` is a thin, single-purpose binary
(PLAN.md's own framing): it has exactly two commands, `serve` and `interactive`, and none
of `similarity-search`'s eleven data-operation subcommands (`build` through `load`) are
reachable through it at all -- `similarity-search-server build ...` is now an ArgParse
"unrecognized command" error, not a working (if undocumented) escape hatch, same as how
plain `similarity-search serve` is now the mirror-image error (`serve` isn't one of
`build_settings()`'s commands either). `--host`/`--port`/`--workdir` are all optional
(`default = nothing`) -- an omitted flag means "fall back to `--config`'s TOML file, then a
hardcoded literal," layered in `main_serve`/`run_serve`, not here (an `ArgParseField`
default can only be one fixed value, not "whatever the config file says"). This is also
what gives `run_interactive_serve` (`interactive.jl`) real fields to introspect at all --
PLAN.md §1's Interactive Mode section's own prerequisite for `serve`'s guided form.
"""
function build_serve_settings()
    s = ArgParseSettings(description = "SimilaritySearchServer HTTP API server")

    @add_arg_table! s begin
        "--config", "-c"
            help = "Path to the TOML configuration file"
            default = "config.toml"
        "serve"
            help = "Start the HTTP REST API server"
            action = :command
        "interactive"
            help = "Start the interactive guided CLI"
            action = :command
    end

    @add_arg_table! s["serve"] begin
        "--host"
            help = "Listen host (overrides --config's [server].host)"
            default = nothing
        "--port"
            help = "Listen port (overrides --config's [server].port)"
            arg_type = Int
            default = nothing
        "--workdir"
            help = "Working directory (overrides --config's [paths].workdir)"
            default = nothing
    end

    return s
end

"""
    parse_serve_commandline(args::Vector{String}=ARGS) -> Dict

`similarity-search-serve`'s counterpart to `parse_commandline` -- parses against
[`build_serve_settings`](@ref) instead of the data-operations `build_settings`.
"""
function parse_serve_commandline(args::Vector{String}=ARGS)
    return parse_args(args, build_serve_settings())
end
