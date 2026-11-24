# CLI argument parsing

"""
    CLI_CHOICES

Enumerated choice lists for arguments whose valid values are a closed set.
`ArgParse.jl` has no first-class "choices" field on `ArgParseField` -- only an opaque
`range_tester` validation closure -- so this is the one place those lists are declared,
used both to build each argument's `range_tester` below (so the non-interactive CLI now
rejects an invalid value immediately, rather than failing later inside the relevant
`execute_*` function or a library call) and by the interactive layer (`interactive.jl`) to
render a selection menu instead of free text for these fields, per CHANGELOG.md's Interactive
Mode "single source of truth" principle -- applied here to the one piece of metadata
`ArgParseField` genuinely can't carry on its own.
"""
const CLI_CHOICES = Dict(
    "index-kind" => ["searchgraph", "exhaustive_search", "parallel_exhaustive_search", "invfile", "bm25_invfile"],
    "distance" => ["L2", "SqL2", "Cosine", "NormCosine", "Angle", "NormAngle"],
    # A key `command:argument` applies to that argument of that command only. `follower-create`
    # has a `--distance` whose values are distances between sets, not the distances of a dense
    # dataset.
    "follower-create:distance" => ["jaccard", "dice", "intersection", "cosine_set"],
    "follower-create:policy" => ["on_index", "on_write", "manual", "periodic"],
    "migrate-partitions:disjoint" => ["true", "false"],
)

_choice_tester(name::String) = x -> x in CLI_CHOICES[name]

"""
    field_choices(cmd_name, dest_name) -> Union{Nothing, Vector{String}}

The values that the argument `dest_name` of the command `cmd_name` accepts, from
[`CLI_CHOICES`](@ref): the key `cmd_name:dest_name` first, then `dest_name`. `nothing` when the
argument takes any value.
"""
field_choices(cmd_name::AbstractString, dest_name::AbstractString) =
    get(CLI_CHOICES, "$cmd_name:$dest_name", get(CLI_CHOICES, dest_name, nothing))

"""
    add_follower_create_args!(table) -> table

Adds the arguments of `follower-create` to `table`, the settings of that command in
`similarity-search` or in `similarity-search-ctl` (CHANGELOG.md §5.12). Both commands send the same
request: [`follower_request`](@ref) turns the parsed arguments into it.
"""
function add_follower_create_args!(t::ArgParseSettings)
    @add_arg_table! t begin
        "--dataset"
            help = "The leader: the dataset whose items the follower reads"
            required = true
        "--name"
            help = "Name of the follower: letters, digits, '_' and '-', at most 128 characters"
            required = true
        "--kind"
            help = "partition (a search restricted to the items of one key), keyword (an inverted file of the keys), or Plugin.name, a kind of a plugin"
            required = true
            range_tester = x -> x in ("partition", "keyword") || occursin('.', x)
        "--source"
            help = "Where the keys of an item come from: keywords, refs, doc_id, meta:<field>, or Plugin.name, a key source of a plugin"
            required = true
        "--source-options"
            help = "The options of a key source of a plugin, as a JSON object, such as '{\"field\": \"year\"}'"
        "--kind-options"
            help = "The options of a kind of a plugin, as a JSON object"
        "--non-disjoint"
            help = "partition only: an item can have several keys and belong to several partitions"
            action = :store_true
        "--graph-min"
            help = "partition of a dense dataset only: the number of members from which a partition has its own graph"
            arg_type = Int
        "--distance"
            help = "keyword only: the distance between the keys of an item and the query (jaccard, dice, intersection, cosine_set; default jaccard)"
            range_tester = _choice_tester("follower-create:distance")
        "--policy"
            help = "When the follower processes new items: on_index, on_write, manual, or periodic (processed by a running server)"
            default = "on_index"
            range_tester = _choice_tester("follower-create:policy")
        "--periodic-seconds"
            help = "--policy periodic: process the new items every this many seconds"
            arg_type = Float64
        "--periodic-items"
            help = "--policy periodic: process the new items when this many are waiting"
            arg_type = Int
    end
    return t
end

# The arguments `--dataset` and `--name` of a command on one follower.
function _add_follower_name_args!(t::ArgParseSettings)
    @add_arg_table! t begin
        "--dataset"
            help = "The leader: the dataset that lists the follower"
            required = true
        "--name"
            help = "Name of the follower"
            required = true
    end
    return t
end

# The arguments of `migrate-partitions`, in both command lines.
function _add_migrate_partitions_args!(t::ArgParseSettings)
    @add_arg_table! t begin
        "--dataset"
            help = "A dataset created with --partition-by"
            required = true
        "--name"
            help = "Name of the new partition follower"
            required = true
        "--disjoint"
            help = "true when each item has one value of the partition field, false when an item can have several. It has no default: the command does not choose the kind of partition"
            required = true
            range_tester = _choice_tester("migrate-partitions:disjoint")
    end
    return t
end

"""
    build_settings() -> ArgParse.ArgParseSettings

The `similarity-search` command line: every data-plane subcommand (`build`, `search`,
`searchbatch`, `allknn`, `fft`, `neardup`, `hsp`, `closestpair`, `describe`, `rebuild`,
`dump`, `load`, the follower commands of CHANGELOG.md §5.12 and `postings`) with its options. Built fresh on each call rather than held in a constant,
because `interactive.jl` introspects the same object to generate its guided forms.
"""
function build_settings()
    s = ArgParseSettings(description = "SimilaritySearchServer CLI")

    @add_arg_table! s begin
        "--config", "-c"
            help = "Path to the TOML configuration file. The commands that open a follower load the plugins of its [plugins] section"
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
            help = "Export a dataset (its index and its metadata) as a portable bundle directory, for backup or migration"
            action = :command
        "load"
            help = "Import a dataset from a bundle directory produced by dump"
            action = :command
        "add-token"
            help = "Create an access token directly in the workdir, without a running server -- the way the first token is created when [auth] enabled is true"
            action = :command
        "follower-create"
            help = "Create a follower of a dataset (a partition or a keyword index of its items) and process every indexed item"
            action = :command
        "follower-list"
            help = "Show the status of every follower of a dataset"
            action = :command
        "follower-sync"
            help = "Process the items of the leader that a follower has not processed yet"
            action = :command
        "follower-rebuild"
            help = "Build a follower again from every indexed item of its leader (needed after the leader is rebuilt)"
            action = :command
        "follower-delete"
            help = "Delete a follower and remove it from the list of its leader"
            action = :command
        "migrate-partitions"
            help = "Move the partitions of a dataset created with --partition-by to a partition follower"
            action = :command
        "postings"
            help = "Print the ids of the items that hold one key: a token of a text dataset, a dimension of a sparse dataset, or a key of a keyword follower"
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
            help = "Distance function (L2, SqL2, Cosine, NormCosine, Angle, NormAngle); L2 is computed as SqL2"
            default = "L2"
            range_tester = _choice_tester("distance")
        "--workdir"
            help = "Working directory for the database"
            default = "data"
        "--partition-by"
            help = "Text or dense index: the meta field that partitions the dataset; a search can then be restricted to one value of it (searchbatch --partition)"
        "--partition-graph-min"
            help = "Dense index with --partition-by only: the number of indexed members from which a partition has its own graph (default 4096); a smaller partition is searched with an exact scan"
            arg_type = Int
        "--storage"
            help = "Dense index only: how the vectors are stored, a JSON object as the 'storage' field of POST /api/v1/datasets, for example '{\"kind\": \"quantized\", \"bits\": 4}' or '{\"kind\": \"sketched\", \"hyperplanes\": 256, \"bits\": 8}' (default: Float32 vectors)"
        "--max-k"
            help = "The largest k that a search of the dataset can ask for when a server serves it (default 100); POST /api/v1/admin/datasets/{id}/reload with {\"max_k\": n} changes it"
            arg_type = Int
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
        "--edit-correction"
            help = "Text dataset only: replace a query token absent from the vocabulary with the only vocabulary token at edit distance 1"
            action = :store_true
        "--partition"
            help = "Text or dense dataset: search one partition (a value of the dataset's partition field) instead of the whole dataset"
        "--follower"
            help = "Search through a follower: name=value searches one partition of a partition follower; the name of a keyword follower searches it with the 'keys' (and 't') of each query line"
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

    @add_arg_table! s["add-token"] begin
        "--user"
            help = "User or identifier the token belongs to"
            required = true
        "--permissions"
            help = "Comma-separated permissions, each `operation:dataset` with operation in read|write|admin and dataset an id or `*` (e.g. \"admin:*\" or \"read:*,write:corpus_es\")"
            required = true
        "--expires-at"
            help = "Optional ISO 8601 expiry, e.g. 2027-01-01T00:00:00"
        "--workdir"
            help = "Working directory holding the token database"
            default = "./workdir"
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
        "--rebuild-followers"
            help = "Also build every follower of the dataset again. Without it, the followers stay attached and stale until follower-rebuild"
            action = :store_true
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

    add_follower_create_args!(s["follower-create"])
    for cmd in ("follower-sync", "follower-rebuild", "follower-delete")
        _add_follower_name_args!(s[cmd])
    end
    _add_migrate_partitions_args!(s["migrate-partitions"])
    @add_arg_table! s["follower-list"] begin
        "--dataset"
            help = "The leader"
            required = true
    end
    @add_arg_table! s["postings"] begin
        "--dataset"
            help = "Dataset ID/name"
            required = true
        "--follower"
            help = "A keyword follower of the dataset: the key is then one of its keys"
        "--key"
            help = "A token (text dataset), a dimension (sparse dataset), or a key of the follower"
            required = true
        "--limit"
            help = "Return only the first N ids, in ascending order. A whole list of a frequent key in a large dataset can hold most of its items"
            arg_type = Int
        "--output"
            help = "Output file (single JSON object). Prints to stdout if omitted"
    end
    for cmd in ("follower-create", "follower-list", "follower-sync", "follower-rebuild", "follower-delete",
                "migrate-partitions", "postings")
        @add_arg_table! s[cmd] begin
            "--workdir"
                help = "Working directory for the database"
                default = "data"
        end
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

`similarity-search-serve`'s own settings object (CHANGELOG.md §1, chunk 21) -- a genuinely
different, separate `ArgParseSettings` from `build_settings()`'s data-operations one, not
a shared parser both binaries happen to use. `serve` is a thin, single-purpose binary
(CHANGELOG.md's own framing): it has exactly two commands, `serve` and `interactive`, and none
of `similarity-search`'s eleven data-operation subcommands (`build` through `load`) are
reachable through it at all -- `similarity-search-server build ...` is now an ArgParse
"unrecognized command" error, not a working (if undocumented) escape hatch, same as how
plain `similarity-search serve` is now the mirror-image error (`serve` isn't one of
`build_settings()`'s commands either). `--host`/`--port`/`--workdir` are all optional
(`default = nothing`) -- an omitted flag means "fall back to `--config`'s TOML file, then a
hardcoded literal," layered in `main_serve`/`run_serve`, not here (an `ArgParseField`
default can only be one fixed value, not "whatever the config file says"). This is also
what gives `run_interactive_serve` (`interactive.jl`) real fields to introspect at all --
CHANGELOG.md §1's Interactive Mode section's own prerequisite for `serve`'s guided form.
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
