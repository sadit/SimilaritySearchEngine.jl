# Interactive Mode (PLAN.md §1 "Interactive Mode (Term.jl-guided CLI, new)")
#
# !!! note "All three binaries now have a real guided form (chunk 18 completes it)"
#     PLAN.md's original vision covers `similarity-search` (`run_interactive`, the
#     command-menu form for its eleven data-operation subcommands), `similarity-search-ctl`
#     (`run_interactive_ctl`, chunk 16), and `similarity-search-serve` (`run_interactive_serve`
#     below, chunk 18 -- gated on `serve` finally having real `--host`/`--port`/`--workdir`
#     `ArgParseField`s to introspect at all, `cli.jl`, which is also this chunk's work).
#     `serve`'s form has no command menu at all, unlike the other two -- it's a
#     single-purpose binary with exactly one thing to configure, matching PLAN.md's own "the
#     menu degenerates to that one guided form directly" wording -- and its three fields are
#     pre-filled from whatever `--config`'s TOML file (or a hardcoded literal, absent that)
#     already resolves to, not a static `ArgParseField` default, since `serve`'s real
#     settings are layered (CLI flag > config file > hardcoded literal, see `main_serve`) --
#     a `RadioMenu`/free-text prompt showing "optional" instead of the actual current value
#     would be a worse guided form than the plain CLI it's replacing.
#
# !!! note "Chunk 21: `serve` gets its own `ArgParseSettings` entirely, not just its own fields"
#     Chunk 18 above gave `serve` real `--host`/`--port`/`--workdir` fields, but it still
#     shared `build_settings()`/`main` with the data-ops binary (`similarity-search-server
#     build ...` was technically still reachable). Chunk 21 finished the split PLAN.md's
#     "thin, single-purpose serve binary" framing always described: `build_serve_settings`
#     (`cli.jl`) is a genuinely separate `ArgParseSettings` with exactly `serve`/`interactive`,
#     and `main_serve` (`SimilaritySearchServer.jl`) is its own entry point, called directly
#     by `src/apps/similarity-search-server.jl` instead of going through the shared `main`.
#     `run_interactive_serve` below didn't need to change at all -- it already took
#     `config_path` as a plain argument, not anything tied to the shared settings object.
# !!! note "Term.jl is not used -- REPL.TerminalMenus (stdlib) only"
#     PLAN.md's own warning already found that the locally available `Term.jl` v1.2.0 has
#     no prompt/menu capability at all (only rendering) and recommended pairing it with
#     `REPL.TerminalMenus` for the actual interaction loop. This implementation goes one
#     step further: `Term` isn't even a declared dependency of this project (verified --
#     absent from `Project.toml`), and adding it now would mean a `Pkg.add`/manifest
#     resolution step against a dependency graph that currently includes multiple *local
#     dev* packages (`SimilaritySearch.jl`/`TextSearch.jl`/`RocksDB.jl`) under active,
#     uncommitted edits by the user this same session (see the memory checkpoint's
#     "External dependency volatility" notes) -- a real, demonstrated risk, not a
#     hypothetical one. `REPL.TerminalMenus` alone (added to `Project.toml` as a plain
#     stdlib dependency, zero resolution risk) fully satisfies every *functional*
#     requirement below (menus, per-argument prompts, confirm-before-run); only the bordered
#     `Panel`/`Table` visual polish is left out. Swapping it in later is purely additive.

using REPL.TerminalMenus

"""
    _command_menu_names(settings::ArgParseSettings) -> Vector{String}

The ordered list of top-level subcommand names `settings` declares (declaration order,
via its own `args_table.fields`, *not* `args_table.subsettings`'s `Dict` -- dict iteration
order isn't guaranteed), excluding `interactive` (itself). `serve` never appears in either
settings object this is called on (`build_settings`/`build_ctl_settings`) -- it has its own
genuinely separate `ArgParseSettings` (`build_serve_settings`, chunk 21) that this function
is never called against at all, since `run_interactive_serve` has no command menu (a
single-purpose binary with exactly one thing to configure has no business being one menu
item among many, see this file's header note).
"""
function _command_menu_names(settings::ArgParseSettings)
    names = String[]
    for f in settings.args_table.fields
        ArgParse.is_cmd(f) || continue
        name = f.metavar isa AbstractString ? f.metavar : first(f.metavar)
        name == "interactive" && continue
        push!(names, name)
    end
    return names
end

"""
    _command_fields(settings::ArgParseSettings, cmd_name::String) -> Vector{ArgParse.ArgParseField}

The real, promptable arguments of subcommand `cmd_name` -- every field of its own
`args_table.fields` except an auto-added `--help`/`-h` flag, if one is present (kept as a
guard even though empirically none of this app's subcommand settings currently carry one).
"""
function _command_fields(settings::ArgParseSettings, cmd_name::String)
    sub = settings[cmd_name]
    return [f for f in sub.args_table.fields if f.dest_name != "help" && !("help" in f.long_opt_name)]
end

"""
    _is_new_id_field(cmd_name, dest_name) -> Bool

Whether `--dataset` for this subcommand names a *dataset to create* (`build`'s target, or
`load`'s restore target) rather than an *existing* dataset to select from the workdir.
Only used to decide whether `_discover_project_ids`'s menu applies -- everywhere else,
`--dataset` refers to something that should already exist.
"""
_is_new_id_field(cmd_name::String, dest_name::String) = dest_name == "dataset" && cmd_name in ("build", "load")

"""
    _discover_project_ids(workdir::String) -> Vector{String}

Scans both on-disk dataset layouts this app has (see `_load_engine_for_job`'s docstring in
`cli_handlers.jl`): `<workdir>/datasets/*` (HTTP-created) and `<workdir>/*` with a sibling
`CURRENT` (a RocksDB project directory). This is the dynamic-pre-fill piece PLAN.md's Interactive
Mode section requires -- offering real, currently-valid ids instead of a blank field the
operator has to remember or `describe`/`list` separately first.
"""
function _discover_project_ids(workdir::String)
    ids = String[]
    http_dir = joinpath(workdir, "datasets")
    if isdir(http_dir)
        for id in readdir(http_dir)
            isdir(joinpath(http_dir, id)) && push!(ids, id)
        end
    end
    if isdir(workdir)
        for id in readdir(workdir)
            id == "datasets" && continue
            path = joinpath(workdir, id)
            # "CURRENT" is RocksDB's own marker file: a dataset is built when its project
            # directory exists, where this used to look for a `{id}.snapshot.jld2` beside it.
            isdir(path) && isfile(joinpath(path, "CURRENT")) && push!(ids, id)
        end
    end
    return sort(unique(ids))
end

"""
    _convert_value(f::ArgParse.ArgParseField, raw::AbstractString)

Converts a raw string answer to the type `parse_args` itself would have produced for this
field -- `f.arg_type` is `Int64`/`Float64` for `--k`/`--min-k`/`--epsilon`-style fields
(confirmed by introspecting `build_settings()` directly) and `Any` (meaning: pass the
string through as-is) for everything else. Booleans never reach here -- see
`_prompt_field`'s own `:store_true` branch.
"""
_convert_value(f::ArgParse.ArgParseField, raw::AbstractString) = f.arg_type === Any ? raw : parse(f.arg_type, raw)

"""
    _format_invocation(cmd_name, cmd_args, fields) -> String

Renders the exact non-interactive-equivalent CLI invocation for the confirm-before-run
step, in `fields`' declared order, omitting a field whose *collected* value equals its
declared default (so the printed command is the shortest one that reproduces the same
`cmd_args`, exactly like a human tuning flags by hand would omit the defaults).
"""
function _format_invocation(cmd_name::String, cmd_args::AbstractDict, fields::Vector{ArgParse.ArgParseField})
    parts = ["similarity-search", cmd_name]
    for f in fields
        value = get(cmd_args, f.dest_name, nothing)
        value === nothing && continue
        value == f.default && continue
        push!(parts, "--$(f.dest_name)")
        # A `:store_true` flag takes no value: it is present or it is absent.
        f.action === :store_true || push!(parts, string(value))
    end
    return join(parts, " ")
end

"""
    _prompt_field(cmd_name, f, workdir) -> Any

Prompts for one argument, choosing among four presentations, in this priority order:
1. An existing-dataset menu (`_discover_project_ids`) when `f` is a `--dataset` field that
   refers to something that should already exist (`!_is_new_id_field`) and the workdir
   actually has candidates -- falls through to free text otherwise (empty workdir, or a
   dataset-creating field). Skipped entirely when `allow_dataset_discovery=false` (see
   `run_interactive_ctl` -- a `--dataset` field there names something on a *remote*
   server, not a local workdir this process can `readdir`).
2. A fixed-choice menu (`CLI_CHOICES`) when `f.dest_name` has one declared.
3. A Yes/No menu for a boolean (`:store_true` action) field.
4. Free text via `readline`, pre-filled *as a shown default*, not an editable inline field
   -- accepted on a bare Enter, exactly PLAN.md's "shows and accepts its default" wording;
   `REPL.TerminalMenus`/the stdlib have no portable "editable pre-filled text field"
   primitive, so this is the honest, simplest thing that satisfies the actual requirement.
"""
function _prompt_field(cmd_name::String, f::ArgParse.ArgParseField, workdir::AbstractString; allow_dataset_discovery::Bool=true)
    default_str = f.default === nothing ? nothing : string(f.default)

    if allow_dataset_discovery && f.dest_name == "dataset" && !_is_new_id_field(cmd_name, f.dest_name)
        ids = _discover_project_ids(workdir)
        if !isempty(ids)
            options = vcat(ids, ["(enter manually)"])
            idx = request("Select --dataset:", RadioMenu(options))
            idx == -1 && error("interactive mode cancelled")
            idx <= length(ids) && return ids[idx]
            # falls through to free text below for the "(enter manually)" choice
        end
    end

    choices = get(CLI_CHOICES, f.dest_name, nothing)
    if choices !== nothing
        default_idx = default_str === nothing ? 1 : something(findfirst(==(default_str), choices), 1)
        idx = request("Select --$(f.dest_name)$(default_str === nothing ? "" : " [default: $default_str]"):", RadioMenu(choices); cursor=default_idx)
        idx == -1 && error("interactive mode cancelled")
        return choices[idx]
    end

    if f.action === :store_true
        default_idx = f.default === true ? 1 : 2
        idx = request("--$(f.dest_name)? ($(f.help)):", RadioMenu(["yes", "no"]); cursor=default_idx)
        idx == -1 && error("interactive mode cancelled")
        return idx == 1
    end

    while true
        suffix = default_str === nothing ? (f.required ? "" : " [optional]") : " [default: $default_str]"
        print("--$(f.dest_name)$(isempty(f.help) ? "" : " ($(f.help))")$suffix: ")
        raw = strip(readline())
        if isempty(raw)
            default_str !== nothing && return f.default
            f.required || return nothing
            println("  (required -- please enter a value)")
            continue
        end
        try
            return _convert_value(f, raw)
        catch
            println("  (couldn't parse '$raw' as $(f.arg_type) -- try again)")
        end
    end
end

"""
    run_interactive() -> Int

Entry point for the `interactive` subcommand (PLAN.md §1). Gated on a real TTY (the
PLAN.md-mandated defense-in-depth against a malformed `LocalCLIExecutor`-spawned Job
subprocess deadlocking on `readline`/`request` with no human on the other end, §5.5).
Command menu → per-argument prompts (in `_command_fields`'s declared order, `--workdir`
first so dataset-discovery prompts after it can use the workdir the operator actually
picked) → confirm-before-run panel → dispatch via the same `dispatch_command` the
non-interactive path uses, so there is exactly one command-name → `execute_*` mapping
regardless of how the arguments were collected.
"""
function run_interactive()
    if !(stdin isa Base.TTY)
        println(stderr, "Error: 'interactive' requires a real terminal (stdin is not a TTY)")
        return 1
    end

    settings = build_settings()
    names = _command_menu_names(settings)

    println("SimilaritySearchServer -- interactive mode")
    idx = request("Choose a command:", RadioMenu(names))
    if idx == -1
        println("Cancelled.")
        return 0
    end
    cmd_name = names[idx]

    fields = _command_fields(settings, cmd_name)
    sort!(fields, by = f -> f.dest_name == "workdir" ? 0 : 1)

    cmd_args = Dict{String, Any}()
    println()
    println("-- $cmd_name --")
    for f in fields
        cmd_args[f.dest_name] = _prompt_field(cmd_name, f, get(cmd_args, "workdir", "data"))
    end

    println()
    println(_format_invocation(cmd_name, cmd_args, fields))
    confirm_idx = request("Run this command?", RadioMenu(["yes", "no"]))
    if confirm_idx != 1
        println("Cancelled.")
        return 0
    end

    return dispatch_command(cmd_name, cmd_args)
end

"""
    _format_ctl_invocation(path, cmd_args, fields) -> String

`-ctl` counterpart of `_format_invocation`: same "omit a field still at its declared
default" logic, but the command name is a `path` (one segment for most commands, two for
`jobs <sub>` -- the only nested command tree either binary has) rather than always one
name, and the printed prefix is the `-ctl` binary, not `similarity-search`.
"""
function _format_ctl_invocation(path::Vector{String}, cmd_args::AbstractDict, fields::Vector{ArgParse.ArgParseField})
    parts = vcat(["similarity-search-ctl"], path)
    for f in fields
        value = get(cmd_args, f.dest_name, nothing)
        value === nothing && continue
        value == f.default && continue
        push!(parts, "--$(f.dest_name)")
        # A `:store_true` flag takes no value: it is present or it is absent.
        f.action === :store_true || push!(parts, string(value))
    end
    return join(parts, " ")
end

"""
    run_interactive_ctl(base_url::String) -> Int

Entry point for `similarity-search-ctl interactive` (PLAN.md §1) -- the `-ctl` counterpart
of `run_interactive()`, gated on the same real-TTY check. `base_url` is already resolved
by `main_ctl` from `--host`/`--port` (global flags parsed *before* this ever runs, unlike
`similarity-search`'s per-command `--workdir`), so there is no prompt for it here.

Two structural differences from the data-operations version, not oversights:
- `--dataset` fields (`stats`/`dump`/`load`) are always free text, never a menu of
  locally-discovered ids -- `-ctl` is a remote HTTP client with no workdir of its own
  (`_prompt_field(...; allow_dataset_discovery=false)`), so there is nothing on this
  process's filesystem to scan.
- `jobs` is a nested subcommand (`jobs queue|result|block|resume|kill|gc`) -- picking it
  shows a second menu for its own children before any field prompting starts, since
  `_command_menu_names` only enumerates one level at a time.

Dispatches via the exact same `dispatch_ctl_command(parsed, base_url)` the non-interactive
path uses (reconstructing the same `{"%COMMAND%" => ..., name => args}` shape
`parse_ctl_commandline` itself would have produced), so there is exactly one
command-name → `ctl_*` mapping regardless of how the arguments were collected.
"""
function run_interactive_ctl(base_url::String)
    if !(stdin isa Base.TTY)
        println(stderr, "Error: 'interactive' requires a real terminal (stdin is not a TTY)")
        return 1
    end

    settings = build_ctl_settings()
    names = _command_menu_names(settings)

    println("similarity-search-ctl -- interactive mode")
    idx = request("Choose a command:", RadioMenu(names))
    if idx == -1
        println("Cancelled.")
        return 0
    end
    path = [names[idx]]
    sub = settings[path[1]]

    if path[1] == "jobs"
        subnames = _command_menu_names(sub)
        idx2 = request("Choose a jobs subcommand:", RadioMenu(subnames))
        if idx2 == -1
            println("Cancelled.")
            return 0
        end
        push!(path, subnames[idx2])
        sub = sub[path[2]]
    end

    fields = [f for f in sub.args_table.fields if f.dest_name != "help" && !("help" in f.long_opt_name)]

    cmd_args = Dict{String, Any}()
    println()
    println("-- $(join(path, " ")) --")
    for f in fields
        cmd_args[f.dest_name] = _prompt_field("", f, ""; allow_dataset_discovery=false)
    end

    println()
    println(_format_ctl_invocation(path, cmd_args, fields))
    confirm_idx = request("Run this command?", RadioMenu(["yes", "no"]))
    if confirm_idx != 1
        println("Cancelled.")
        return 0
    end

    parsed = if length(path) == 1
        Dict("%COMMAND%" => path[1], path[1] => cmd_args)
    else
        Dict("%COMMAND%" => "jobs", "jobs" => Dict("%COMMAND%" => path[2], path[2] => cmd_args))
    end
    return dispatch_ctl_command(parsed, base_url)
end

"""
    _prompt_with_default(label::String, current::AbstractString) -> String

Free-text prompt pre-filled with `current` as the shown default (Enter accepts it, the
same convention `_prompt_field`'s free-text branch already established) -- used by
`run_interactive_serve` for its three plain `host`/`port`/`workdir` fields, whose "current
value" comes from the loaded `--config` TOML file rather than a static `ArgParseField`
default the way every other interactive prompt in this codebase works.
"""
function _prompt_with_default(label::String, current::AbstractString)
    print("$label [default: $current]: ")
    raw = strip(readline())
    return isempty(raw) ? current : raw
end

"""
    run_interactive_serve(config_path::String) -> Int

Entry point for `similarity-search-server interactive` (PLAN.md §1) -- the guided form
for `serve`'s own settings. No command menu (unlike `run_interactive`/`run_interactive_ctl`)
since `serve` is a single-purpose binary with exactly one thing to configure, matching
PLAN.md's own "the menu degenerates to that one guided form directly" wording. Prompts for
`host`/`port`/`workdir`, each pre-filled with whatever `--config`'s TOML file already
resolves to (or the hardcoded literal, absent that) -- the same layering `main`'s
non-interactive `serve` branch applies, so the shown default is honest about what would
actually happen on a bare Enter, not a fixed placeholder. Confirm-before-run, then
dispatches through `run_serve`, the exact function `main`'s own `serve` branch calls, so
there is exactly one "start the server" code path regardless of how the settings were
collected.
"""
function run_interactive_serve(config_path::String)
    if !(stdin isa Base.TTY)
        println(stderr, "Error: 'interactive' requires a real terminal (stdin is not a TTY)")
        return 1
    end

    config = load_config(config_path)
    server_cfg = get(config, "server", Dict{String, Any}())
    paths_cfg = get(config, "paths", Dict{String, Any}())

    default_host = string(get(server_cfg, "host", "127.0.0.1"))
    default_port = string(get(server_cfg, "port", 8080))
    default_workdir = string(get(paths_cfg, "workdir", "data"))

    println("similarity-search-server -- interactive mode")
    println()
    println("-- serve --")
    host = _prompt_with_default("--host", default_host)
    port_str = _prompt_with_default("--port", default_port)
    port = tryparse(Int, port_str)
    while port === nothing
        println("  (couldn't parse '$port_str' as Int -- try again)")
        port_str = _prompt_with_default("--port", default_port)
        port = tryparse(Int, port_str)
    end
    workdir = _prompt_with_default("--workdir", default_workdir)

    println()
    println("similarity-search-server serve --host $host --port $port --workdir $workdir")
    confirm_idx = request("Run this command?", RadioMenu(["yes", "no"]))
    if confirm_idx != 1
        println("Cancelled.")
        return 0
    end

    auth_enabled = get(get(config, "auth", Dict{String, Any}()), "enabled", false) === true
    resources_cfg = get(config, "resources", Dict{String, Any}())
    batch_threads_pct = resolve_batch_threads_pct(resources_cfg)
    configured_slots = get(resources_cfg, "max_concurrent_queries", 0)
    max_concurrent_queries = configured_slots isa Integer ? Int(configured_slots) : 0
    return run_serve(host, port, workdir; auth_enabled, batch_threads_pct, max_concurrent_queries)
end
