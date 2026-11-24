# Plugins of the server (CHANGELOG.md §5.13, issue #5, phase 6). Included in `module Server`.
#
# A plugin is a Julia module that adds key sources and kinds of follower. `serve` loads the
# plugins of `[plugins]` in its configuration before it opens any dataset, registers each one in
# the engine, and reads the wire names that the module declares. The module does not depend on
# the server: it declares its wire names with two constants and an optional `setup`.
#
# Code of a plugin is defined after the functions of this module were compiled, so every call
# into it goes through `Base.invokelatest`.

"""
    LoadedPlugin

A plugin that this process loaded: its name, which is the name of its module, the module, where
it came from (`"path" => file` or `"package" => name`), and the constructors of its key sources
and of its kinds of follower by wire name.
"""
struct LoadedPlugin
    name::String
    mod::Module
    origin::Pair{String,String}
    key_sources::Dict{String,Any}
    follower_kinds::Dict{String,Any}
end

"The plugins this process loaded, by name. A plugin is loaded once per process."
const PLUGINS = Dict{String,LoadedPlugin}()
const PLUGINS_LOCK = ReentrantLock()

"""
    PluginLoadError(plugin, msg)

A plugin of the configuration that could not be loaded. `serve` does not start, and prints the
message (CHANGELOG.md §5.13).
"""
struct PluginLoadError <: Exception
    plugin::String
    msg::String
end
Base.showerror(io::IO, e::PluginLoadError) = print(io, "the plugin ", e.plugin, " was not loaded: ", e.msg)

const _PLUGIN_TABLE_FIELDS = ("path", "package", "settings")
const _WIRE_NAME = r"^[A-Za-z0-9_-]+$"

_latest_isdefined(m::Module, s::Symbol) = Base.invokelatest(isdefined, m, s)
_latest_getfield(m::Module, s::Symbol) = Base.invokelatest(getfield, m, s)

"""
    load_plugins!(config; base_dir=pwd()) -> Vector{String}

Loads every plugin of the `[plugins]` section of `config`, a parsed configuration, in the order
of their names, and returns their names. Each table is `[plugins.<Module>]` with `path`, a `.jl`
file relative to `base_dir` that defines `module <Module>`, or `package`, a package of the
active environment; and an optional table `settings`, passed to the function `setup` of the
module. Raises [`PluginLoadError`](@ref) for the first plugin that cannot be loaded.
"""
function load_plugins!(config::AbstractDict; base_dir::AbstractString=pwd())
    tables = get(config, "plugins", nothing)
    tables === nothing && return String[]
    tables isa AbstractDict || throw(PluginLoadError("[plugins]",
        "the section [plugins] holds one table per plugin, such as [plugins.Decades] with `path` or `package`"))
    names = sort!(String[String(k) for k in keys(tables)])
    for name in names
        load_plugin!(name, tables[name]; base_dir)
    end
    return names
end

"""
    load_plugin!(name, table; base_dir=pwd()) -> LoadedPlugin

Loads one plugin, as [`load_plugins!`](@ref) does for each table of `[plugins]`: its module, then
`setup(settings)` if the module has it, then `register_plugin!` of the engine, and then its wire
names. A plugin that this process already loaded is returned as it is, without loading it again.
"""
function load_plugin!(name::AbstractString, table; base_dir::AbstractString=pwd())
    name = String(name)
    loaded = lock(() -> get(PLUGINS, name, nothing), PLUGINS_LOCK)
    loaded === nothing || return loaded
    Base.isidentifier(name) || throw(PluginLoadError(name,
        "the name of a plugin is the name of its module, a Julia identifier"))
    table isa AbstractDict || throw(PluginLoadError(name, "[plugins.$name] must be a table with `path` or `package`"))
    unknown = sort!([String(k) for k in keys(table) if !(k in _PLUGIN_TABLE_FIELDS)])
    isempty(unknown) || throw(PluginLoadError(name, "[plugins.$name] has the unknown field(s) $(join(unknown, ", ")); " *
        "it takes `path` or `package`, and `settings`"))
    path, package = get(table, "path", nothing), get(table, "package", nothing)
    (path === nothing) == (package === nothing) && throw(PluginLoadError(name,
        "[plugins.$name] needs `path`, a .jl file that defines `module $name`, or `package`, a package of the " *
        "active environment: one of the two"))
    settings = get(table, "settings", Dict{String,Any}())
    settings isa AbstractDict || throw(PluginLoadError(name, "[plugins.$name.settings] must be a table"))

    mod, origin = path !== nothing ? _load_plugin_file(name, path, base_dir) : _load_plugin_package(name, package)
    key_sources = _plugin_constructors(name, mod, :KEY_SOURCES)
    follower_kinds = _plugin_constructors(name, mod, :FOLLOWER_KINDS)
    if _latest_isdefined(mod, :setup)
        try
            Base.invokelatest(_latest_getfield(mod, :setup), Dict{String,Any}(String(k) => v for (k, v) in settings))
        catch e
            throw(PluginLoadError(name, "$name.setup(settings) raised: " * sprint(showerror, e)))
        end
    end
    SSE.register_plugin!(mod)
    plugin = LoadedPlugin(name, mod, origin, key_sources, follower_kinds)
    lock(() -> (PLUGINS[name] = plugin), PLUGINS_LOCK)
    return plugin
end

function _load_plugin_file(name::String, path, base_dir::AbstractString)
    path isa AbstractString || throw(PluginLoadError(name, "`path` of [plugins.$name] must be a string"))
    file = normpath(isabspath(path) ? path : joinpath(base_dir, path))
    isfile(file) || throw(PluginLoadError(name, "the file $file does not exist; `path` is relative to the directory of the configuration file"))
    sym = Symbol(name)
    # A module of this name that the process already has is not included again: including the
    # file again would make a new module, and the followers saved with its types could not be
    # read with the new one.
    if !(_latest_isdefined(Main, sym) && _latest_getfield(Main, sym) isa Module)
        try
            Base.include(Main, file)
        catch e
            throw(PluginLoadError(name, "including $file raised: " * sprint(showerror, e)))
        end
    end
    mod = _latest_isdefined(Main, sym) ? _latest_getfield(Main, sym) : nothing
    mod isa Module || throw(PluginLoadError(name, "the file $file does not define `module $name` at its top level"))
    return mod, "path" => file
end

function _load_plugin_package(name::String, package)
    package isa AbstractString || throw(PluginLoadError(name, "`package` of [plugins.$name] must be a string"))
    package == name || throw(PluginLoadError(name, "`package` is $(repr(package)); the table of a package is named " *
        "after its module, [plugins.$package]"))
    mod = try
        Base.require(Main, Symbol(package))
    catch e
        throw(PluginLoadError(name, "loading the package $package raised: " * sprint(showerror, e)))
    end
    return mod, "package" => String(package)
end

# The constructors that a plugin declares in `KEY_SOURCES` or `FOLLOWER_KINDS`, by wire name.
function _plugin_constructors(name::String, mod::Module, field::Symbol)
    _latest_isdefined(mod, field) || return Dict{String,Any}()
    table = _latest_getfield(mod, field)
    table isa AbstractDict || throw(PluginLoadError(name, "$name.$field must be a Dict from wire names to constructors"))
    out = Dict{String,Any}()
    for (k, f) in table
        k isa AbstractString && occursin(_WIRE_NAME, k) || throw(PluginLoadError(name,
            "$name.$field has the name $(repr(k)); a wire name has letters, digits, '_' and '-'"))
        out[String(k)] = f
    end
    return out
end

# --- Wire names ------------------------------------------------------------------------

# A source or a kind named `Plugin.name`. A source that starts with `meta:` is the engine's,
# whatever its field name holds.
is_plugin_name(name::AbstractString) = !startswith(name, "meta:") && occursin('.', name)

# The constructor of the wire name `Plugin.name` in the table `field` of its plugin, or the
# message of a `400`. A plugin that is not loaded raises `PluginNotLoaded`.
function _plugin_constructor(wire::AbstractString, field::Symbol)
    i = findfirst('.', wire)
    plugin, entry = wire[1:prevind(wire, i)], wire[nextind(wire, i):end]
    p = lock(() -> get(PLUGINS, plugin, nothing), PLUGINS_LOCK)
    if p === nothing
        loaded = lock(() -> sort!(collect(keys(PLUGINS))), PLUGINS_LOCK)
        throw(SSE.PluginNotLoaded(plugin, "$(repr(wire)) names the plugin $plugin, which this server has not loaded; " *
            "a plugin is loaded from [plugins.$plugin] of the configuration of serve. " *
            (isempty(loaded) ? "No plugin is loaded." : "The loaded plugins are $(join(loaded, ", ")).")))
    end
    table, what = field === :KEY_SOURCES ? (p.key_sources, "key source") : (p.follower_kinds, "kind of follower")
    f = get(table, entry, nothing)
    f === nothing || return f
    names = sort!(["$plugin.$k" for k in keys(table)])
    return "the plugin $plugin has no $what named $(repr(entry)); " *
           (isempty(names) ? "it declares none" : "it has $(join(names, ", "))")
end

# Calls the constructor of a plugin. A constructor that refuses its options answers `400`: an
# `EngineError` with its kind, the usual errors of a wrong option with their message.
function _call_plugin_constructor(wire::AbstractString, f, args...)
    try
        return Base.invokelatest(f, args...)
    catch e
        e isa SSE.EngineError && rethrow()
        e isa Union{ArgumentError,KeyError,MethodError,InexactError,DomainError,ErrorException} || rethrow()
        return "the options of $wire were refused: " * sprint(showerror, e)
    end
end

"""
    plugin_key_source(wire, options) -> Union{SSE.KeySource, String}

The key source `Plugin.name` built with `options`, or the message of a `400`. Raises
`PluginNotLoaded` when the plugin is not loaded.
"""
function plugin_key_source(wire::AbstractString, options::AbstractDict)
    f = _plugin_constructor(wire, :KEY_SOURCES)
    f isa String && return f
    source = _call_plugin_constructor(wire, f, Dict{String,Any}(options))
    source isa String && return source
    source isa SSE.KeySource || return "the constructor of $wire returned a $(typeof(source)), not a KeySource"
    return source
end

"""
    plugin_follower_spec(wire, source, options) -> Union{SSE.FollowerSpec, String}

The kind of follower `Plugin.name` built with the key source `source` and `options`, or the
message of a `400`. Raises `PluginNotLoaded` when the plugin is not loaded.
"""
function plugin_follower_spec(wire::AbstractString, source::SSE.KeySource, options::AbstractDict)
    f = _plugin_constructor(wire, :FOLLOWER_KINDS)
    f isa String && return f
    spec = _call_plugin_constructor(wire, f, source, Dict{String,Any}(options))
    spec isa String && return spec
    spec isa SSE.FollowerSpec || return "the constructor of $wire returned a $(typeof(spec)), not a FollowerSpec"
    return spec
end

"""
    plugin_dict(p::LoadedPlugin) -> Dict

The wire form of a loaded plugin: its name, its `path` or `package`, and the wire names of its
key sources and of its kinds of follower.
"""
plugin_dict(p::LoadedPlugin) = Dict{String,Any}("name" => p.name, first(p.origin) => last(p.origin),
    "key_sources" => sort!(["$(p.name).$k" for k in keys(p.key_sources)]),
    "follower_kinds" => sort!(["$(p.name).$k" for k in keys(p.follower_kinds)]))

"`GET /api/v1/plugins`: the plugins that this server loaded (CHANGELOG.md §5.13)."
function handle_list_plugins(req::HTTP.Request, app)
    list = lock(() -> [plugin_dict(p) for p in sort!(collect(values(PLUGINS)); by=p -> p.name)], PLUGINS_LOCK)
    return json_response(200, Dict("status" => "ok", "plugins" => list))
end
