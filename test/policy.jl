# Enforces the no-defaults-below-the-public-surface policy (DEVELOPMENT_STRATEGY.md).
#
# A policy nobody verifies drifts, and this one drifts silently: adding a default to an
# internal function breaks nothing and reads like a convenience. What it actually does is put
# a second answer to one question in a second place, which is how `create_project` ended up
# with a `distance=nothing` sentinel whose only job was to arbitrate against
# `create_engine`'s own default.
#
# Parsed with Julia's own parser rather than matched with a regex: a signature can span lines,
# carry `where` clauses and nest default expressions containing commas and parens, and a regex
# that handles all three is harder to trust than this.

"""
    public_names(path) -> Set{Symbol}

Every name the package's own module file exports -- read from the source rather than from a
loaded module, so this stays honest about the *package's* surface instead of picking up
everything its submodules re-export into it.
"""
function public_names(path)
    names = Set{Symbol}()
    for m in eachmatch(r"^export\s+(.+?)(?=\n(?!\s+\w)|\Z)"ms, read(path, String))
        for tok in split(m.captures[1], r"[,\s]+"; keepempty=false)
            startswith(tok, "#") && continue
            push!(names, Symbol(strip(tok, [',', ' '])))
        end
    end
    names
end

# `Module.f(...)` is implementing somebody else's contract (Base.show, TextSearch.fit_profile),
# and the shape of that signature -- defaults included -- belongs to whoever owns it.
_defname(e) = e isa Symbol            ? e :
              Meta.isexpr(e, :.)      ? nothing :
              Meta.isexpr(e, :curly)  ? _defname(e.args[1]) : nothing

_has_default(sig) = any(sig.args[2:end]) do a
    Meta.isexpr(a, :kw) || (Meta.isexpr(a, :parameters) && any(p -> Meta.isexpr(p, :kw), a.args))
end

function _defaults_in(expr, file, out)
    expr isa Expr || return
    isdef = Meta.isexpr(expr, :function) ||
            (Meta.isexpr(expr, :(=)) && (Meta.isexpr(expr.args[1], :call) ||
                                         Meta.isexpr(expr.args[1], :where)))
    if isdef
        sig = expr.args[1]
        Meta.isexpr(sig, :where) && (sig = sig.args[1])
        if Meta.isexpr(sig, :call) && _has_default(sig)
            n = _defname(sig.args[1])
            n === nothing || push!(out, (file, n))
        end
    end
    for a in expr.args
        _defaults_in(a, file, out)
    end
end

@testset "policy: no defaults below the public surface" begin
    src = normpath(joinpath(@__DIR__, "..", "src"))
    public = public_names(joinpath(src, "SimilaritySearchEngine.jl"))
    @test !isempty(public)          # a bad parse would vacuously pass every check below

    offenders = Tuple{String,Symbol}[]
    for f in sort(readdir(src; join=true))
        endswith(f, ".jl") || continue
        _defaults_in(Meta.parseall(read(f, String)), basename(f), offenders)
    end
    internal = sort(unique(o for o in offenders if !(o[2] in public)); by = o -> (o[1], String(o[2])))

    if !isempty(internal)
        println("internal functions carrying a default (defaults belong on the public surface):")
        for (file, name) in internal
            println("  ", file, "  ", name)
        end
    end
    @test isempty(internal)
end
