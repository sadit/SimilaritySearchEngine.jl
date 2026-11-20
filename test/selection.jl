# Which test files a run includes. The engine's suite (`runtests.jl` in this directory) and the
# server's suite (`server/test/runtests.jl`) both include this file.
#
#     Pkg.test()                                    # light level: the files marked light
#     Pkg.test(test_args=["full"])                  # every file, heavy testsets included
#     Pkg.test(test_args=["text", "persistence"])   # the files with these names or tags
#     Pkg.test(test_args=["light", "jobs"])         # the light level and the files tagged jobs
#
# The environment gives the same choices: `SSE_TEST_LEVEL=full` and
# `SSE_TEST_SELECT="text persistence"`. Use the environment when the command is fixed, as in CI
# or in a SLURM script.
#
# A name or tag that matches no file is an error. A misspelled tag that ran nothing would look
# like a passing run.

"""
    TestFile(name, path; tags=String[], light=true)

One test file of a suite. `name` is what a run selects it by. `tags` name the topics it covers.
A `light` file runs in the default run, and the files that are not light run only in a full
run or when selected.
"""
struct TestFile
    name::String
    path::String
    tags::Vector{String}
    light::Bool
end

TestFile(name, path; tags=String[], light=true) = TestFile(name, path, tags, light)

struct TestSelection
    full::Bool
    words::Vector{String}
end

function TestSelection(args)
    words = String[lowercase(w) for w in vcat(collect(args), split(get(ENV, "SSE_TEST_SELECT", "")))]
    full = "full" in words || lowercase(get(ENV, "SSE_TEST_LEVEL", "light")) == "full"
    TestSelection(full, filter(!=("full"), words))
end

"""
    run_heavy(sel)

Whether the heavy testsets run. They run in a full run and in a run that names its files. They
are skipped in the default light run.
"""
run_heavy(sel::TestSelection) = sel.full || !isempty(sel.words)

_matches(f::TestFile, w) = w == f.name || w in f.tags || (w == "light" && f.light)

"""
    selected_files(sel, files)

The files of `files` that the run `sel` includes, in the order of `files`.
"""
function selected_files(sel::TestSelection, files::Vector{TestFile})
    sel.full && return files
    isempty(sel.words) && return filter(f -> f.light, files)
    for w in sel.words
        if !any(f -> _matches(f, w), files)
            names = join((f.name for f in files), ", ")
            tags = join(sort!(unique!(vcat((f.tags for f in files)...))), ", ")
            error("no test file is named or tagged \"$w\".\n  names: $names\n  tags: light, full, $tags")
        end
    end
    return filter(f -> any(w -> _matches(f, w), sel.words), files)
end

"""
    report_selection(suite, sel, files, chosen)

Logs which files a run includes, so that a partial run is not mistaken for a full one.
"""
function report_selection(suite, sel::TestSelection, files, chosen)
    skipped = [f.name for f in files if !(f in chosen)]
    level = sel.full ? "full" : isempty(sel.words) ? "light" : join(sel.words, " ")
    if isempty(skipped)
        @info "$suite: $level run, all $(length(files)) test files"
    else
        @info "$suite: $level run, $(length(chosen)) of $(length(files)) test files" ran = [f.name for f in chosen] skipped
    end
end
