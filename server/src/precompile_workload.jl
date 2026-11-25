# A command of `similarity-search` compiles most of its code the first time it runs, and every job
# runs its command in a new Julia process. Measured on 2026-11-25, a `build` of 724 vectors took
# 19.0 s in a new process, and 16.9 s of that was compilation. The workload below runs the
# commands during the precompilation of this package, and Julia stores the compiled code in the
# package image (PrecompileTools.jl).

"""
    _precompile_cli(workdir) -> Vector{Cint}

Writes a dataset of 300 items to `workdir`, runs commands of `similarity-search` on it with
[`main`](@ref), and returns their exit codes. The commands are the ones that a job runs, plus
`searchbatch` and `describe`. The precompilation of this package runs it. The light test suite
checks that every code is 0, because the precompilation ignores the codes.

A search graph tunes its parameters when it reaches 256 items (`starting_callback` of
SimilaritySearch.jl), so the dataset has 300 items. The text `build` does not compile the graph of
its query expansion: TextSearch.jl builds that graph for a vocabulary of more than 4096 tokens, and
such a dataset adds about 10 s to the precompilation.
"""
function _precompile_cli(workdir::AbstractString)
    words = ["river", "mountain", "creature", "light", "night", "storm", "friend", "letter",
             "journey", "fire", "ice", "voice"]
    input = joinpath(workdir, "input.jsonl")
    open(input, "w") do io
        for i in 1:300
            item = Dict{String,Any}(
                "doc_id" => "w$i",
                "text" => "the $(words[mod1(i, 12)]) and the $(words[mod1(5i, 12)]) of the $(words[mod1(7i + 1, 12)])",
                "vector" => [sin(i * j) for j in 1:8],
                "word_count" => 9,
                "keywords" => [words[mod1(i, 12)]])
            println(io, JSON.json(item))
        end
    end
    out(name) = joinpath(workdir, name)
    commands = [
        ["build", "--dataset", "d", "--input", input, "--index-kind", "searchgraph", "--distance", "L2"],
        ["build", "--dataset", "t", "--input", input, "--index-kind", "bm25_invfile"],
        ["searchbatch", "--dataset", "d", "--queries", input, "--output", out("d.jsonl"), "--k", "5"],
        ["searchbatch", "--dataset", "t", "--queries", input, "--output", out("t.jsonl"), "--k", "5"],
        ["describe", "--dataset", "d", "--output", out("describe.json")],
        ["allknn", "--dataset", "d", "--k", "5", "--output", out("allknn.json")],
        ["fft", "--dataset", "d", "--k", "8", "--output", out("fft.json")],
        ["neardup", "--dataset", "d", "--epsilon", "0.1", "--output", out("neardup.json")],
        ["hsp", "--dataset", "d", "--queries", input, "--k", "5", "--output", out("hsp.json")],
        ["closestpair", "--dataset", "d", "--output", out("closestpair.json")],
        ["rebuild", "--dataset", "d"],
        ["dump", "--dataset", "d", "--output", out("bundle")],
        ["load", "--bundle", out("bundle"), "--dataset", "d2"],
    ]
    return Cint[main(vcat(c, ["--workdir", workdir])) for c in commands]
end

using PrecompileTools: @compile_workload

@compile_workload begin
    try
        workdir = mktempdir()
        try
            # The commands print their progress. The precompilation of a package does not print it.
            redirect_stdout(devnull) do
                redirect_stderr(devnull) do
                    _precompile_cli(workdir)
                end
            end
        finally
            rm(workdir; force=true, recursive=true)
        end
    catch
        # A workload that fails makes the first command slower. It does not stop the precompilation.
    end
end
