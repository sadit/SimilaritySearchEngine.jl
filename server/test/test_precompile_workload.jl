using Test
using SimilaritySearchServer
import PrecompileTools

# The precompilation of this package runs `_precompile_cli` and ignores its exit codes
# (`server/src/precompile_workload.jl`). A command that fails there only makes the first call of
# that command slower, so this file runs the same commands and checks the codes.
#
# The preference `precompile_workload = false` turns the workload off. The commands are then not
# compiled in the package image, and this file would compile them: 1 min 30 s, measured on
# 2026-11-25, against 6 s with the workload. So the file runs only when the workload is on.
# `workload_enabled` is not in the README of PrecompileTools; it exists from version 1.1.1 to 1.4.0.
@testset "precompile workload" begin
    if PrecompileTools.workload_enabled(SimilaritySearchServer)
        workdir = mktempdir()
        codes = redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                SimilaritySearchServer._precompile_cli(workdir)
            end
        end
        # One code per command: build (2), searchbatch (2), describe, allknn, fft, neardup, hsp,
        # closestpair, rebuild, dump and load.
        @test codes == zeros(Cint, 13)
        rm(workdir; force=true, recursive=true)
    else
        @test_skip "the preference precompile_workload = false turns the workload off"
    end
end
