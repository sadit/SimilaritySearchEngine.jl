# The job dispatcher (`Executors.run_dispatcher!`) with an executor that starts no process. The
# end-to-end files test the dispatcher with real subprocesses; this file tests what happens when
# starting a job fails, which those files reach only by chance.
using SimilaritySearchServer: Jobs, Executors

# Records the commands it receives. A command whose first word is "fail" raises in `submit`.
struct RecordingExecutor <: Executors.AbstractJobExecutor
    commands::Vector{Vector{String}}
end

function Executors.submit(ex::RecordingExecutor, cmd::Vector{String}; threads::Int=1)
    !isempty(cmd) && cmd[1] == "fail" && error("this executor cannot start the job")
    push!(ex.commands, cmd)
    return "h$(length(ex.commands))"
end

Executors.poll(::RecordingExecutor, ::String) = "completed"

function wait_for_state(job_mgr, id, state; timeout=10.0)
    t0 = time()
    while time() - t0 < timeout
        first(Jobs.get_job(job_mgr, id)) === state && return true
        sleep(0.05)
    end
    return false
end

@testset "job dispatcher" begin
    workdir = mktempdir()
    job_mgr = Jobs.init_job_manager(workdir)

    @testset "a job with an empty command is started" begin
        # TOML reads `command = []` as `Any[]`. Until 2026-11-20 the dispatcher passed that to
        # `submit` as a `Vector{Any}`, and the `MethodError` ended the dispatcher.
        ex = RecordingExecutor(Vector{String}[])
        id = Jobs.create_job!(job_mgr, "allknn", String[])
        active = Dict{String,String}()
        Executors._dispatch_one!(job_mgr, ex, active, id, 1)
        @test ex.commands == [String[]]
        state, record = Jobs.get_job(job_mgr, id)
        @test state === Jobs.Running
        @test record["executor_handle"] == "h1"
        @test active[id] == "h1"
    end

    @testset "a job that cannot start fails, and the dispatcher continues" begin
        ex = RecordingExecutor(Vector{String}[])
        bad = Jobs.create_job!(job_mgr, "allknn", ["fail"])
        # The dispatcher has no stop operation. Its task stays alive until the test process
        # ends, polling an empty spool in a temporary directory.
        task = errormonitor(@async Executors.run_dispatcher!(job_mgr, ex; poll_interval=0.05))
        @test wait_for_state(job_mgr, bad, Jobs.Failed)
        @test occursin("cannot start the job", Jobs.get_job(job_mgr, bad)[2]["error"])

        good = Jobs.create_job!(job_mgr, "allknn", ["allknn", "--dataset", "x"])
        @test wait_for_state(job_mgr, good, Jobs.Completed)
        @test ex.commands == [["allknn", "--dataset", "x"]]
        @test !istaskdone(task)
    end
end
