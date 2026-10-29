using Jutul, Test, KernelAbstractions

struct FailingPolyesterSolve end

function Jutul.solve_and_update!(::Nothing, ::FailingPolyesterSolve, dt; kwarg...)
    return Jutul.Polyester.@batch per = thread for i in 1:Threads.nthreads()
        if i == 1
            error("Batched solve failed")
        end
    end
end

@testset "Polyester recovery after a caught solve exception" begin
    ids = zeros(Int, 4096)
    config = Dict{Symbol, Any}(
        :linear_solver => nothing,
        :safe_mode => true,
        :failure_cuts_timestep => true,
        :info_level => -1,
    )
    report = Dict{Symbol, Any}()
    try
        Jutul.Polyester.reset_threads!()
        Jutul.threaded_loop(i -> (ids[i] = Threads.threadid()), length(ids), :batch)
        workers_before = length(unique(ids))

        Jutul.perform_step_solve_impl!(
            report, nothing, FailingPolyesterSolve(), config,
            1.0, 1, nothing, 1.0, Jutul.default_executor()
        )
        @test report[:failure_exception] isa Exception
        @test occursin("Batched solve failed", sprint(showerror, report[:failure_exception]))

        fill!(ids, 0)
        Jutul.threaded_loop(i -> (ids[i] = Threads.threadid()), length(ids), :batch)
        @test all(>(0), ids)
        @test length(unique(ids)) == workers_before
    finally
        Jutul.Polyester.reset_threads!()
    end
end

@testset "Threading with contexts" begin
    ctx_def = DefaultContext()
    ctx_csr = ParallelCSRContext(thread_type = :threads)
    ctx_csr_batch = ParallelCSRContext(thread_type = :batch)
    ctx_csr_serial = ParallelCSRContext(thread_type = :serial)
    ctx_csr_static = ParallelCSRContext(thread_type = :threads_static)

    function test_context(ctx)
        N = 1000
        v1 = zeros(N)
        function test_inner(i)
            v1[i] = i
        end
        Jutul.threaded_loop(test_inner, N, ctx)
        @test v1 == collect(1:N)

        v2 = zeros(N)
        function test_inner_minbatch(i)
            v2[i] = i
        end
        Jutul.threaded_loop_minbatch(test_inner_minbatch, N, ctx)
        @test v2 == collect(1:N)
    end

    for ctx in (ctx_def, ctx_csr, ctx_csr_batch, ctx_csr_serial, ctx_csr_static)
        test_context(ctx)
    end
end

@testset "CPU batches and context worker limits" begin
    available = Threads.nthreads(:default)
    for workers in unique([1, min(2, available), available])
        batch_workers = min(workers, Int(Jutul.Polyester.num_cores()))
        for kind in (:threads, :threads_static, :batch)
            ctx = ParallelCSRContext(workers; minbatch = 64, thread_type = kind)
            ids = zeros(Int, 4096)
            Jutul.threaded_loop_minbatch(i -> (ids[i] = Threads.threadid()), length(ids), ctx)
            used = length(unique(ids))
            expected = kind == :batch ? batch_workers : workers
            @test kind == :threads ? 1 <= used <= workers : used == expected
            @test all(>(0), ids)
            @test isnothing(Jutul.threaded_loop_minbatch(identity, 0, ctx))
        end
        ctx = KernelAbstractionsContext(CPU(); nthreads = workers, minbatch = 64)
        ids = zeros(Int, 4096)
        Jutul.threaded_loop_minbatch(i -> (ids[i] = Threads.threadid()), length(ids), ctx)
        @test length(unique(ids)) == batch_workers
        @test Jutul.nthreads(adjoint(ctx)) == workers
        mixed = KernelAbstractionsContext(CPU(); nthreads = workers, linear_float_type = Float32)
        @test Jutul.nthreads(Jutul.linear_solver_context(mixed)) == workers
        small = zeros(Int, 32)
        Jutul.threaded_loop(i -> (small[i] = Threads.threadid()), length(small), ctx)
        @test all(==(Threads.threadid()), small)
    end
end
