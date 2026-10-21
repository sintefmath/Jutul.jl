using Jutul, Test, KernelAbstractions

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
        for kind in (:threads, :threads_static, :batch)
            ctx = ParallelCSRContext(workers; minbatch = 64, thread_type = kind)
            ids = zeros(Int, 4096)
            Jutul.threaded_loop_minbatch(i -> (ids[i] = Threads.threadid()), length(ids), ctx)
            used = length(unique(ids))
            @test kind == :threads ? 1 <= used <= workers : used == workers
            @test all(>(0), ids)
            @test isnothing(Jutul.threaded_loop_minbatch(identity, 0, ctx))
        end
        ctx = KernelAbstractionsContext(CPU(); nthreads = workers, minbatch = 64)
        ids = zeros(Int, 4096)
        Jutul.threaded_loop_minbatch(i -> (ids[i] = Threads.threadid()), length(ids), ctx)
        @test length(unique(ids)) == workers
        @test Jutul.nthreads(adjoint(ctx)) == workers
        mixed = KernelAbstractionsContext(CPU(); nthreads = workers, linear_float_type = Float32)
        @test Jutul.nthreads(Jutul.linear_solver_context(mixed)) == workers
        small = zeros(Int, 32)
        Jutul.threaded_loop(i -> (small[i] = Threads.threadid()), length(small), ctx)
        @test all(==(Threads.threadid()), small)
    end
end
