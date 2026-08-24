using Jutul, Test

@testset "Threading with contexts" begin
    ctx_def = DefaultContext()
    ctx_csr = ParallelCSRContext(thread_type = :threads)
    ctx_csr_batch = ParallelCSRContext(thread_type = :batch)
    ctx_csr_serial = ParallelCSRContext(thread_type = :serial)

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

    for ctx in (ctx_def, ctx_csr, ctx_csr_batch, ctx_csr_serial)
        test_context(ctx)
    end
end
