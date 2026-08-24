export ParallelCSRContext
"""
    ParallelCSRContext()
    # 5 threads (provided that Julia was started at least --threads=5)
    ParalellCSRContext(5)

A context that uses a CSR sparse matrix format together with threads.
"""
struct ParallelCSRContext <: CPUJutulContext
    matrix_layout
    minbatch::Integer
    nthreads::Integer
    partitioner::JutulPartitioner
    thread_type::Symbol
    function ParallelCSRContext(nthreads = Threads.nthreads();
            partitioner = MetisPartitioner(),
            matrix_layout = EquationMajorLayout(),
            minbatch = minbatch(nothing),
            thread_type = :threads
        )
        maxthreads = Threads.nthreads()
        if nthreads > maxthreads
            @warn "nthreads > Threads.nthreads() in ParallelCSRContext. Using Threads.nthreads() instead."
            nthreads = maxthreads
        end
        thread_type in (:threads, :batch, :serial) || throw(ArgumentError("thread_type must be :threads, :serial or :batch"))
        new(matrix_layout, minbatch, nthreads, partitioner, thread_type)
    end
end

matrix_layout(c::ParallelCSRContext) = c.matrix_layout
function initialize_context!(context::ParallelCSRContext, domain, system, formulation)
    context
end

nthreads(ctx::ParallelCSRContext) = ctx.nthreads
minbatch(ctx::ParallelCSRContext) = ctx.minbatch

function thread_type(context::ParallelCSRContext)
    return context.thread_type
end

function build_sparse_matrix(context::ParallelCSRContext, I, J, V, n, m)
    return static_sparsity_sparse(I, J, V, n, m, nthreads = nthreads(context), minbatch = minbatch(context))
end
