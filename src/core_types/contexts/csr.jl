export ParallelCSRContext
"""
    ParallelCSRContext()
    # 5 threads (provided that Julia was started at least --threads=5)
    ParalellCSRContext(5)
    # Use @threads :static for the threaded loops
    ParalellCSRContext(thread_type = :threads_static)
    # Use batch threading for the threaded loops
    ParalellCSRContext(thread_type = :batch)
    # Use serial loops for the threaded loops
    ParalellCSRContext(thread_type = :serial)

A context that uses a CSR sparse matrix format together with threads.
See [`secondary_scheme`](@ref) for secondary-variable scheduling.
"""
struct ParallelCSRContext <: CPUJutulContext
    matrix_layout
    minbatch::Integer
    nthreads::Integer
    partitioner::JutulPartitioner
    thread_type::Symbol
    secondary_scheme::Symbol
    function ParallelCSRContext(
            nthreads = Threads.nthreads();
            partitioner = MetisPartitioner(),
            matrix_layout = EquationMajorLayout(),
            minbatch = minbatch(nothing),
            thread_type = :batch,
            secondary_scheme = :flat
        )
        maxthreads = Threads.nthreads()
        if nthreads > maxthreads
            @warn "nthreads > Threads.nthreads() in ParallelCSRContext. Using Threads.nthreads() instead."
            nthreads = maxthreads
        end
        thread_type in (:threads, :threads_static, :batch, :serial) || throw(ArgumentError("thread_type must be :threads, :serial, :threads_static or :batch"))
        check_secondary_scheme(secondary_scheme)
        return new(matrix_layout, minbatch, nthreads, partitioner, thread_type, secondary_scheme)
    end
end

matrix_layout(c::ParallelCSRContext) = c.matrix_layout
function initialize_context!(context::ParallelCSRContext, domain, system, formulation)
    return context
end

nthreads(ctx::ParallelCSRContext) = ctx.nthreads
minbatch(ctx::ParallelCSRContext) = ctx.minbatch
secondary_scheme(ctx::ParallelCSRContext) = ctx.secondary_scheme

function Base.adjoint(c::ParallelCSRContext)
    return ParallelCSRContext(nthreads(c); matrix_layout = adjoint(matrix_layout(c)),
        minbatch = minbatch(c), partitioner = c.partitioner,
        thread_type = thread_type(c), secondary_scheme = secondary_scheme(c))
end

function thread_type(context::ParallelCSRContext)
    return context.thread_type
end

function build_sparse_matrix(context::ParallelCSRContext, I, J, V, n, m)
    return static_sparsity_sparse(I, J, V, n, m, nthreads = nthreads(context), minbatch = minbatch(context))
end
