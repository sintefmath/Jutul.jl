export ParallelCSRContext
"Context that uses threads etc to accelerate loops"
struct ParallelCSRContext <: CPUJutulContext
    matrix_layout
    minbatch::Integer
    thread_division::ThreadDivision
    function ParallelCSRContext(arg...; matrix_layout = EquationMajorLayout(), minbatch = minbatch(nothing))
        new(matrix_layout, minbatch, ThreadDivision(arg...))
    end
end

matrix_layout(c::ParallelCSRContext) = c.matrix_layout
function initialize_context!(context::ParallelCSRContext, domain, system, formulation)
    tdiv = context.thread_division
    n = number_of_cells(domain)
    if length(tdiv.partition) != n
        m = nthreads(tdiv)
        partition = zeros(Int64, n);
        for i in eachindex(partition)
            partition[i] = ceil(i / (n/m))
        end
        initialize_thread_division!(tdiv, partition)
    end
    context
end

nthreads(ctx::ParallelCSRContext) = nthreads(ctx.thread_division)
minbatch(ctx::ParallelCSRContext) = ctx.minbatch

function build_sparse_matrix(context::ParallelCSRContext, I, J, V, n, m)
    return static_sparsity_sparse(I, J, V, n, m, nthreads = nthreads(context), minbatch = minbatch(context))
end
