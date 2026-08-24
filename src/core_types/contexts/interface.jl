export minbatch

function minbatch(x::Any)
    nt = nthreads(x)
    if nt == 1
        nb = typemax(Int)
    else
        nb = 1000
    end
    return nb
end

function nthreads(::Any)
    Threads.nthreads()
end

minbatch(x, n) = max(n ÷ nthreads(x), minbatch(x))

function thread_type(context::JutulContext)
    return :threads
end

function threaded_loop(F, N, context::JutulContext)
    threads = thread_type(context)
    if threads == :threads
        Threads.@threads for i in 1:N
            F(i)
        end
    elseif threads == :batch
        @batch for i in 1:N
            F(i)
        end
    elseif threads == :serial
        for i in 1:N
            F(i)
        end
    else
        throw(ArgumentError("Unknown thread_type $threads"))
    end
end

function threaded_loop_minbatch(F, N, context::JutulContext, minbatch::Int = minbatch(context))
    N_threads = nthreads(context)
    N_batches = clamp(N_threads ÷ minbatch, 1, N)
    threads = thread_type(context)
    if N_batches == 1 || threads == :serial
        for i in 1:N
            F(i)
        end
    else
        if threads == :threads
            Threads.@threads for batch in 1:N_batches
                for i in load_balanced_interval(batch, N, N_batches)
                    F(i)
                end
            end
        elseif threads == :batch
            @batch minbatch = minbatch for i in 1:N
                F(i)
            end
        else
            throw(ArgumentError("Unknown thread_type $threads"))
        end
    end
end

function jacobian_eltype(context, layout, block_size)
    return float_type(context)
end

function r_eltype(context, layout, block_size)
    return float_type(context)
end

function jacobian_eltype(context::CPUJutulContext, layout::BlockMajorLayout, block_size)
    F = float_type(context)
    if block_size[1] == block_size[2] == 1
        M = Float64
    else
        M = SMatrix{block_size..., F, prod(block_size)}
    end
    return M
end

function r_eltype(context::CPUJutulContext, layout::BlockMajorLayout, block_size)
    F = float_type(context)
    if block_size == 1
        V = F
    else
        V = SVector{block_size, F}
    end
    return V
end

function build_sparse_matrix(context, I, J, V, n, m)
    return sparse(I, J, V, n, m)
end

