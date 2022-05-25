export thread_batch
thread_batch(::Any) = 1000

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
