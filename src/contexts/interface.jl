export thread_batch
thread_batch(::Any) = 1000

function jacobian_eltype(context, layout, block_size)
    return float_type(context)
end

function r_eltype(context, layout, block_size)
    return float_type(context)
end

function jacobian_eltype(context::CPUJutulContext, layout::BlockMajorLayout, block_size)
    return SMatrix{block_size..., float_type(context), prod(block_size)}
end

function r_eltype(context::CPUJutulContext, layout::BlockMajorLayout, block_size)
    return SVector{block_size[1], float_type(context)}
end

function build_sparse_matrix(context, I, J, V, n, m)
    return sparse(I, J, V, n, m)
end
