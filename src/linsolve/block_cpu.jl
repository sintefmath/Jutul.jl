function block_mul!(res, jac, Vt, x, α, β::T) where T
    @tic "spmv (block)" begin
        as_svec = (x) -> reinterpret(Vt, x)
        res_v = as_svec(res)
        x_v = as_svec(x)
        mul!(res_v, jac, x_v, α, β)
    end
    return res
end

function get_mul!(sys::LinearizedSystem{BlockMajorLayout})
    jac = sys.jac
    Vt = eltype(sys.r)
    return (res, x, α, β) -> block_mul!(res, jac, Vt, x, α, β)
end

function vector_residual(sys::LinearizedSystem{BlockMajorLayout})
    n = length(sys.r_buffer)
    return reshape(sys.r_buffer, n)
end

function update_dx_from_vector!(sys::LinearizedSystem{BlockMajorLayout}, dx)
    sys.dx_buffer .= -reshape(dx, size(sys.dx_buffer))
end

function block_size(lsys::LinearizedSystem{S}) where {S <: BlockMajorLayout}
    return size(lsys.r_buffer, 1)
end
