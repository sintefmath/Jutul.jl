function block_mul!(res, jac, Vt, N, x, α, β::T) where T
    @tic "spmv (block)" begin
        as_svec = (x) -> unsafe_reinterpret(Vt, x, length(x) ÷ N)
        res_v = as_svec(res)
        x_v = as_svec(x)
        mul!(res_v, jac, x_v, α, β)
    end
    return res
end

function get_mul!(sys::LinearizedSystem{BlockMajorLayout})
    jac = sys.jac
    Vt = eltype(sys.r)
    N = size(Vt, 1)
    Vt_typed = Val(Vt)
    return (res, x, α, β) -> block_mul!(res, jac, Vt_typed, N, x, α, β)
end

function vector_residual(sys::LinearizedSystem{BlockMajorLayout})
    n = length(sys.r_buffer)
    return reshape(sys.r_buffer, n)
end

function update_dx_from_vector!(sys::LinearizedSystem{BlockMajorLayout}, dx_from_solver; dx = sys.dx_buffer)
    dx .= -reshape(dx_from_solver, size(dx))
end

function block_size(lsys::LinearizedSystem{S}) where {S <: BlockMajorLayout}
    return size(lsys.r_buffer, 1)
end
