"""
Jacobian for matrix with block layout
"""
function build_jacobian(sparse_arg, context, layout::BlockMajorLayout)
    I, J, V_buf, n, m = sparse_arg
    nb = size(V_buf, 1)
    bz = Int(sqrt(nb))
    @assert bz ≈ round(sqrt(nb)) "Buffer had $nb rows which is not square divisible."
    @assert size(V_buf, 2) == length(I) == length(J)
    @assert n == m "Expected square system. Recieved $n (eqs) by $m (variables)."

    float_t = eltype(V_buf)
    mt = SMatrix{bz, bz, float_t, bz*bz}
    V = zeros(mt, length(I))
    jac = sparse(I, J, V, n, m)
    nzval = get_nzval(jac)
    V_buf = reinterpret(reshape, Float64, nzval)
    return (jac, V_buf, bz)
end

function get_jacobian_vector(n, context, layout::BlockMajorLayout, v = nothing, bz = 1)
    if isnothing(v)
        v_buf = zeros(bz, n)
    end
    float_t = float_type(context)
    vt = SVector{bz, float_t}
    v = reinterpret(reshape, vt, v_buf)

    return (v, v_buf)
end

function get_mul!(sys::LinearizedSystem{BlockMajorLayout})
    jac = sys.jac

    Vt = eltype(sys.r)
    # Mt = eltype(jac)

    as_svec = (x) -> reinterpret(Vt, x)
    # as_smat = (x) -> reinterpret(Mt, x)
    # as_float = (x) -> reinterpret(Float64, x)

    function block_mul!(res, x, α, β::T) where T
        res_v = as_svec(res)
        x_v = as_svec(x)
        if β == zero(T)
            mul!(res_v, jac, x_v)
            if α != one(T)
                lmul!(α, res_v)
            end
        else
            # TODO: optimize me like the three argument version.
            res_v .= α.*jac*x_v + β.*res_v
        end
    end
    return block_mul!
end

function vector_residual(sys::LinearizedSystem{BlockMajorLayout})
    n = length(sys.r_buffer)
    r = reshape(sys.r_buffer, n)

end

function update_dx_from_vector!(sys::LinearizedSystem{BlockMajorLayout}, dx)
    sys.dx_buffer .= -reshape(dx, size(sys.dx_buffer))
end

function block_size(lsys::LinearizedSystem{S}) where {S <: BlockMajorLayout}
    return size(lsys.r_buffer, 1)
end
