"""
Linearized system for matrix with block layout
"""
function LinearizedSystem(sparse_arg, context, layout::BlockMajorLayout)
    I, J, V_buf, n, m = sparse_arg
    nb = size(V_buf, 1)
    bz = Int(sqrt(nb))
    @assert bz ≈ round(sqrt(nb)) "Buffer had $nb rows which is not square divisible."
    @assert size(V_buf, 2) == length(I) == length(J)
    @assert n == m "Expected square system. Recieved $n (eqs) by $m (variables)."
    r_buf = zeros(bz, n)
    dx_buf = zeros(bz, n)

    float_t = eltype(V_buf)
    mt = SMatrix{bz, bz, float_t, bz*bz}
    vt = SVector{bz, float_t}
    V = zeros(mt, length(I))
    r = reinterpret(reshape, vt, r_buf)
    dx = reinterpret(reshape, vt, dx_buf)

    jac = sparse(I, J, V, n, m)
    nzval = get_nzval(jac)
    V_buf = reinterpret(reshape, Float64, nzval)
    return LinearizedSystem(jac, r, dx, V_buf, r_buf, dx_buf, layout)
end


function block_size(lsys::LinearizedSystem{S}) where {S <: BlockMajorLayout}
    return size(lsys.r_buffer, 1)
end

struct BlockDQGMRES end

function solve!(sys::LinearizedSystem, solver::BlockDQGMRES)
    # Simple block solver for testing. Not efficiently implemented.
    n = length(sys.r_buffer)

    r = reshape(sys.r_buffer, n)
    jac = sys.jac

    Vt = eltype(sys.r)
    Mt = eltype(jac)

    as_svec = (x) -> reinterpret(Vt, x)
    # as_smat = (x) -> reinterpret(Mt, x)
    # as_float = (x) -> reinterpret(Float64, x)

    function op_mult!(res, x, α, β::T) where T
        res_v = as_svec(res)
        x_v = as_svec(x)
        if β == zero(T)
            mul!(res_v, jac, x_v)
            if α != one(T)
                lmul!(α, res_v)
                # @. res_v = α*res_v
            end
            # res_v .= α.*jac*x_v
        else
            res_v .= α.*jac*x_v + β.*res_v
        end
    end

    opA = LinearOperator(Float64, n, n, false, false, op_mult!)
    (x, stats) = dqgmres(opA, r)
    sys.dx_buffer .= -reshape(x, size(sys.dx_buffer))
end
