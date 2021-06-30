export LinearizedSystem, solve!, AMGSolver, CuSparseSolver, transfer, BlockDQGMRES, LUSolver

using SparseArrays, LinearOperators, StaticArrays
using IterativeSolvers, Krylov, AlgebraicMultigrid
using CUDA, CUDA.CUSPARSE

struct LinearizedSystem{L}
    jac
    r
    dx
    jac_buffer
    r_buffer
    dx_buffer
    matrix_layout::L
end

function LinearizedSystem(sparse_arg, context, layout)
    I, J, V, n, m = sparse_arg
    @assert n == m "Expected square system. Recieved $n (eqs) by $m (variables)."
    r = zeros(n)
    dx = zeros(n)
    jac = sparse(I, J, V, n, m)

    jac_buf, dx_buf, r_buf = get_nzval(jac), dx, r

    return LinearizedSystem(jac, r, dx, jac_buf, r_buf, dx_buf, layout)
end

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


@inline function get_nzval(jac)
    return jac.nzval
end

@inline function get_nzval(jac::AbstractCuSparseMatrix)
    # Why does CUDA and Base differ on capitalization?
    return jac.nzVal
end

function block_size(lsys::LinearizedSystem) 1 end

function block_size(lsys::LinearizedSystem{S}) where {S <: BlockMajorLayout}
    return size(lsys.r_buffer, 1)
end

function solve!(sys::LinearizedSystem, linsolve::Nothing)
    solve!(sys)
end


function solve!(sys)
    if length(sys.dx) > 50000
        error("System too big for default direct solver.")
    end
    J = sys.jac
    r = sys.r
    sys.dx .= -(J\r)
    @assert all(isfinite, sys.dx) "Linear solve resulted in non-finite values."
end

