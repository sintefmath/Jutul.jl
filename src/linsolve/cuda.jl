
@inline function get_nzval(jac::AbstractCuSparseMatrix)
    # Why does CUDA and Base differ on capitalization?
    return jac.nzVal
end


function build_jacobian(sparse_arg, context::SingleCUDAContext, layout)
    # TODO: Fix me.

    @assert sparse_arg.layout == layout
    I, J, n, m = ijnm(sparse_arg)
    bz = block_size(sparse_arg)
    Jt = jacobian_eltype(context, layout, bz)
    Ft = float_type(context)
    It = index_type(context)

    V = zeros(Jt, length(I))
    jac_cpu = sparse(It.(I), It.(J), V, n, m)
    display(typeof(jac_cpu))
    jac = CUDA.CUSPARSE.CuSparseMatrixCSC{Ft}(jac_cpu)

    nzval = get_nzval(jac)
    if Ft == Jt
        V_buf = nzval
    else
        V_buf = reinterpret(reshape, Ft, nzval)
    end
    return (jac, V_buf, bz)
end

#
function transfer(context::SingleCUDAContext, lsys::LinearizedSystem)
    F_t = float_type(context)
    I_t = index_type(context)
    
    # I, J, V, n, m = sparse_arg

    A = lsys.jac
    n = size(A, 1)

    # A = sparse(I_t.(I), I_t.(J), F_t.(V), I_t(n), I_t(m))
    jac = CUDA.CUSPARSE.CuSparseMatrixCSC{F_t}(A)

    V_buf = get_nzval(jac)

    r = CuArray{F_t}(undef, n)
    dx = CuArray{F_t}(undef, n)

    r_buf = r
    dx_buf = dx
    return LinearizedSystem(jac, r, dx, V_buf, r_buf, dx_buf, lsys.matrix_layout)
end


# CUDA solvers
mutable struct CuSparseSolver
    method
    reltol
    storage
end

function CuSparseSolver(method = "Chol", reltol = 1e-6)
    CuSparseSolver(method, reltol, nothing)
end

function solve!(sys::LinearizedSystem, solver::CuSparseSolver)
    J = sys.jac
    r = sys.r
    n = length(r)

    t_solve = @elapsed begin
        prec = ilu02(J, 'O')
        
        function ldiv!(y, prec, x)
            # Perform inversion of upper and lower part of ILU preconditioner
            copyto!(y, x)
            sv2!('N', 'L', 'N', 1.0, prec, y, 'O')
            sv2!('N', 'U', 'U', 1.0, prec, y, 'O')
            return y
        end
        
        y = similar(r)
        T = eltype(r)
        op = LinearOperator(T, n, n, false, false, x -> ldiv!(y, prec, x))
        
        rt = convert(eltype(r), solver.reltol)
        (x, stats) = dqgmres(J, r, M = op, rtol = rt, verbose = 0, itmax=20)
    end
    @debug "Solved linear system to with message '$(stats.status)' in $t_solve seconds."
    sys.dx .= -x
end

