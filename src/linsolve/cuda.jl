
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

