export LinearizedSystem, solve!, AMGSolver, CuSparseSolver, transfer, BlockDQGMRES

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

function solve!(sys::LinearizedSystem, linsolve)
    if isnothing(linsolve)
        # Fall back to default Julia direct solver
        solve!(sys)
    else
        # Use what was provided
        solve!(sys, linsolve)
    end
end

struct BlockDQGMRES end

function solve!(sys)
    if length(sys.dx) > 50000
        error("System too big for default direct solver.")
    end
    J = sys.jac
    r = sys.r
    sys.dx .= -(J\r)
    @assert all(isfinite, sys.dx) "Linear solve resulted in non-finite values."
end

function solve!(sys::LinearizedSystem, solver::BlockDQGMRES)
    # Simple block solver for testing. Not efficiently implemented.
    n = length(sys.r_buffer)

    r = reshape(sys.r_buffer, n)
    jac = sys.jac

    Vt = eltype(sys.r)
    Mt = eltype(jac)

    as_svec = (x) -> reinterpret(Vt, x)
    as_smat = (x) -> reinterpret(Mt, x)
    as_float = (x) -> reinterpret(Float64, x)

    opA = LinearOperator(Float64, n, n, false, false, x -> Vector(as_float(jac*as_svec(x))))
    (x, stats) = dqgmres(opA, r)

    sys.dx_buffer .= -reshape(x, size(sys.dx_buffer))
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

# AMG solver (Julia-native)
mutable struct AMGSolver 
    method
    reltol
    preconditioner
    hierarchy
end

function AMGSolver(method = "RugeStuben", reltol = 1e-6)
    AMGSolver(method, reltol, nothing, nothing)
end

function solve!(sys::LinearizedSystem, solver::AMGSolver)
    if isnothing(solver.preconditioner)
        @debug string("Setting up preconditioner ", solver.method)
        if solver.method == "RugeStuben"
            t_amg = @elapsed solver.hierarchy = ruge_stuben(sys.jac)
        else
            t_amg = @elapsed solver.hierarchy = smoothed_aggregation(sys.jac)
        end
        @debug "Set up AMG in $t_amg seconds."
        solver.preconditioner = aspreconditioner(solver.hierarchy)
    end
    t_solve = @elapsed begin 
        gmres!(sys.dx, sys.jac, -sys.r, reltol = solver.reltol, maxiter = 20, 
                                        Pl = solver.preconditioner, verbose = false)
    end
    @debug "Solved linear system to $(solver.reltol) in $t_solve seconds."
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

