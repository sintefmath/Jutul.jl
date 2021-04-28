export LinearizedSystem, solve!, AMGSolver, transfer

using IterativeSolvers, AlgebraicMultigrid

struct LinearizedSystem
    jac
    r
    dx
end

function LinearizedSystem(context::TervContext, jac, r, dx)
    return LinearizedSystem(jac, r, dx)
end

function transfer(context::SingleCUDAContext, lsys::LinearizedSystem)
    F = context.float_t
    # Transfer types
    r = CuArray{F}(lsys.r)
    dx = CuArray{F}(lsys.dx)
    jac = CUDA.CUSPARSE.CuSparseMatrixCSC{F}(lsys.jac)
    return LinearizedSystem(jac, r, dx)
end

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

function solve!(sys::LinearizedSystem, linsolve = nothing)
    if isnothing(linsolve)
        sys.dx .= -(sys.jac\sys.r)
    else
        solve!(sys, linsolve)
    end
end
