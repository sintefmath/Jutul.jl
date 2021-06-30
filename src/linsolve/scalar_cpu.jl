mutable struct LUSolver
    F
    reuse_memory::Bool
    check::Bool
    max_size
    function LUSolver(; reuse_memory = true, check = true, max_size = 50000)
        new(nothing, reuse_memory, check, max_size)
    end
end

function solve!(sys, solver::LUSolver)
    if length(sys.dx) > solver.max_size
        error("System too big for LU solver. You can increase max_size at your own peril.")
    end
    J = sys.jac
    r = sys.r
    if !solver.reuse_memory
        F = lu(J)
    else
        if isnothing(solver.F)
            solver.F = lu(J)
        else
            lu!(solver.F, J)
        end
        F = solver.F
    end

    sys.dx .= -(F\r)
    @assert all(isfinite, sys.dx) "Linear solve resulted in non-finite values."
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
