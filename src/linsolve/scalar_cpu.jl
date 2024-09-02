
"""
    LUSolver(; reuse_memory = true, check = true, max_size = 50000)

Direct solver that calls `lu` directly. Direct solvers are highly accurate, but
are costly in terms of memory usage and execution speed for larger systems.
"""
mutable struct LUSolver
    F
    reuse_memory::Bool
    check::Bool
    max_size
    function LUSolver(; reuse_memory = true, check = true, max_size = 50000)
        new(nothing, reuse_memory, check, max_size)
    end
end

function linear_solve!(sys, solver::LUSolver, arg...;dx = sys.dx, r = sys.r, kwargs...)
    if length(sys.dx) > solver.max_size
        error("System too big for LU solver. You can increase max_size at your own peril.")
    end
    J = sys.jac
    #r = sys.r
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

    dx .= -(F\r)
    @assert all(isfinite, sys.dx) "Linear solve resulted in non-finite values."
    return linear_solve_return()
end
