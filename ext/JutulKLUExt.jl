module JutulKLUExt
using Jutul
using LinearAlgebra: ldiv!
using SparseArrays: SparseMatrixCSC
import KLU

Jutul.check_klu_availability_impl() = true

function Jutul.linear_solve!(sys, solver::KLUSolver, arg...; dx = sys.dx, r = sys.r, kwargs...)
    if length(dx) > solver.max_size
        error("System too big for KLU solver. You can increase max_size at your own peril.")
    end
    J = sys.jac
    J isa SparseMatrixCSC || throw(ArgumentError("KLUSolver requires a SparseMatrixCSC Jacobian, got $(typeof(J))."))
    F = solver.F
    if solver.reuse_memory && !isnothing(F)
        KLU.klu!(F, J)
    else
        F = KLU.klu(J)
        if solver.reuse_memory
            solver.F = F
        end
    end
    ldiv!(dx, F, r)
    @. dx = -dx
    if solver.check
        all(isfinite, dx) || error("Linear solve resulted in non-finite values.")
    end
    return Jutul.linear_solve_return()
end

end
