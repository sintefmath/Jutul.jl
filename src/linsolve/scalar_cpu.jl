"""
    LUSolver(; reuse_memory = true, check = true, max_size = 50000, backend = :umfpack, klu_max_size = 10000, klu_max_fill = 8.0)

Direct solver based on a sparse LU factorization of the Jacobian. Direct
solvers are highly accurate, but are costly in terms of memory usage and
execution speed for larger systems.

`backend` selects the sparse LU implementation:

- `:umfpack` (default): `lu` from SparseArrays.
- `:klu`: KLU from the KLU.jl package, which must be loaded (`using KLU`). KLU
  refactorizes small systems with little fill-in several times faster than
  UMFPACK and does not allocate per factorization, but it is much slower when
  the factors fill in (for example 3D problems with more than about a thousand
  unknowns).
- `:auto`: KLU when it is loaded, the Jacobian is a `SparseMatrixCSC`
  with `Float64` entries, the system has at most `klu_max_size` unknowns and the
  first KLU factorization has a fill ratio `nnz(L + U) / nnz(J)` of at most
  `klu_max_fill`. Otherwise UMFPACK. The choice is made at the first solve and
  stored in `backend`.
"""
mutable struct LUSolver
    F
    reuse_memory::Bool
    check::Bool
    max_size
    backend::Symbol
    klu_max_size::Int
    klu_max_fill::Float64
    function LUSolver(;
            reuse_memory = true,
            check = true,
            max_size = 50000,
            backend = :umfpack,
            klu_max_size = 10000,
            klu_max_fill = 8.0
        )
        backend in (:auto, :klu, :umfpack) || throw(ArgumentError("backend must be :auto, :klu or :umfpack, was :$backend"))
        return new(nothing, reuse_memory, check, max_size, backend, klu_max_size, klu_max_fill)
    end
end

function linear_solve!(sys, solver::LUSolver, arg...; dx = sys.dx, r = sys.r, kwargs...)
    if length(dx) > solver.max_size
        error("System too big for LU solver. You can increase max_size at your own peril.")
    end
    F = factorize_jacobian!(solver, sys.jac)
    ldiv!(dx, F, r)
    @. dx = -dx
    if solver.check
        all(isfinite, dx) || error("Linear solve resulted in non-finite values.")
    end
    return linear_solve_return()
end

"""
    factorize_jacobian!(solver::LUSolver, J)

Factorize `J` with the backend of `solver`. With `reuse_memory`, a stored
factorization from the same backend is refactorized in place, reusing its
symbolic analysis since the sparsity pattern does not change.
"""
function factorize_jacobian!(solver::LUSolver, J)
    F = solver.F
    if solver.reuse_memory && lu_backend(F) == solver.backend
        refactorize_jacobian!(F, J)
    else
        F = factorize_jacobian(solver, J)
        if solver.reuse_memory
            solver.F = F
        end
    end
    return F
end

function factorize_jacobian(solver::LUSolver, J)
    backend = solver.backend
    if backend == :auto
        # Try KLU once for small systems and keep it unless the factors fill in
        # too much for KLU to stay competitive with UMFPACK.
        if klu_is_applicable(solver, J)
            F = factorize_jacobian(Val(:klu), J)
            if lu_fill_ratio(F, J) <= solver.klu_max_fill
                solver.backend = :klu
                return F
            end
        end
        backend = :umfpack
        solver.backend = backend
    end
    return factorize_jacobian(Val(backend), J)
end

function klu_is_applicable(solver::LUSolver, J)
    return check_klu_availability(throw = false) &&
        J isa SparseMatrixCSC{Float64} &&
        size(J, 1) <= solver.klu_max_size
end

factorize_jacobian(::Val{:umfpack}, J) = lu(J)
refactorize_jacobian!(F, J) = lu!(F, J)

# Backend that produced a factorization. Anything returned by `lu` counts as
# UMFPACK; the KLU extension adds a method for its factorization type.
lu_backend(F) = :umfpack
lu_backend(::Nothing) = :none
