
"""
Damped Jacobi preconditioner on CPU
"""
mutable struct JacobiPreconditioner <: DiagonalPreconditioner
    factor
    dim::Tuple{Int64, Int64}
    w::Float64
    minbatch::Int64
    function JacobiPreconditioner(; w = 2.0/3.0, minbatch = 1000)
        new(nothing, (1, 1), w, minbatch)
    end
end

@inline function diagonal_precond(A, i, jac::JacobiPreconditioner)
    @inbounds A_ii = A[i, i]
    return jac.w*inv(A_ii)
end
