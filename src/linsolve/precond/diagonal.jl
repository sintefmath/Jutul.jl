
function update!(jac::DiagonalPreconditioner, A, b, context)
    if isnothing(jac.factor)
        n = size(A, 1)
        D = Vector{eltype(A)}(undef, n)
        @inbounds for i in 1:n
            D[i] = A[i, i]
        end
        jac.factor = D
        d = length(b[1])
        jac.dim = d .* size(A)
    end
    D = jac.factor
    mb = minbatch(A)
    jac.minbatch = mb
    diagonal_precond!(D, A, jac)
end

function diagonal_precond!(D, A, jac)
    mb = minbatch(A)
    @batch minbatch = mb for i in eachindex(D)
        @inbounds D[i] = diagonal_precond(A, i, jac)
    end
end

function apply!(x, jac::DiagonalPreconditioner, y, arg...)
    D = jac.factor

    s = D[1]
    N = size(s, 1)
    T = eltype(s)
    Vt = SVector{N, T}
    as_svec = (x) -> reinterpret(Vt, x)

    # Solve by reinterpreting vectors to block (=SVector) vectors
    diag_parmul!(as_svec(x), D, as_svec(y), minbatch(jac))
end

function operator_nrows(jac::DiagonalPreconditioner)
    return jac.dim[1]
end

function diag_parmul!(x, D, y, mb)
    @batch minbatch = mb for i in eachindex(x, y, D)
        @inbounds x[i] = D[i]*y[i]
    end
end
