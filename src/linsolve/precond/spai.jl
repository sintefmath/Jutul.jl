"""
Sparse Approximate Inverse preconditioner of lowest order -- SPAI(0)
"""
mutable struct SPAI0Preconditioner <: DiagonalPreconditioner
    factor
    buffer
    dim::Tuple{Int64, Int64}
    minbatch::Int64
    function SPAI0Preconditioner(; minbatch = 1000)
        new(nothing, nothing, (1, 1), minbatch)
    end
end

function diagonal_precond!(Diag, A::SparseMatrixCSC, spai::SPAI0Preconditioner)
    D = Diag.D
    if isnothing(spai.buffer)
        spai.buffer = zeros(length(D))
    end
    buf = spai.buffer
    @inbounds for i in eachindex(D)
        D[i] = zero(eltype(D))
    end
    rows = rowvals(A)
    vals = nonzeros(A)
    for col in axes(A, 2)
        @inbounds for p in nzrange(A, col)
            row = rows[p]
            val = vals[p]
            nv = zero(eltype(val))
            for v in val
                nv += v*adjoint(v)
            end
            buf[row] += nv
        end
    end

    @inbounds for i in eachindex(D)
        D[i] = inv(buf[i])*A[i, i]
    end
end

function diagonal_precond!(Diag, A::StaticSparsityMatrixCSR, spai::SPAI0Preconditioner)
    D = Diag.D
    cols = colvals(A)
    vals = nonzeros(A)
    T = eltype(vals)
    for row in axes(A, 1)
        norm_sum = zero(eltype(T))
        A_ii = zero(T)
        @inbounds for p in nzrange(A, row)
            col = cols[p]
            val = vals[p]
            for v in val
                norm_sum += v*adjoint(v)
            end
            if col == row
                A_ii = val
            end
        end
        @inbounds D[row] = inv(norm_sum)*A_ii
    end
end


