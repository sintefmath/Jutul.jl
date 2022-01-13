import LinearAlgebra.ldiv!, LinearAlgebra.\, SparseArrays.nnz

# Note: This is a local copy of https://github.com/mcovalt/ILUZero.jl authored by Matt Covalt under the MIT license.
# The license file accompanies this local copy. The copy is only required pending the release of 2.1.0 since the
# registry only contains 1.0.0.

# ILU0 type definition
struct ILU0Precon{T <: Any, N <: Integer, M<: Any} <: Factorization{T}
    m::N
    n::N
    l_colptr::Vector{N}
    l_rowval::Vector{N}
    l_nzval::Vector{T}
    u_colptr::Vector{N}
    u_rowval::Vector{N}
    u_nzval::Vector{T}
    l_map::Vector{N}
    u_map::Vector{N}
    wrk::Vector{M}
end

# Allocates ILU0Precon type
function ILU0Precon(A::SparseMatrixCSC{T,N}, b_type = T) where {T <: Any,N <: Integer}
    m, n = size(A)

    # Determine number of elements in lower/upper
    lnz = 0
    unz = 0
    @inbounds for i = 1:n
        for j = A.colptr[i]:A.colptr[i + 1] - 1
            if A.rowval[j] > i
                lnz += 1
            else
                unz += 1
            end
        end
    end

    # Preallocate variables
    l_colptr = zeros(N, n + 1)
    u_colptr = zeros(N, n + 1)
    l_nzval = zeros(T, lnz)
    u_nzval = zeros(T, unz)
    l_rowval = zeros(Int64, lnz)
    u_rowval = zeros(Int64, unz)
    l_map = Vector{N}(undef, lnz)
    u_map = Vector{N}(undef, unz)
    wrk = zeros(b_type, n)
    l_colptr[1] = 1
    u_colptr[1] = 1

    # Map elements of A to lower and upper triangles, fill out colptr, and fill out rowval
    lit = 1
    uit = 1
    @inbounds for i = 1:n
        l_colptr[i + 1] = l_colptr[i]
        u_colptr[i + 1] = u_colptr[i]
        for j = A.colptr[i]:A.colptr[i + 1] - 1
            if A.rowval[j] > i
                l_colptr[i + 1] += 1
                l_rowval[lit] = A.rowval[j]
                l_map[lit] = j
                lit += 1
            else
                u_colptr[i + 1] += 1
                u_rowval[uit] = A.rowval[j]
                u_map[uit] = j
                uit += 1
            end
        end
    end

    return ILU0Precon(m, n, l_colptr, l_rowval, l_nzval, u_colptr, u_rowval, u_nzval, l_map, u_map, wrk)
end

# Updates ILU0Precon type in-place based on matrix A
function ilu0!(LU::ILU0Precon{T,N}, A::SparseMatrixCSC{T,N}) where {T <: Any,N <: Integer}
    m = LU.m
    n = LU.n
    l_colptr = LU.l_colptr
    l_rowval = LU.l_rowval
    l_nzval = LU.l_nzval
    u_colptr = LU.u_colptr
    u_rowval = LU.u_rowval
    u_nzval = LU.u_nzval
    l_map = LU.l_map
    u_map = LU.u_map

    # Redundant data or better speed... speed is chosen, but this might be changed.
    # This shouldn't be inbounded either.
    for i = 1:length(l_map)
        l_nzval[i] = A.nzval[l_map[i]]
    end
    for i = 1:length(u_map)
        u_nzval[i] = A.nzval[u_map[i]]
    end


    @inbounds for i = 1:m - 1
        m_inv = inv(u_nzval[u_colptr[i + 1] - 1])
        for j = l_colptr[i]:l_colptr[i + 1] - 1
            l_nzval[j] = l_nzval[j] * m_inv
        end
        for j = u_colptr[i + 1]:u_colptr[i + 2] - 2
            multiplier = u_nzval[j]
            qn = j + 1
            rn = l_colptr[i + 1]
            pn = l_colptr[u_rowval[j]]
            while pn < l_colptr[u_rowval[j] + 1] && l_rowval[pn] <= i + 1
                while qn < u_colptr[i + 2] && u_rowval[qn] < l_rowval[pn]
                    qn += 1
                end
                if qn < u_colptr[i + 2] && l_rowval[pn] == u_rowval[qn]
                    u_nzval[qn] -= l_nzval[pn] * multiplier
                end
                pn += 1
            end
            while pn < l_colptr[u_rowval[j] + 1]
                while rn < l_colptr[i + 2] && l_rowval[rn] < l_rowval[pn]
                    rn += 1
                end
                if rn < l_colptr[i + 2] && l_rowval[pn] == l_rowval[rn]
                    l_nzval[rn] -= l_nzval[pn] * multiplier
                end
                pn += 1
            end
        end
    end
    return
end

# Constructs ILU0Precon type based on matrix A
function ilu0(A::SparseMatrixCSC{T,N}, arg...) where {T <: Any,N <: Integer}
    LU = ILU0Precon(A, arg...)
    ilu0!(LU, A)
    return LU
end

# Solves L\b and stores the solution in y
function forward_substitution!(y, LU::ILU0Precon{T,N,M}, b) where {T, N <: Integer, M}
    n = LU.n
    l_colptr = LU.l_colptr
    l_rowval = LU.l_rowval
    l_nzval  = LU.l_nzval

    for i in eachindex(y)
        y[i] = zero(M)
    end

    @inbounds for i = 1:n
        y[i] += b[i]
        for j = l_colptr[i]:l_colptr[i + 1] - 1
            y[l_rowval[j]] -= l_nzval[j] * y[i]
        end
    end
    return y
end

# Solves U\y and stores the solution in x
function backward_substitution!(x, LU::ILU0Precon, y)
    n = LU.n
    u_colptr = LU.u_colptr
    u_rowval = LU.u_rowval
    u_nzval = LU.u_nzval

    wrk = LU.wrk
    wrk .= y

    @inbounds for i = n:-1:1
        x[i] = u_nzval[u_colptr[i + 1] - 1] \ wrk[i]
        for j = u_colptr[i]:u_colptr[i + 1] - 2
            wrk[u_rowval[j]] -= u_nzval[j] * x[i]
        end
    end
    return x
end

# Solves LU\b overwriting x
function ldiv!(x::AbstractVector{M}, LU::ILU0Precon{T,N,M}, b::AbstractVector{M}) where {T,N <: Integer,M}
    (length(b) == LU.n) || throw(DimensionMismatch())
    n = LU.n
    l_colptr = LU.l_colptr
    l_rowval = LU.l_rowval
    l_nzval = LU.l_nzval
    u_colptr = LU.u_colptr
    u_rowval = LU.u_rowval
    u_nzval = LU.u_nzval
    wrk = LU.wrk

    for i in eachindex(wrk)
        wrk[i] = zero(M)
    end

    @inbounds for i = 1:n
        wrk[i] += b[i]
        for j = l_colptr[i]:l_colptr[i + 1] - 1
            wrk[l_rowval[j]] -= l_nzval[j] * wrk[i]
        end
    end
    @inbounds for i = n:-1:1
        x[i] = u_nzval[u_colptr[i + 1] - 1] \ wrk[i]
        for j = u_colptr[i]:u_colptr[i + 1] - 2
            wrk[u_rowval[j]] -= u_nzval[j] * x[i]
        end
    end
    return x
end

# Solves LU\b overwriting b
function ldiv!(LU::ILU0Precon{T,N}, b::AbstractVector{T}) where {T <: Any,N <: Integer}
    ldiv!(b, LU, b)
end

# Returns LU\b
function \(LU::ILU0Precon{T,N}, b::Vector{T}) where {T <: Any,N <: Integer}
    length(b) == LU.n || throw(DimensionMismatch())
    x = zeros(T, length(b))
    ldiv!(x, LU, b)
end

# Returns the number of nonzero
function nnz(LU::ILU0Precon{T,N}) where {T <: Any,N <: Integer}
    return length(LU.l_nzval) + length(LU.u_nzval)
end

