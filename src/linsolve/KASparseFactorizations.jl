module KASparseFactorizations

using Graphs: SimpleGraph, add_edge!, connected_components
using KernelAbstractions
using LinearAlgebra
using SparseArrays: nnz

using ..Jutul: StaticSparsityMatrixCSR
using ..KAPreconditioners: backend_copy, matrix_backend, same_backend

struct BlockSparseLU{V, I, B, R, C, Ti}
    factors::V
    work::V
    pivots::I
    status::I
    order::I
    blockptr::I
    factorptr::I
    source_map::I
    backend::B
    source_rowptr::R
    source_colval::C
    rowptr::Vector{Ti}
    colval::Vector{Ti}
    host_status::Vector{Ti}
end

block_workgroupsize(backend, nblocks) = 1

"""
    try_factorize(A::StaticSparsityMatrixCSR)

Analyze the structural components of a square CSR matrix and compute a
pivoted LU when it decomposes into independent blocks of at most 32 rows.
The row limit applies to each block. The number of blocks is limited by the
index type and available memory, and a block of size `b` stores `b^2` factor
entries. Return `nothing` for patterns outside this path. Symbolic analysis
runs on the host once. `refactorize!` and `ldiv!` run on the matrix backend.
"""
function try_factorize(A::StaticSparsityMatrixCSR)
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("sparse LU requires a square matrix"))
    T = eltype(A.nzval)
    T <: Union{Float32, Float64, ComplexF32, ComplexF64} || return nothing
    n > 1 || return nothing

    rowptr, colval = Array(A.rowptr), Array(A.colval)
    graph = SimpleGraph(n)
    for i in 1:n, k in rowptr[i]:(rowptr[i + 1] - 1)
        j = colval[k]
        i != j && add_edge!(graph, i, j)
    end
    components = connected_components(graph)
    sizes = length.(components)
    length(sizes) > 1 && maximum(sizes) <= 32 || return nothing

    Ti = eltype(rowptr)
    order = reduce(vcat, sort!.(components))
    inverse_order = invperm(order)
    blockptr = cumsum(vcat(1, sizes))
    factorptr = cumsum(vcat(1, sizes .^ 2))
    factorptr[end] <= typemax(Ti) || return nothing
    source_map = zeros(Ti, factorptr[end] - 1)
    for block in eachindex(sizes)
        lo, hi = blockptr[block], blockptr[block + 1] - 1
        for packed_row in lo:hi
            row = order[packed_row]
            start = factorptr[block] + (packed_row - lo) * sizes[block]
            for k in rowptr[row]:(rowptr[row + 1] - 1)
                packed_col = inverse_order[colval[k]]
                source_map[start + packed_col - lo] = k
            end
        end
    end

    backend = matrix_backend(A)
    blocks = length(sizes)
    F = BlockSparseLU(
        KernelAbstractions.allocate(backend, T, length(source_map)),
        KernelAbstractions.allocate(backend, T, n),
        KernelAbstractions.allocate(backend, Ti, n),
        KernelAbstractions.allocate(backend, Ti, blocks),
        backend_copy(backend, Ti.(order)),
        backend_copy(backend, Ti.(blockptr)),
        backend_copy(backend, Ti.(factorptr)),
        backend_copy(backend, source_map),
        backend, A.rowptr, A.colval, rowptr, colval, Vector{Ti}(undef, blocks)
    )
    return refactorize!(F, A)
end

function refactorize!(F::BlockSparseLU, A::StaticSparsityMatrixCSR)
    n = length(F.order)
    size(A) == (n, n) && nnz(A) == length(F.colval) ||
        throw(ArgumentError("sparse LU refactorization requires the same CSR pattern"))
    same_backend(F.backend, matrix_backend(A)) && eltype(F.factors) === eltype(A.nzval) ||
        throw(ArgumentError("sparse LU refactorization requires the same backend and value type"))
    if !(F.source_rowptr === A.rowptr && F.source_colval === A.colval)
        F.rowptr == Array(A.rowptr) && F.colval == Array(A.colval) ||
            throw(ArgumentError("sparse LU refactorization requires the same CSR pattern"))
    end

    nblocks = length(F.status)
    factor! = factor_blocks_kernel!(F.backend, block_workgroupsize(F.backend, nblocks))
    event = factor!(
        F.factors, F.pivots, F.status, A.nzval, F.order, F.blockptr,
        F.factorptr, F.source_map; ndrange = nblocks
    )
    isnothing(event) || wait(event)
    copyto!(F.host_status, F.status)
    failed = findfirst(!iszero, F.host_status)
    isnothing(failed) || throw(SingularException(Int(F.host_status[failed])))
    return F
end

@kernel function factor_blocks_kernel!(
        factors, pivots, status, @Const(values), @Const(order), @Const(blockptr),
        @Const(factorptr), @Const(source_map)
    )
    block = @index(Global, Linear)
    @inbounds begin
        lo = blockptr[block] - one(eltype(blockptr))
        n = blockptr[block + 1] - blockptr[block]
        base = factorptr[block] - one(eltype(factorptr))
        status[block] = zero(eltype(status))
        for i in factorptr[block]:(factorptr[block + 1] - 1)
            j = source_map[i]
            factors[i] = iszero(j) ? zero(eltype(factors)) : values[j]
        end

        for k in 1:n
            pivot_row = k
            pivot_size = abs(factors[base + (k - 1) * n + k])
            for row in (k + 1):n
                candidate = abs(factors[base + (row - 1) * n + k])
                if candidate > pivot_size
                    pivot_size = candidate
                    pivot_row = row
                end
            end
            if iszero(pivot_size) || !isfinite(pivot_size)
                status[block] = order[lo + k]
                break
            end
            pivots[lo + k] = pivot_row
            first = base + (k - 1) * n
            if pivot_row != k
                second = base + (pivot_row - 1) * n
                for j in 1:n
                    factors[first + j], factors[second + j] =
                        factors[second + j], factors[first + j]
                end
            end
            pivot = factors[first + k]
            for row in (k + 1):n
                offset = base + (row - 1) * n
                multiplier = factors[offset + k] / pivot
                factors[offset + k] = multiplier
                for j in (k + 1):n
                    factors[offset + j] -= multiplier * factors[first + j]
                end
            end
        end
    end
end

@kernel function solve_blocks_kernel!(
        x, b, work, @Const(factors), @Const(pivots), @Const(order), @Const(blockptr),
        @Const(factorptr)
    )
    block = @index(Global, Linear)
    @inbounds begin
        lo = blockptr[block] - one(eltype(blockptr))
        n = blockptr[block + 1] - blockptr[block]
        base = factorptr[block] - one(eltype(factorptr))
        for i in 1:n
            work[lo + i] = b[order[lo + i]]
        end
        for i in 1:n
            row = pivots[lo + i]
            work[lo + i], work[lo + row] = work[lo + row], work[lo + i]
        end
        for i in 1:n
            offset = base + (i - 1) * n
            value = work[lo + i]
            for j in 1:(i - 1)
                value -= factors[offset + j] * work[lo + j]
            end
            work[lo + i] = value
        end
        for i in n:-1:1
            offset = base + (i - 1) * n
            value = work[lo + i]
            for j in (i + 1):n
                value -= factors[offset + j] * work[lo + j]
            end
            work[lo + i] = value / factors[offset + i]
            x[order[lo + i]] = work[lo + i]
        end
    end
end

function LinearAlgebra.ldiv!(x, F::BlockSparseLU, b)
    length(x) == length(F.order) == length(b) || throw(DimensionMismatch())
    nblocks = length(F.status)
    solve! = solve_blocks_kernel!(F.backend, block_workgroupsize(F.backend, nblocks))
    event = solve!(
        x, b, F.work, F.factors, F.pivots, F.order, F.blockptr, F.factorptr;
        ndrange = nblocks
    )
    isnothing(event) || wait(event)
    return x
end

end
