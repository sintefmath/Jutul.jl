matrix_nrows(A::StaticSparsityMatrixCSR) = size(A, 1)
matrix_ncols(A::StaticSparsityMatrixCSR) = size(A, 2)
matrix_nonzeros(A::StaticSparsityMatrixCSR) = nnz(A)

function matrix_backend(A::StaticSparsityMatrixCSR)
    if isnothing(A.backend)
        return KernelAbstractions.get_backend(A.nzval)
    else
        return A.backend
    end
end

# KernelAbstractions backends describe where an array can be used. Backend
# options may still differ within the same backend type: notably, `CPU` carries
# a `static` scheduling flag while both variants operate on ordinary `Array`s.
# Compare backend types when checking storage compatibility instead of backend
# object identity.
@inline same_backend(first, second) = typeof(first) === typeof(second)

function matrix_block_size(A::StaticSparsityMatrixCSR)
    minbatch = A.minbatch
    return minbatch > 1 ? minbatch : 128
end

function synchronize_backend(backend)
    if hasmethod(KernelAbstractions.synchronize, Tuple{typeof(backend)})
        KernelAbstractions.synchronize(backend)
    end
    return nothing
end

function backend_copy(backend, source::AbstractVector{T}) where T
    destination = KernelAbstractions.allocate(backend, T, length(source))
    if source isa BitVector
        copyto!(destination, Vector{Bool}(source))
    else
        copyto!(destination, source)
    end
    return destination
end

backend_zeros(backend, ::Type{T}, n::Integer) where T =
    KernelAbstractions.zeros(backend, T, Int(n))

function csr_matrix(rowptr::AbstractVector{Ti}, colval::AbstractVector{Ti},
        nzval::AbstractVector{Tv}, nrow::Integer, ncol::Integer;
        backend = nothing, block_size::Integer = 128) where {Tv, Ti<:Integer}
    return StaticSparsityMatrixCSR(
        nzval, colval, rowptr, Int(nrow), Int(ncol), backend;
        nthreads = 1, minbatch = Int(block_size), thread_type = :serial)
end

function host_csr_from_csc(A::SparseMatrixCSC{Tv}, ::Type{Ti};
        reuse = nothing, cursor_reuse = nothing,
        block_size::Integer = 128) where {Tv, Ti<:Integer}
    nrow, ncol = size(A)
    number_of_nonzeros = nnz(A)
    max_index = max(number_of_nonzeros + 1, nrow, ncol)
    max_index <= typemax(Ti) || throw(ArgumentError(
        "$Ti cannot index this matrix; pass index_type=Int64"))

    compatible = reuse isa StaticSparsityMatrixCSR{
        Tv, Ti, <:Vector, <:Vector, <:Vector}
    if compatible
        rowptr, colval, nzval = reuse.rowptr, reuse.colval, reuse.nzval
        resize!(rowptr, nrow + 1)
        resize!(colval, number_of_nonzeros)
        resize!(nzval, number_of_nonzeros)
        fill!(rowptr, zero(Ti))
    else
        rowptr = zeros(Ti, nrow + 1)
        colval = Vector{Ti}(undef, number_of_nonzeros)
        nzval = Vector{Tv}(undef, number_of_nonzeros)
    end

    rows = rowvals(A)
    @inbounds for position in eachindex(rows)
        rowptr[Int(rows[position]) + 1] += one(Ti)
    end
    @inbounds for row in 1:nrow
        rowptr[row + 1] += rowptr[row]
    end
    @inbounds for row in eachindex(rowptr)
        rowptr[row] += one(Ti)
    end

    cursor = cursor_reuse isa Vector{Ti} ? cursor_reuse : Vector{Ti}(undef, nrow)
    resize!(cursor, nrow)
    copyto!(cursor, 1, rowptr, 1, nrow)
    source_values = nonzeros(A)
    @inbounds for column in 1:ncol
        for position in nzrange(A, column)
            row = Int(rows[position])
            destination = cursor[row]
            colval[destination] = Ti(column)
            nzval[destination] = source_values[position]
            cursor[row] += one(Ti)
        end
    end
    return StaticSparsityMatrixCSR(
        nzval, colval, rowptr, nrow, ncol, nothing;
        nthreads = 1, minbatch = Int(block_size), thread_type = :serial)
end

function copy_csc_values_to_csr!(destination::Vector{Tv},
        A::SparseMatrixCSC{Tv}, rowptr::Vector{Ti},
        cursor::Vector{Ti}) where {Tv, Ti}
    nrow, ncol = size(A)
    resize!(destination, nnz(A))
    resize!(cursor, nrow)
    copyto!(cursor, 1, rowptr, 1, nrow)
    rows = rowvals(A)
    values = nonzeros(A)
    @inbounds for column in 1:ncol
        for position in nzrange(A, column)
            row = Int(rows[position])
            destination_position = cursor[row]
            destination[destination_position] = values[position]
            cursor[row] += one(Ti)
        end
    end
    return destination
end

"Convert Julia CSC storage to Jutul's CSR representation."
function csr_matrix(A::SparseMatrixCSC{Tv}; backend = nothing,
        block_size::Integer = 128,
        index_type::Type{Ti} = Int32) where {Tv, Ti<:Integer}
    host = host_csr_from_csc(A, Ti; block_size = block_size)
    selected_backend = isnothing(backend) ? matrix_backend(host) : backend
    if selected_backend isa KernelAbstractions.CPU
        return host
    end
    return matrix_to_backend(host, selected_backend, Int(block_size))
end

function csr_matrix(A::StaticSparsityMatrixCSR;
        backend = matrix_backend(A), block_size = matrix_block_size(A))
    if backend === matrix_backend(A)
        return A
    else
        return matrix_to_backend(A, backend, Int(block_size))
    end
end

function host_prefix(source::AbstractVector{T}, n::Integer) where T
    output_length = Int(n)
    if source isa Vector
        destination = Vector{T}(undef, output_length)
        copyto!(destination, 1, source, 1, output_length)
        return destination
    else
        host = Array(source)
        resize!(host, output_length)
        return host
    end
end

function host_csr(A::StaticSparsityMatrixCSR)
    rowptr = host_prefix(A.rowptr, size(A, 1) + 1)
    colval = host_prefix(A.colval, nnz(A))
    nzval = host_prefix(A.nzval, nnz(A))
    return StaticSparsityMatrixCSR(
        nzval, colval, rowptr, size(A, 1), size(A, 2), nothing;
        nthreads = 1, minbatch = matrix_block_size(A), thread_type = :serial)
end

function host_prefix_reusing(old::Vector{T}, source::AbstractVector{T},
        n::Integer) where T
    resize!(old, Int(n))
    copyto!(old, 1, source, 1, Int(n))
    return old
end

function host_csr_reusing(A::StaticSparsityMatrixCSR{Tv, Ti}, old) where {Tv, Ti}
    if old isa StaticSparsityMatrixCSR{Tv, Ti, <:Vector, <:Vector, <:Vector}
        rowptr = host_prefix_reusing(old.rowptr, A.rowptr, size(A, 1) + 1)
        colval = host_prefix_reusing(old.colval, A.colval, nnz(A))
        nzval = host_prefix_reusing(old.nzval, A.nzval, nnz(A))
        return StaticSparsityMatrixCSR(
            nzval, colval, rowptr, size(A, 1), size(A, 2), nothing;
            nthreads = 1, minbatch = matrix_block_size(A), thread_type = :serial)
    end
    return host_csr(A)
end

function sparse_matrix(A::StaticSparsityMatrixCSR{Tv, Ti}) where {Tv, Ti}
    host = if A.rowptr isa Vector && A.colval isa Vector && A.nzval isa Vector
        A
    else
        host_csr(A)
    end
    stored_transpose = SparseMatrixCSC{Tv, Ti}(
        size(host, 2), size(host, 1), host.rowptr, host.colval, host.nzval)
    return copy(transpose(stored_transpose))
end

function matrix_to_backend(A::StaticSparsityMatrixCSR, backend, block_size::Int)
    matrix_backend(A) === backend && return A
    host = host_csr(A)
    rowptr = backend_copy(backend, host.rowptr)
    colval = backend_copy(backend, host.colval)
    nzval = backend_copy(backend, host.nzval)
    return StaticSparsityMatrixCSR(
        nzval, colval, rowptr, size(host, 1), size(host, 2), backend;
        nthreads = 1, minbatch = block_size, thread_type = :serial)
end
