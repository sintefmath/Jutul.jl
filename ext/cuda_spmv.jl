# The sparse pattern and values alias Jutul's matrix. Descriptors, workspace
# and preprocessing belong to that matrix and survive numeric assembly updates.
struct CUDASpMV{M, D, X, Y, W}
    matrix::M
    descriptor::D
    x_descriptor::X
    y_descriptor::Y
    workspace::W
end

struct CUDABlockSpMV{M, D}
    matrix::M
    descriptor::D
end

function cuda_spmv_setup(A::StaticSparsityMatrixCSR{T}, x, y) where {T <: Number}
    matrix = cusparse_wrapper(A)
    descriptor = CUDA.CUSPARSE.CuSparseMatrixDescriptor(matrix, 'O')
    x_descriptor = CUDA.CUSPARSE.CuDenseVectorDescriptor(x)
    y_descriptor = CUDA.CUSPARSE.CuDenseVectorDescriptor(y)
    algorithm = CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG1
    buffer_size = Ref{Csize_t}(0)
    CUDA.CUSPARSE.cusparseSpMV_bufferSize(
        CUDA.CUSPARSE.handle(), 'N', Ref{T}(one(T)), descriptor,
        x_descriptor, Ref{T}(zero(T)), y_descriptor, T, algorithm, buffer_size
    )
    workspace = CuVector{UInt8}(undef, Int(buffer_size[]))
    if CUDA.CUSPARSE.version() >= v"12.4"
        CUDA.CUSPARSE.cusparseSpMV_preprocess(
            CUDA.CUSPARSE.handle(), 'N', Ref{T}(one(T)), descriptor,
            x_descriptor, Ref{T}(zero(T)), y_descriptor, T, algorithm, workspace
        )
    end
    return CUDASpMV(matrix, descriptor, x_descriptor, y_descriptor, workspace)
end

function cuda_spmv_setup(A::StaticSparsityMatrixCSR{T}, x, y) where {T <: StaticMatrix}
    scalar_type = eltype(T)
    block_size = size(T, 1)
    # Legacy BSR SpMV requires 32-bit indices. Convert once, without copying
    # values: their column-major block storage is the vendor's 'C' layout.
    rowptr = convert(CuVector{Cint}, KAPreconditioners.logical_backend_buffer(A.rowptr))
    colval = convert(CuVector{Cint}, KAPreconditioners.logical_backend_buffer(A.colval))
    matrix = CuSparseMatrixBSR{scalar_type}(
        rowptr, colval, reinterpret(scalar_type, KAPreconditioners.logical_backend_buffer(A.nzval)),
        (block_size * size(A, 1), block_size * size(A, 2)),
        block_size, 'C', length(A.nzval)
    )
    descriptor = CUDA.CUSPARSE.CuMatrixDescriptor('G', 'L', 'N', 'O')
    return CUDABlockSpMV(matrix, descriptor)
end

function cuda_spmv!(plan::CUDASpMV, y, x, alpha, beta)
    T = eltype(plan.matrix)
    GC.@preserve plan x y begin
        # Krylov methods alternate vectors/views while reusing the matrix.
        CUDA.CUSPARSE.cusparseDnVecSetValues(plan.x_descriptor, x)
        CUDA.CUSPARSE.cusparseDnVecSetValues(plan.y_descriptor, y)
        CUDA.CUSPARSE.cusparseSpMV(
            CUDA.CUSPARSE.handle(), 'N', Ref{T}(alpha), plan.descriptor,
            plan.x_descriptor, Ref{T}(beta), plan.y_descriptor, T,
            CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG1, plan.workspace
        )
    end
    return nothing
end

function cuda_spmv!(plan::CUDABlockSpMV, y, x, alpha, beta)
    matrix = plan.matrix
    T = eltype(matrix)
    prefix = T === Float32 ? "S" : T === Float64 ? "D" :
        T === ComplexF32 ? "C" : "Z"
    multiply = getproperty(CUDA.CUSPARSE, Symbol("cusparse", prefix, "bsrmv"))
    GC.@preserve plan x y begin
        multiply(
            CUDA.CUSPARSE.handle(), matrix.dir, 'N',
            div(size(matrix, 1), matrix.blockDim), div(size(matrix, 2), matrix.blockDim),
            matrix.nnzb, Ref{T}(alpha), plan.descriptor, matrix.nzVal,
            matrix.rowPtr, matrix.colVal, matrix.blockDim, x, Ref{T}(beta), y
        )
    end
    return nothing
end

function KAPreconditioners.vendor_spmv!(
        y::CUDA.DenseCuVector,
        A::StaticSparsityMatrixCSR{Tv, Ti, V, I, R, B},
        x::CUDA.DenseCuVector, alpha, beta
    ) where {Tv, Ti, V, I, R, B <: CUDA.CUDABackend}
    T = KAPreconditioners.matrix_scalar_type(Tv)
    T <: CUSPARSEValue || return false
    KAPreconditioners.logical_backend_buffer(A.nzval) isa CUDA.DenseCuVector || return false
    KAPreconditioners.logical_backend_buffer(A.rowptr) isa CUDA.DenseCuVector || return false
    KAPreconditioners.logical_backend_buffer(A.colval) isa CUDA.DenseCuVector || return false
    if Tv <: Number
        Ti <: Union{Int32, Int64} || return false
        eltype(x) === T && eltype(y) === T || return false
    elseif Tv <: StaticMatrix
        size(Tv, 1) == size(Tv, 2) || return false
        max(size(A)..., length(A.nzval) + 1) <= typemax(Cint) || return false
        eltype(x) <: StaticVector{size(Tv, 2), T} || return false
        eltype(y) <: StaticVector{size(Tv, 1), T} || return false
        x = reinterpret(T, x)
        y = reinterpret(T, y)
    else
        return false
    end
    plan = A.vendor_linalg[]
    if isnothing(plan)
        plan = cuda_spmv_setup(A, x, y)
        A.vendor_linalg[] = plan
    end
    cuda_spmv!(plan, y, x, convert(T, alpha), convert(T, beta))
    return true
end
