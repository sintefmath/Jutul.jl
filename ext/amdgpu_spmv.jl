struct ROCSpMV{M, D, X, Y, W}
    matrix::M
    descriptor::D
    x_descriptor::X
    y_descriptor::Y
    workspace::W
    buffer_size::Base.RefValue{Csize_t}
end

struct ROCBlockSpMV{M, D}
    matrix::M
    descriptor::D
end

function roc_spmv_setup(A::StaticSparsityMatrixCSR{T}, x, y) where {T <: Number}
    matrix = ROCSparseMatrixCSR{T, eltype(A.rowptr)}(
        KAPreconditioners.logical_backend_buffer(A.rowptr),
        KAPreconditioners.logical_backend_buffer(A.colval),
        KAPreconditioners.logical_backend_buffer(A.nzval), size(A)
    )
    descriptor = rocSPARSE.ROCSparseMatrixDescriptor(matrix, 'O')
    x_descriptor = rocSPARSE.ROCDenseVectorDescriptor(x)
    y_descriptor = rocSPARSE.ROCDenseVectorDescriptor(y)
    buffer_size = Ref{Csize_t}(0)
    algorithm = rocSPARSE.rocsparse_spmv_alg_csr_adaptive
    if AMDGPU.HIP.runtime_version() >= v"6-"
        rocSPARSE.rocsparse_spmv(
            rocSPARSE.handle(), 'N', Ref{T}(one(T)), descriptor, x_descriptor,
            Ref{T}(zero(T)), y_descriptor, T, algorithm,
            rocSPARSE.rocsparse_spmv_stage_buffer_size, buffer_size, C_NULL
        )
    else
        rocSPARSE.rocsparse_spmv(
            rocSPARSE.handle(), 'N', Ref{T}(one(T)), descriptor, x_descriptor,
            Ref{T}(zero(T)), y_descriptor, T, algorithm, buffer_size, C_NULL
        )
    end
    workspace = ROCVector{UInt8}(undef, Int(buffer_size[]))
    if AMDGPU.HIP.runtime_version() >= v"6-"
        rocSPARSE.rocsparse_spmv(
            rocSPARSE.handle(), 'N', Ref{T}(one(T)), descriptor, x_descriptor,
            Ref{T}(zero(T)), y_descriptor, T, algorithm,
            rocSPARSE.rocsparse_spmv_stage_preprocess, buffer_size, workspace
        )
    end
    return ROCSpMV(matrix, descriptor, x_descriptor, y_descriptor, workspace, buffer_size)
end

function roc_spmv_setup(A::StaticSparsityMatrixCSR{T}, x, y) where {T <: StaticMatrix}
    scalar_type = eltype(T)
    block_size = size(T, 1)
    matrix = ROCSparseMatrixBSR{scalar_type}(
        convert(ROCVector{Cint}, KAPreconditioners.logical_backend_buffer(A.rowptr)),
        convert(ROCVector{Cint}, KAPreconditioners.logical_backend_buffer(A.colval)),
        reinterpret(scalar_type, KAPreconditioners.logical_backend_buffer(A.nzval)),
        (block_size * size(A, 1), block_size * size(A, 2)),
        block_size, 'C', length(A.nzval)
    )
    descriptor = rocSPARSE.ROCMatrixDescriptor('G', 'L', 'N', 'O')
    return ROCBlockSpMV(matrix, descriptor)
end

function roc_spmv!(plan::ROCSpMV, y, x, alpha, beta)
    T = eltype(plan.matrix)
    GC.@preserve plan x y begin
        rocSPARSE.rocsparse_dnvec_set_values(plan.x_descriptor, x)
        rocSPARSE.rocsparse_dnvec_set_values(plan.y_descriptor, y)
        if AMDGPU.HIP.runtime_version() >= v"6-"
            rocSPARSE.rocsparse_spmv(
                rocSPARSE.handle(), 'N', Ref{T}(alpha), plan.descriptor,
                plan.x_descriptor, Ref{T}(beta), plan.y_descriptor, T,
                rocSPARSE.rocsparse_spmv_alg_csr_adaptive,
                rocSPARSE.rocsparse_spmv_stage_compute, plan.buffer_size, plan.workspace
            )
        else
            rocSPARSE.rocsparse_spmv(
                rocSPARSE.handle(), 'N', Ref{T}(alpha), plan.descriptor,
                plan.x_descriptor, Ref{T}(beta), plan.y_descriptor, T,
                rocSPARSE.rocsparse_spmv_alg_csr_adaptive, plan.buffer_size, plan.workspace
            )
        end
    end
    return nothing
end

function roc_spmv!(plan::ROCBlockSpMV, y, x, alpha, beta)
    matrix = plan.matrix
    T = eltype(matrix)
    multiply = roc_sparse_function(T, :bsrmv)
    GC.@preserve plan x y begin
        multiply(
            rocSPARSE.handle(), matrix.dir, 'N',
            div(size(matrix, 1), matrix.blockDim), div(size(matrix, 2), matrix.blockDim),
            matrix.nnzb, Ref{T}(alpha), plan.descriptor, matrix.nzVal,
            matrix.rowPtr, matrix.colVal, matrix.blockDim, x, Ref{T}(beta), y
        )
    end
    return nothing
end

function KAPreconditioners.vendor_spmv!(
        y::AMDGPU.DenseROCVector,
        A::StaticSparsityMatrixCSR{Tv, Ti, V, I, R, B},
        x::AMDGPU.DenseROCVector, alpha, beta
    ) where {Tv, Ti, V, I, R, B <: AMDGPU.ROCBackend}
    T = KAPreconditioners.matrix_scalar_type(Tv)
    T <: ROCSPARSEValue || return false
    KAPreconditioners.logical_backend_buffer(A.nzval) isa AMDGPU.DenseROCVector || return false
    KAPreconditioners.logical_backend_buffer(A.rowptr) isa AMDGPU.DenseROCVector || return false
    KAPreconditioners.logical_backend_buffer(A.colval) isa AMDGPU.DenseROCVector || return false
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
        plan = roc_spmv_setup(A, x, y)
        A.vendor_linalg[] = plan
    end
    roc_spmv!(plan, y, x, convert(T, alpha), convert(T, beta))
    return true
end
