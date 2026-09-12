module JutulKAPreconditionersAMDGPUExt

using Jutul
using Jutul.KAPreconditioners
using Jutul: StaticSparsityMatrixCSR
using KernelAbstractions
import AMDGPU
using AMDGPU: ROCArray, ROCSparseMatrixCSR

@static if pkgversion(AMDGPU) >= v"2.6.0"
    KAPreconditioners.native_dense_lu(::ROCArray) =
        AMDGPU.functional(:rocsolver)
end

function KAPreconditioners.csr_matrix(A::ROCSparseMatrixCSR;
        block_size::Integer = 128)
    return StaticSparsityMatrixCSR(
        A.nzVal, A.colVal, A.rowPtr, size(A, 1), size(A, 2),
        KernelAbstractions.get_backend(A.nzVal);
        nthreads = 1, minbatch = Int(block_size), thread_type = :serial)
end

end
