module JutulKAPreconditionersMetalExt

using Jutul
using Jutul.KAPreconditioners
using Jutul: StaticSparsityMatrixCSR
using KernelAbstractions
using Metal: MtlVector

function KAPreconditioners.csr_matrix(
        rowptr::MtlVector{Ti}, colval::MtlVector{Ti},
        nzval::MtlVector{Tv}, dims::NTuple{2, <:Integer};
        block_size::Integer = 128) where {Tv, Ti<:Integer}
    return StaticSparsityMatrixCSR(
        nzval, colval, rowptr, dims[1], dims[2],
        KernelAbstractions.get_backend(nzval);
        nthreads = 1, minbatch = Int(block_size), thread_type = :serial)
end

end
