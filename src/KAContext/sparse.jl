# Sparse matrix construction for KernelAbstractionsContext
# Build CSR matrix with device arrays for values and column indices.

"""
    build_sparse_matrix(context::KernelAbstractionsContext, I, J, V, n, m)

Build a `StaticSparsityMatrixCSR` on the CPU, then convert the value and
column-index vectors to device arrays on the KA backend.
"""
function Jutul.build_sparse_matrix(context::KernelAbstractionsContext, I, J, V, n, m)
    # First build the CPU CSR matrix
    cpu_csr = static_sparsity_sparse(I, J, V, n, m, nthreads = 1, minbatch = typemax(Int))
    return transfer_csr_to_device(cpu_csr, context.backend)
end

"""
    transfer_csr_to_device(cpu_csr, backend)

Transfer the internal arrays of a `StaticSparsityMatrixCSR` to device memory,
returning a new `StaticSparsityMatrixCSR` with device arrays for nzval and colval
but CPU-resident rowptr (since it is only needed during kernel launches for range
computation).
"""
function transfer_csr_to_device(cpu_csr::StaticSparsityMatrixCSR, backend)
    nz_cpu = nonzeros(cpu_csr)
    cv_cpu = colvals(cpu_csr)
    rp_cpu = cpu_csr.rowptr

    nz_dev = to_device(backend, nz_cpu)
    cv_dev = to_device(backend, cv_cpu)
    m, n = size(cpu_csr)
    return StaticSparsityMatrixCSR(nz_dev, cv_dev, rp_cpu, m, n, nthreads = 1, minbatch = typemax(Int))
end
