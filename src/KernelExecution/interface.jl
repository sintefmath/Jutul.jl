export KernelAbstractionsContext
export transfer_to_backend, backend_copyto!, prepare_backend_transfer!

"""
    transfer_to_backend(simulator, backend; kwargs...)

Transfer an initialized simulator to an execution backend. Backend modules
implement this interface for the contexts they support.
"""
function transfer_to_backend end

"""
    backend_copyto!(destination, source)

Copy compatible simulation storage from host evaluation buffers to backend
buffers. Unsupported metadata is deliberately left unchanged by default.
"""
backend_copyto!(destination, source) = destination

"""
    prepare_backend_transfer!(storage, model)

Application hook for refreshing host-side numeric buffers immediately before
they are copied to a backend mirror.
"""
prepare_backend_transfer!(storage, model) = storage

"Create a factorization for a linear-system block."
factorize_linear_system(constructor, matrix) = constructor(matrix)

"Refresh an existing factorization for a linear-system block."
refactorize_linear_system!(update!, factorization, matrix) =
    update!(factorization, matrix)

"Copy a CPU CSR matrix to the backend used by a reference CSR matrix."
transfer_csr_to_backend(reference, matrix) = matrix
