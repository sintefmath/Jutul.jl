export KernelAbstractionsContext

"""
    KernelAbstractionsContext(; <keyword arguments>)

Context for running Jutul simulations using KernelAbstractions.jl kernels on
GPU or other accelerator backends. Sparsity detection and equation alignment
happen on the CPU, but state variables, equation storage, and the linearized
system are transferred to device arrays after setup. Linearization, auto-diff
equation evaluation, convergence checks, variable updates and Jacobian assembly
are all performed via kernel launches on the target device.

# Keyword arguments
- `backend`: A `KernelAbstractions.Backend` (e.g. `CUDABackend()`, `JLBackend()`).
- `float_t::Type = Float64`: Floating-point type for device arrays.
- `index_t::Type = Int64`: Integer type for indices.
- `matrix_layout = EquationMajorLayout()`: Matrix layout for the linearized system.
"""
struct KernelAbstractionsContext{B} <: GPUJutulContext
    backend::B
    float_t::Type
    index_t::Type
    matrix_layout
    function KernelAbstractionsContext(backend::B;
            float_t::Type = Float64,
            index_t::Type = Int64,
            matrix_layout = EquationMajorLayout()) where B
        return new{B}(backend, float_t, index_t, matrix_layout)
    end
end

matrix_layout(c::KernelAbstractionsContext) = c.matrix_layout
