"""
    JutulKernelAbstractions

Module providing GPU / KernelAbstractions-backed simulation support for Jutul.

Key features:
- `KernelAbstractionsContext` – context tag that selects KA-based kernels.
- Device-parametric `StaticSparsityMatrixCSR` (nzval & colval on device).
- Kernels for equation evaluation, Jacobian assembly, variable updates,
  convergence checks and CSR matrix-vector multiply.
- `transfer_to_device` to move a fully set-up `Simulator` onto the device.
"""
module JutulKernelAbstractions
    using ..Jutul
    using ..Jutul: StaticSparsityMatrixCSR, colvals, static_sparsity_sparse
    using ..Jutul: KernelAbstractionsContext
    using ..Jutul: JutulAutoDiffCache, CompactAutoDiffCache, GenericAutoDiffCache
    using ..Jutul: JutulStorage, JutulModel, SimulationModel, JutulSimulator, Simulator
    using ..Jutul: LinearizedSystem, JutulLinearSystem
    using ..Jutul: JutulEntity
    using ..Jutul: value, ad_dims, get_entries, get_entry, get_jacobian_pos
    using ..Jutul: number_of_entities, equations_per_entity, number_of_partials
    using ..Jutul: update_jacobian_entry!, insert_residual_value, fill_equation_entries!
    using ..Jutul: update_equation_in_entity!, local_discretization
    using ..Jutul: update_equation_for_entity!
    using ..Jutul: EquationMajorLayout, EntityMajorLayout, BlockMajorLayout, JutulMatrixLayout
    using ..Jutul: update_values!
    using ..Jutul: matrix_layout, float_type, index_type, nzval_index_type, minbatch
    using ..Jutul: build_sparse_matrix, jacobian_eltype
    using ..Jutul: linear_solve!, linear_solve_return
    using ..Jutul: Cells, Faces
    using ..Jutul: JutulContext
    using ..Jutul: convert_to_immutable_storage, data
    using ..Jutul: setup_equations_and_primary_variable_views

    using KernelAbstractions
    using GPUArrays
    using SparseArrays
    using LinearAlgebra
    import ForwardDiff

    export transfer_to_device

    include("context.jl")
    include("sparse.jl")
    include("device_caches.jl")
    include("kernels.jl")
    include("transfer.jl")
end
