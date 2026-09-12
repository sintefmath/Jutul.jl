module KAPreconditioners

using KernelAbstractions
using LinearAlgebra
using SparseArrays

using ..Jutul: StaticSparsityMatrixCSR, colvals, minbatch
import ..Jutul: apply_ka_amg!, apply_ka_smoother!, setup_ka_amg,
    setup_ka_smoother, update_ka_amg!, update_ka_smoother!

const CAN_RESIZE_SHARED_ARRAY = VERSION >= v"1.11"

include("smoothers/types.jl")
include("types.jl")
include("csr.jl")
include("kernels.jl")
include("smoothers/interface.jl")
include("smoothers/spai0.jl")
include("smoothers/ilu0.jl")
include("setup.jl")
include("reset.jl")
include("cycle.jl")

setup_ka_amg(A, options) = setup_amg(A, options)
update_ka_amg!(hierarchy, A, reuse) = resetup_amg!(hierarchy, A, reuse)
apply_ka_amg!(x, hierarchy, b) = apply!(x, hierarchy, b)

setup_ka_smoother(A, config; reuse = nothing) =
    setup_smoother(A, config; reuse = reuse)
update_ka_smoother!(state, A) = update_smoother!(state, A)
function update_ka_smoother!(state, A::SparseMatrixCSC)
    index_type = if state isa SPAI0State
        Int32
    else
        eltype(state.host_rowptr)
    end
    matrix = csr_matrix(A;
        backend = state.backend,
        block_size = state.block_size,
        index_type = index_type
    )
    return update_smoother!(state, matrix)
end
apply_ka_smoother!(x, state, b) = apply!(x, state, b)

export AbstractCoarsening, Aggregation, RugeStuben, HMIS
export ExtendedIInterpolation, AMGOptions, AMGHierarchy
export AbstractSmoother, AbstractSmootherState, SPAI0, ILU0, DILU
export csr_matrix
export setup_smoother, update_smoother!, smooth!
export setup_amg, resetup_amg!, cycle!, solve!

end
