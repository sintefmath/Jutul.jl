module KAPreconditioners

const JULIA_VER_CAN_SHRINK = VERSION >= v"1.11"

using KernelAbstractions
using LinearAlgebra
using Polyester: @batch
using SparseArrays
using StaticArrays: StaticMatrix, SVector
import Adapt
import ..Jutul

using ..Jutul: StaticSparsityMatrixCSR, colvals, minbatch, @tic
import ..Jutul: apply_ka_amg!, apply_ka_smoother!, setup_ka_amg,
    setup_ka_smoother, update_ka_amg!, update_ka_smoother!

const CAN_RESIZE_SHARED_ARRAY = VERSION >= v"1.11"

include("smoothers/types.jl")
include("types.jl")
include("csr.jl")
include("kernels.jl")
include("sparse_lu.jl")
include("smoothers/interface.jl")
include("smoothers/spai0.jl")
include("smoothers/gauss_seidel.jl")
include("smoothers/hybrid_gauss_seidel.jl")
include("smoothers/ilu0.jl")
include("smoothers/multicolor_ilu.jl")
include("smoothers/vendor_ilu.jl")
include("setup.jl")
include("pmis.jl")
include("aggressive.jl")
include("reset.jl")
include("cycle.jl")

function setup_ka_amg(A, options)
    return @tic "setup_amg" setup_amg(A, options)
end

function update_ka_amg!(hierarchy, A, reuse; kwargs...)
    return @tic "resetup_amg" resetup_amg!(hierarchy, A, reuse; kwargs...)
end

function apply_ka_amg!(x, hierarchy, b)
    return @tic "amg apply" apply!(x, hierarchy, b)
end

function setup_ka_smoother(A, config; reuse = nothing)
    return @tic "setup_smoother" setup_smoother(A, config; reuse = reuse)
end

function update_ka_smoother!(state, A)
    return @tic "update_smoother" update_smoother!(state, A)
end

smoother_index_type(::SPAI0State) = Int32
smoother_index_type(state::GaussSeidelState) = eltype(state.matrix.colval)
smoother_index_type(state) = eltype(state.host_rowptr)

function update_ka_smoother!(state, A::SparseMatrixCSC)
    matrix = csr_matrix(
        A;
        backend = state.backend,
        block_size = state.block_size,
        index_type = smoother_index_type(state)
    )
    return update_smoother!(state, matrix)
end
apply_ka_smoother!(x, state, b) = apply!(x, state, b)

export AbstractCoarsening, Aggregation, RugeStuben, HMIS, PMIS
export AbstractInterpolation, ConstantInterpolation, ClassicalInterpolation,
    ExtendedIInterpolation, TwoStageExtendedIInterpolation, AMGOptions, AMGHierarchy
export AbstractSmoother, AbstractSmootherState, SPAI0, GaussSeidel, HybridGaussSeidel, ILU0, DILU,
    VendorILU
export csr_matrix
export setup_smoother, update_smoother!, smooth!
export setup_amg, resetup_amg!, cycle!, solve!
export SparseLU

end
