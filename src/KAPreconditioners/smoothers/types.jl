abstract type AbstractSmoother end
abstract type AbstractSmootherState end

function positive_smoother_parameters(
        name::AbstractString, steps::Integer,
        damping::Real
    )
    steps > 0 || throw(ArgumentError("$name steps must be positive"))
    damping > 0 || throw(ArgumentError("$name damping must be positive"))
    return Int(steps), Float64(damping)
end

"""Sparse approximate-inverse relaxation with the matrix sparsity pattern."""
struct SPAI0 <: AbstractSmoother
    steps::Int
    damping::Float64
    function SPAI0(steps::Integer = 2, damping::Real = 1.0)
        return new(positive_smoother_parameters("SPAI0", steps, damping)...)
    end
end

"""
    GaussSeidel(steps=1, damping=1.0)

Forward Gauss-Seidel on the down cycle and backward Gauss-Seidel on the up
cycle. This CPU smoother mirrors BoomerAMG's default serial relaxation.
"""
struct GaussSeidel <: AbstractSmoother
    steps::Int
    damping::Float64
    function GaussSeidel(steps::Integer = 1, damping::Real = 1.0)
        steps > 0 || throw(ArgumentError("Gauss-Seidel steps must be positive"))
        0 < damping <= 1 || throw(
            ArgumentError(
                "Gauss-Seidel damping must be in (0, 1]"
            )
        )
        return new(Int(steps), Float64(damping))
    end
end

"""
    HybridGaussSeidel(steps=1, damping=1.0; npartitions=Threads.nthreads(),
        min_partition_size=8192, partitioner=:contiguous, l1=true)

CPU hybrid Gauss-Seidel for real scalar matrices. Each partition sweeps its
owned rows sequentially, with cross-partition contributions frozen at the
start of every sweep. Down/up cycles use forward/backward ordering. Standalone
application uses both directions. No coloring or per-color launches are used.

By default, half the sum of absolute cross-partition coefficients is added to
the diagonal magnitude (scaled ℓ₁ GS). This guarantees convergence for SPD
matrices and changes only interface rows. `l1=false` selects ordinary hybrid
GS. Contiguous partitions preserve memory locality and avoid graph setup;
`partitioner=:metis` reduces interfaces on irregular meshes. Small levels use
fewer partitions so each has at least `min_partition_size` rows when possible.
An explicit row ownership vector can be passed to
`setup_smoother(A, config; partition=owners)`.
"""
struct HybridGaussSeidel <: AbstractSmoother
    steps::Int
    damping::Float64
    npartitions::Int
    min_partition_size::Int
    partitioner::Symbol
    l1::Bool
    function HybridGaussSeidel(
            steps::Integer = 1, damping::Real = 1.0;
            npartitions::Integer = Threads.nthreads(),
            min_partition_size::Integer = 8192,
            partitioner::Symbol = :contiguous, l1::Bool = true
        )
        validated = GaussSeidel(steps, damping)
        npartitions > 0 || throw(ArgumentError("npartitions must be positive"))
        min_partition_size > 0 || throw(ArgumentError("min_partition_size must be positive"))
        partitioner in (:metis, :contiguous) || throw(ArgumentError("partitioner must be :metis or :contiguous"))
        return new(
            validated.steps, validated.damping, Int(npartitions),
            Int(min_partition_size), partitioner, l1
        )
    end
end

"""
    ILU0(steps=1, damping=1.0)

Level-scheduled incomplete LU factorization with zero fill. The symbolic level
schedules are retained and reused when matrix coefficients are updated.
"""
struct ILU0 <: AbstractSmoother
    steps::Int
    damping::Float64
    function ILU0(steps::Integer = 1, damping::Real = 1.0)
        return new(positive_smoother_parameters("ILU0", steps, damping)...)
    end
end

"""
    DILU(steps=1, damping=1.0)

Diagonal ILU(0), implemented with level-scheduled factorization and triangular
sweeps following Andersen et al. The level schedule preserves the natural row
ordering, so host and accelerator backends construct the same preconditioner.
"""
struct DILU <: AbstractSmoother
    steps::Int
    damping::Float64
    function DILU(steps::Integer = 1, damping::Real = 1.0)
        return new(positive_smoother_parameters("DILU", steps, damping)...)
    end
end

"""
    VendorILU(steps=1, damping=1.0)

Backend-native ILU(0). Scalar matrices use the vendor CSR implementation and
matrices with static dense blocks use the vendor block CSR implementation.
Setup throws when the matrix backend has no native implementation.
"""
struct VendorILU <: AbstractSmoother
    steps::Int
    damping::Float64
    function VendorILU(steps::Integer = 1, damping::Real = 1.0)
        return new(positive_smoother_parameters("VendorILU", steps, damping)...)
    end
end

mutable struct SPAI0State{D, C} <: AbstractSmootherState
    diagonal::D
    temporary::Any
    residual::Any
    config::C
    backend::Any
    block_size::Int
    n::Int
end

mutable struct GaussSeidelState{D, C, B} <: AbstractSmootherState
    # AMG resetup may replace a host matrix wrapper while retaining its CSR
    # arrays. Keeping the wrapper type-erased avoids making reuse depend on
    # incidental type parameters such as the optional backend field.
    matrix::Any
    inverse_diagonal::D
    correction::D
    residual::D
    config::C
    backend::B
    block_size::Int
    n::Int
end

mutable struct HybridGaussSeidelState{T, I, S, B} <: AbstractSmootherState
    matrix::Any
    inverse_diagonal::Vector{T}
    correction::Vector{T}
    config::HybridGaussSeidel
    backend::B
    block_size::Int
    n::Int
    rowptr::Vector{I}
    colval::Vector{I}
    host_rowptr::Vector{I}
    host_colval::Vector{I}
    schedule::S
    halo_rhs::Vector{T}
end

mutable struct ILU0State{
        F, D, RP, CV, DP, FO, FF, UO, UF, HRP, HCV,
        W, R, FK, SK, SMK, C, B,
    } <: AbstractSmootherState
    factors::F
    inverse_diagonal::D
    work::W
    residual::R
    rowptr::RP
    colval::CV
    diagonal_positions::DP
    factor_offsets::FO
    factor_rows::FF
    upper_offsets::UO
    upper_rows::UF
    host_rowptr::HRP
    host_colval::HCV
    factor_kernel::FK
    solve_kernels::SK
    smooth_kernels::SMK
    config::C
    backend::B
    block_size::Int
    n::Int
end

mutable struct DILUState{
        D, AV, RP, CV, DP, TP, FO, FF, UO, UF, HRP, HCV,
        W, R, FK, SK, SMK, C, B,
    } <: AbstractSmootherState
    inverse_diagonal::D
    work::W
    residual::R
    values::AV
    rowptr::RP
    colval::CV
    diagonal_positions::DP
    transpose_positions::TP
    factor_offsets::FO
    factor_rows::FF
    upper_offsets::UO
    upper_rows::UF
    host_rowptr::HRP
    host_colval::HCV
    factor_kernel::FK
    solve_kernels::SK
    smooth_kernels::SMK
    config::C
    backend::B
    block_size::Int
    n::Int
end

mutable struct VendorILUState{
        Tv, F, FV, W, R, RP, CV, HRP, HCV, C, B,
    } <: AbstractSmootherState
    factor::F
    factor_values::FV
    work::W
    residual::R
    rowptr::RP
    colval::CV
    host_rowptr::HRP
    host_colval::HCV
    config::C
    backend::B
    block_size::Int
    n::Int
end

function VendorILUState(
        ::Type{Tv}, factor::F, factor_values::FV, work::W, residual::R,
        rowptr::RP, colval::CV, host_rowptr::HRP, host_colval::HCV,
        config::C, backend::B, block_size::Int, n::Int
    ) where {Tv, F, FV, W, R, RP, CV, HRP, HCV, C, B}
    return VendorILUState{
        Tv, F, FV, W, R, RP, CV, HRP, HCV, C, B,
    }(
        factor, factor_values, work, residual, rowptr, colval,
        host_rowptr, host_colval, config, backend, block_size, n
    )
end
