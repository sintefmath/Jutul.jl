abstract type AbstractSmoother end
abstract type AbstractSmootherState end

"""Sparse approximate-inverse relaxation with the matrix sparsity pattern."""
struct SPAI0 <: AbstractSmoother
    steps::Int
    damping::Float64
    function SPAI0(steps::Integer=2, damping::Real=1.0)
        steps > 0 || throw(ArgumentError("SPAI0 steps must be positive"))
        damping > 0 || throw(ArgumentError("SPAI0 damping must be positive"))
        new(Int(steps), Float64(damping))
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
    function ILU0(steps::Integer=1, damping::Real=1.0)
        steps > 0 || throw(ArgumentError("ILU0 steps must be positive"))
        damping > 0 || throw(ArgumentError("ILU0 damping must be positive"))
        new(Int(steps), Float64(damping))
    end
end

"""
    DILU(steps=1, damping=1.0)

Diagonal ILU(0), implemented with scheduled factorization and triangular
sweeps following Andersen et al. Accelerator arrays use a graph-colored row
ordering so that each color is processed in parallel with a bounded number of
kernel launches. Host arrays retain the natural row ordering.
"""
struct DILU <: AbstractSmoother
    steps::Int
    damping::Float64
    function DILU(steps::Integer=1, damping::Real=1.0)
        steps > 0 || throw(ArgumentError("DILU steps must be positive"))
        damping > 0 || throw(ArgumentError("DILU damping must be positive"))
        new(Int(steps), Float64(damping))
    end
end

mutable struct SPAI0State{D,C} <: AbstractSmootherState
    diagonal::D
    temporary::Any
    residual::Any
    config::C
    backend::Any
    block_size::Int
    n::Int
end

mutable struct ILU0State{F,D,RP,CV,DP,FO,FF,UO,UF,HRP,HCV,C} <: AbstractSmootherState
    factors::F
    inverse_diagonal::D
    work::Any
    residual::Any
    rowptr::RP
    colval::CV
    diagonal_positions::DP
    factor_offsets::FO
    factor_rows::FF
    upper_offsets::UO
    upper_rows::UF
    host_rowptr::HRP
    host_colval::HCV
    config::C
    backend::Any
    block_size::Int
    n::Int
end

mutable struct DILUState{D,AV,RP,CV,DP,TP,O,FO,FF,UO,UF,HRP,HCV,C} <: AbstractSmootherState
    inverse_diagonal::D
    work::Any
    residual::Any
    values::AV
    rowptr::RP
    colval::CV
    diagonal_positions::DP
    transpose_positions::TP
    ordering::O
    factor_offsets::FO
    factor_rows::FF
    upper_offsets::UO
    upper_rows::UF
    host_rowptr::HRP
    host_colval::HCV
    config::C
    backend::Any
    block_size::Int
    n::Int
end
