abstract type AbstractCoarsening end
abstract type AbstractInterpolation end

"""Unsmoothed, strength-based aggregation."""
struct Aggregation <: AbstractCoarsening
    theta::Float64
end
Aggregation(theta::Real=0.25) = Aggregation(Float64(theta))

"""Classical Ruge-Stuben C/F splitting."""
struct RugeStuben <: AbstractCoarsening
    theta::Float64
end
RugeStuben(theta::Real=0.25) = RugeStuben(Float64(theta))

"""Piecewise-constant interpolation for unsmoothed aggregation."""
struct ConstantInterpolation <: AbstractInterpolation end

"""
    ClassicalInterpolation(truncation=0.0, max_elements=0, norm_p=2, rescale=false)

Classical interpolation through strong coarse neighbors. A zero
`max_elements` retains every candidate.
"""
struct ClassicalInterpolation <: AbstractInterpolation
    truncation::Float64
    max_elements::Int
    norm_p::Int
    rescale::Bool
    function ClassicalInterpolation(truncation::Real=0.0, max_elements::Integer=0,
                                    norm_p::Integer=2, rescale::Bool=false)
        truncation >= 0 || throw(ArgumentError("truncation must be non-negative"))
        max_elements >= 0 || throw(ArgumentError("max_elements must be non-negative"))
        norm_p > 0 || throw(ArgumentError("norm_p must be positive"))
        new(Float64(truncation), Int(max_elements), Int(norm_p), rescale)
    end
end

"""
    ExtendedIInterpolation(truncation=0.0, max_elements=4, norm_p=2, rescale=true)

Configuration for distance-two Extended+i interpolation.
"""
struct ExtendedIInterpolation <: AbstractInterpolation
    truncation::Float64
    max_elements::Int
    norm_p::Int
    rescale::Bool
    function ExtendedIInterpolation(truncation::Real=0.0, max_elements::Integer=4,
                                    norm_p::Integer=2, rescale::Bool=true)
        truncation >= 0 || throw(ArgumentError("truncation must be non-negative"))
        max_elements > 0 || throw(ArgumentError("max_elements must be positive"))
        norm_p > 0 || throw(ArgumentError("norm_p must be positive"))
        new(Float64(truncation), Int(max_elements), Int(norm_p), rescale)
    end
end

"""Hybrid modified independent-set coarsening."""
struct HMIS <: AbstractCoarsening
    theta::Float64
end

HMIS(theta::Real=0.5) = HMIS(Float64(theta))

default_interpolation(::Aggregation) = ConstantInterpolation()
default_interpolation(::RugeStuben) = ClassicalInterpolation()
default_interpolation(::HMIS) = ExtendedIInterpolation()

"""
    AMGOptions(; coarsening=HMIS(), interpolation=default_interpolation(coarsening), ...)

Configuration for the backend-portable AMG hierarchy. Interpolation is an
independent hierarchy option whose default follows the coarsening method:
piecewise constant for aggregation, classical for Ruge-Stuben, and Extended+i
for HMIS.
"""
struct AMGOptions
    coarsening::AbstractCoarsening
    interpolation::AbstractInterpolation
    smoother::AbstractSmoother
    max_levels::Int
    coarse_size::Int
    coarse_solver::Symbol
    coarse_steps::Int
    max_row_sum::Float64
    block_size::Int
    cycle::Symbol
end

function AMGOptions(;
        coarsening::AbstractCoarsening=HMIS(),
        interpolation::AbstractInterpolation=default_interpolation(coarsening),
        smoother::AbstractSmoother=SPAI0(1, 1.0),
        max_levels::Integer=20,
        coarse_size::Integer=50,
        coarse_solver::Symbol=:lu,
        coarse_steps::Integer=8,
        max_row_sum::Real=1.0,
        block_size::Integer=128,
        cycle::Symbol=:V)
    AMGOptions(coarsening, interpolation, smoother, Int(max_levels),
               Int(coarse_size), coarse_solver, Int(coarse_steps),
               Float64(max_row_sum), Int(block_size), cycle)
end

struct Prolongation{Tv,Ti,RP,CV,NZ}
    rowptr::RP
    colval::CV
    nzval::NZ
    nrow::Int
    ncol::Int
end

struct TransposeMap{Ti,OFF,ROW,IDX}
    offsets::OFF
    fine_rows::ROW
    p_indices::IDX
end

"""Triples grouped by coarse nonzero; enables one kernel item per output."""
struct GalerkinMap{Ti,OFF,PI,AI,PJ}
    offsets::OFF
    p_left::PI
    a_index::AI
    p_right::PJ
end

"""Cached sparse LU state for a CPU coarse grid."""
mutable struct CoarseLUState{M,F,Map}
    matrix::M
    factors::F
    csr_to_csc::Map
end

"""Backend-native dense LU factorization for a coarse grid."""
mutable struct DenseLUState{F}
    factorization::F
end

"""Standard host LU fallback for backends without a native dense LU overload."""
mutable struct HostLUState{F,V,RP,CV,B}
    factorization::F
    values::V
    rowptr::RP
    colval::CV
    rhs::B
end

mutable struct AMGLevel{Tv,Ti}
    A::Any
    P::Any
    Pt::Any
    galerkin::Any
    smoother::Any
    coarse_solver::Any
    residual::Any
    correction::Any
    rhs::Any
    cf::Any
    coarse_map::Any
    strength::Any
end

"""Host scratch and staging storage retained across symbolic rebuilds."""
mutable struct SetupWorkspace{Tv,Ti}
    ti1::Vector{Ti}
    ti2::Vector{Ti}
    ti3::Vector{Ti}
    int1::Vector{Int}
    int2::Vector{Int}
    int3::Vector{Int}
    int4::Vector{Int}
    int5::Vector{Int}
    real1::Vector{Float64}
    values::Vector{Tv}
    bool1::Vector{Bool}
    markers::Vector{Ti}
    interpolation_cols::Vector{Vector{Ti}}
    interpolation_vals::Vector{Vector{Tv}}
    galerkin_cols::Vector{Vector{Ti}}
    stage_matrix1::Any
    stage_matrix2::Any
    stage_prolongation::Any
    stage_transpose::Any
    stage_galerkin::Any
    stage_cf::Vector{Int8}
    stage_coarse_map::Vector{Ti}
    stage_strength::Vector{Bool}
end

function SetupWorkspace(::Type{Tv}, ::Type{Ti}) where {Tv,Ti}
    nt = Threads.maxthreadid()
    SetupWorkspace{Tv,Ti}(
        Ti[], Ti[], Ti[], Int[], Int[], Int[], Int[], Int[], Float64[], Tv[], Bool[], Ti[],
        [sizehint!(Ti[], 32) for _ in 1:nt],
        [sizehint!(Tv[], 32) for _ in 1:nt],
        [sizehint!(Ti[], 64) for _ in 1:nt],
        nothing, nothing, nothing, nothing, nothing, Int8[], Ti[], Bool[],
    )
end

"""Reusable AMG hierarchy and Krylov-compatible left preconditioner."""
mutable struct AMGHierarchy{Tv,Ti}
    levels::Vector{AMGLevel{Tv,Ti}}
    workspace::SetupWorkspace{Tv,Ti}
    options::AMGOptions
    backend::Any
    block_size::Int
    pattern_rowptr::Vector{Ti}
    pattern_colval::Vector{Ti}
    last_iterations::Int
    last_residual::Float64
end

Base.size(H::AMGHierarchy) = size(H.levels[1].A)
Base.size(H::AMGHierarchy, d::Integer) = size(H.levels[1].A, d)
Base.eltype(::Type{<:AMGHierarchy{Tv}}) where Tv = Tv
Base.eltype(::AMGHierarchy{Tv}) where Tv = Tv
