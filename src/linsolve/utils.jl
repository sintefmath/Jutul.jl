export IterativeSolverConfig, reservoir_linsolve

mutable struct IterativeSolverConfig
    relative_tolerance::Union{Nothing, AbstractFloat}
    absolute_tolerance::Union{Nothing, AbstractFloat}
    max_iterations::Int
    min_iterations::Int
    nonlinear_relative_tolerance::Union{Nothing, AbstractFloat}
    relaxed_relative_tolerance::Union{Nothing, AbstractFloat}
    true_residual::Bool
    verbose
    arguments
end

function IterativeSolverConfig(;
        relative_tolerance = 1e-3,
        absolute_tolerance = nothing, 
        max_iterations = 100,
        min_iterations = 2,
        verbose = false,
        nonlinear_relative_tolerance = nothing,
        relaxed_relative_tolerance = 0.1,
        true_residual = false,
        kwarg...
    )
    IterativeSolverConfig(
        relative_tolerance,
        absolute_tolerance,
        max_iterations,
        min_iterations,
        nonlinear_relative_tolerance,
        relaxed_relative_tolerance,
        true_residual,
        verbose,
        kwarg
    )
end

function linear_solver_tolerance(cfg::IterativeSolverConfig, variant = :relative, T = Float64)
    if variant == :nonlinear_relative
        tol = cfg.nonlinear_relative_tolerance
    else
        if variant == :relative
            tol = cfg.relative_tolerance
        elseif variant == :relaxed_relative
            tol = cfg.relaxed_relative_tolerance
        else
            @assert variant == :absolute
            tol = cfg.absolute_tolerance
        end
        # default_num_tol = sqrt(eps(T))
        default_num_tol = 1e-12
        tol = T(isnothing(tol) ? default_num_tol : tol)
    end
    return tol
end

to_sparse_pattern(x::SparsePattern) = x

function to_sparse_pattern(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    I, J, _ = findnz(A)
    n, m = size(A)
    layout = matrix_layout(A)
    block_n, block_m = block_dims(A)
    return SparsePattern(I, J, n, m, layout, block_n, block_m)
end

function matrix_layout(A::SparseMatrixCSC{Tv, Ti}) where {Tv<:Real, Ti}
    return EquationMajorLayout()
end

function matrix_layout(A::SparseMatrixCSC{Tv, Ti}) where {Tv<:StaticMatrix, Ti}
    layout = BlockMajorLayout()
    return layout
end

matrix_layout(A::AbstractVector{T}) where {T<:StaticVector} = BlockMajorLayout()

function block_dims(A::SparseMatrixCSC{Tv, Ti}) where {Tv<:Real, Ti}
    return (1, 1)
end

function block_dims(A::SparseMatrixCSC{Tv, Ti}) where {Tv<:StaticMatrix, Ti}
    n, m = size(Tv)
    return (n, m)
end

block_dims(A::AbstractVector) = 1
block_dims(A::AbstractVector{T}) where T<:StaticVector = length(T)

"""
    unsafe_reinterpret(Vt, v, n)

Unsafely reinterpret v as a n length vector of value type Vt
"""
function unsafe_reinterpret(Vt, v, n)
    ptr = Base.unsafe_convert(Ptr{Vt}, v)
    return Base.unsafe_wrap(Array, ptr, n)::Vector{Vt}
end

function unsafe_reinterpret(::Val{Vt}, v, n) where Vt
    ptr = Base.unsafe_convert(Ptr{Vt}, v)
    return Base.unsafe_wrap(Array, ptr, n)::Vector{Vt}
end

function executor_index_to_global(executor::JutulExecutor, index, row_or_column::Symbol)
    return index
end
