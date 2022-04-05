export IterativeSolverConfig, reservoir_linsolve

mutable struct IterativeSolverConfig
    relative_tolerance
    absolute_tolerance
    max_iterations
    nonlinear_relative_tolerance
    relaxed_relative_tolerance
    verbose
    arguments
    function IterativeSolverConfig(;relative_tolerance = 1e-3,
                                    absolute_tolerance = nothing, 
                                    max_iterations = 100,
                                    verbose = false,
                                    nonlinear_relative_tolerance = nothing,
                                    relaxed_relative_tolerance = 0.1,
                                    kwarg...)
        new(relative_tolerance, absolute_tolerance, max_iterations, nonlinear_relative_tolerance, relaxed_relative_tolerance, verbose, kwarg)
    end
end

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

