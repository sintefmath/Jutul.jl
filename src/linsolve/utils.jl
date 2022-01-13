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

function reservoir_linsolve(model, method = :cpr;
                                   rtol = 0.005,
                                   v = 0,
                                   provider = Krylov,
                                   solver = Krylov.bicgstab,
                                    update_interval = :ministep,
                                    amg_type = :smoothed_aggregation,
                                    max_coarse = nothing,
                                    cpr_type = nothing,
                                    partial_update = update_interval == :once,
                                    kwarg...)
    res_model(model) = model
    model = res_model(model)
    if model.context == DefaultContext()
        return nothing
    end
    if method == :cpr
        if isnothing(cpr_type)
            if isa(model.system, ImmiscibleSystem)
                cpr_type = :analytical
            else
                cpr_type = :true_impes
            end
        end
        if isnothing(max_coarse)
            max_coarse = Int64(ceil(0.05*number_of_cells(model.domain)))
            max_coarse = min(1000, max_coarse)
        end
        p_solve = default_psolve(max_coarse = max_coarse, type = amg_type)
        prec = CPRPreconditioner(p_solve, strategy = cpr_type, 
                                 update_interval = update_interval,
                                 partial_update = partial_update)
    elseif method == :ilu0
        prec = ILUZeroPreconditioner()
    else
        return nothing
    end
    max_it = 200
    atol = 0.0#1e-12
    # v = -1
    # v = 0
    # v = 1
    # if true
    #     krylov = Krylov.bicgstab
    #     # krylov = Krylov.gmres
    #     pv = Krylov
    # else
    #     krylov = IterativeSolvers.gmres!
    #     krylov = IterativeSolvers.bicgstabl!
    #     pv = IterativeSolvers
    # end
    # nl_reltol = 1e-3
    # relaxed_reltol = 0.25
    # nonlinear_relative_tolerance = nl_reltol,
    # relaxed_relative_tolerance = relaxed_reltol,

    lsolve = GenericKrylov(solver, provider = provider, verbose = v, preconditioner = prec, 
            relative_tolerance = rtol, absolute_tolerance = atol,
            max_iterations = max_it; kwarg...)
    return lsolve
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

