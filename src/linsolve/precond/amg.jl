"""
    AMGPreconditioner(method = :hmis; kwargs...)

Jutul preconditioner wrapper for the backend-portable algebraic multigrid
implementation in [`KAPreconditioners`](@ref). The default uses HMIS
coarsening with an ILU(0) smoother. The supported compatibility methods are
`:hmis`, `:smoothed_aggregation`, `:aggregation`, and `:ruge_stuben`.
"""
mutable struct AMGPreconditioner{O} <: JutulPreconditioner
    options::O
    factor
    dim
    reuse::Symbol
end

function amg_coarsening(method::Symbol, theta)
    if method == :ruge_stuben
        return KAPreconditioners.RugeStuben(theta)
    elseif method == :smoothed_aggregation || method == :aggregation
        return KAPreconditioners.Aggregation(theta)
    elseif method == :hmis
        return KAPreconditioners.HMIS(theta)
    else
        throw(ArgumentError("Unsupported AMG method: $method"))
    end
end

function ka_smoother(method::Symbol; steps = 1, damping = 1.0)
    if method == :default || method == :spai0
        return KAPreconditioners.SPAI0(steps, damping)
    elseif method == :ilu0
        return KAPreconditioners.ILU0(steps, damping)
    elseif method == :dilu
        return KAPreconditioners.DILU(steps, damping)
    else
        throw(ArgumentError("Unsupported KA smoother: $method"))
    end
end

function AMGPreconditioner(method = :hmis;
        smoother_type::Symbol = :ilu0,
        smoother = nothing,
        cycle = :V,
        npre::Int = 1,
        npost::Int = npre,
        theta = 0.5,
        theta_agg = 0.25,
        max_coarse = 50,
        coarse_size = max_coarse,
        reuse::Symbol = :operators,
        damping = 1.0,
        kwarg...)
    npre == npost || throw(ArgumentError(
        "KAPreconditioners currently requires equal pre- and post-smoothing steps"))
    if isnothing(smoother)
        smoother = ka_smoother(smoother_type; steps = npre, damping = damping)
    end
    cycle in (:V, :W) || throw(ArgumentError("cycle must be :V or :W"))
    if method isa Jutul.KAPreconditioners.AbstractCoarsening
        coarsening = method
    else
        if method == :aggregation || method == :smoothed_aggregation
            coarsening = amg_coarsening(method, theta_agg)
        else
            coarsening = amg_coarsening(method, theta)
        end
    end
    options = KAPreconditioners.AMGOptions(;
        coarsening = coarsening,
        smoother = smoother,
        coarse_size = coarse_size,
        cycle = cycle,
        kwarg...
    )
    return AMGPreconditioner(options, nothing, nothing, reuse)
end

function update_preconditioner!(amg::AMGPreconditioner, A, b, context, executor)
    if isnothing(amg.factor)
        amg.factor = setup_ka_amg(A, amg.options)
        amg.dim = (length(b), length(b))
    else
        update_ka_amg!(amg.factor, A, amg.reuse)
    end
    return amg
end

function partial_update_preconditioner!(amg::AMGPreconditioner,
        A, b, context, executor)
    isnothing(amg.factor) &&
        return update_preconditioner!(amg, A, b, context, executor)
    update_ka_amg!(amg.factor, A, amg.reuse)
    return amg
end

operator_nrows(amg::AMGPreconditioner) = amg.dim[1]

function apply!(x, amg::AMGPreconditioner, y, alpha = 1.0, beta = 0.0)
    if iszero(beta)
        apply_ka_amg!(x, amg.factor, y)
        isone(alpha) || lmul!(alpha, x)
    else
        previous = copy(x)
        apply_ka_amg!(x, amg.factor, y)
        @. x = alpha*x + beta*previous
    end
    return x
end

"""
    KASmootherPreconditioner(config = KAPreconditioners.SPAI0())

Wrap a backend-portable smoother in Jutul's preconditioner lifecycle. A symbol
(`:spai0`, `:ilu0`, or `:dilu`) can be supplied instead of a smoother config.
"""
mutable struct KASmootherPreconditioner{C} <: JutulPreconditioner
    config::C
    factor
    dim
end

KASmootherPreconditioner(config = KAPreconditioners.SPAI0()) =
    KASmootherPreconditioner(config, nothing, nothing)

function KASmootherPreconditioner(method::Symbol; steps = 1, damping = 1.0)
    config = ka_smoother(method; steps = steps, damping = damping)
    return KASmootherPreconditioner(config)
end

function update_preconditioner!(smoother::KASmootherPreconditioner,
        A, b, context, executor)
    if isnothing(smoother.factor)
        smoother.factor = setup_ka_smoother(A, smoother.config)
        factor_type = eltype(smoother.factor)
        degrees_per_row = factor_type <: StaticMatrix ? size(factor_type, 1) : 1
        n = degrees_per_row*size(A, 1)
        smoother.dim = (n, n)
    else
        update_ka_smoother!(smoother.factor, A)
    end
    return smoother
end

function partial_update_preconditioner!(smoother::KASmootherPreconditioner,
        A, b, context, executor)
    return update_preconditioner!(smoother, A, b, context, executor)
end

operator_nrows(smoother::KASmootherPreconditioner) = smoother.dim[1]

function ka_smoother_vectors(smoother, x, y)
    factor_type = eltype(smoother.factor)
    if factor_type <: StaticMatrix && eltype(x) <: Real
        block_size = size(factor_type, 1)
        length(x) % block_size == 0 || throw(DimensionMismatch(
            "output length is not divisible by the smoother block size"))
        length(y) == length(x) || throw(DimensionMismatch(
            "right-hand side and output must have equal lengths"))
        scalar_type = eltype(factor_type)
        vector_type = SVector{block_size, scalar_type}
        x = unsafe_reinterpret(vector_type, x, length(x) ÷ block_size)
        y = unsafe_reinterpret(vector_type, y, length(y) ÷ block_size)
    end
    return x, y
end

function apply!(x, smoother::KASmootherPreconditioner,
        y, alpha = 1.0, beta = 0.0)
    smoother_x, smoother_y = ka_smoother_vectors(smoother, x, y)
    if iszero(beta)
        apply_ka_smoother!(smoother_x, smoother.factor, smoother_y)
        isone(alpha) || lmul!(alpha, x)
    else
        previous = copy(x)
        apply_ka_smoother!(smoother_x, smoother.factor, smoother_y)
        @. x = alpha*x + beta*previous
    end
    return x
end
