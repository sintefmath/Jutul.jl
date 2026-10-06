"""
    AMGPreconditioner(method = :hmis; kwargs...)

Jutul preconditioner wrapper for the backend-portable algebraic multigrid
implementation in the internal `KAPreconditioners` module. The default uses HMIS
coarsening with an ILU(0) smoother. The supported compatibility methods are
`:hmis`, `:aggregation`, and `:ruge_stuben`. They default to Extended+i,
piecewise-constant, and classical interpolation, respectively.

The other options of the AMG preconditioner correspond to the fields of
`AMGOptions` and control various aspects of the multigrid hierarchy, such as the
coarsening strategy, interpolation method, smoother configuration, and cycle
type. These are not a public API and are subject to change without notice or
major version bump.

`aggressive_levels` defaults to zero. A positive value applies a HYPRE-style
second coarsening pass on that many levels, starting with the finest. RS/HMIS
use two-stage Extended+i interpolation on those levels, composing fine-to-first
coarse interpolation with partial interpolation to the final coarse set.

`theta` defaults to 0.25 for `:ruge_stuben` and 0.5 for `:hmis`.
`second_pass` overrides the classical RS second pass when the method is
Ruge-Stuben. The shared `strength_type` can be `:signed_fallback` (default),
`:signed`, or `:absolute`. `aggressive_num_paths` (default 1) controls the
second strength graph for RS/HMIS. `aggressive_interpolation` accepts an
`TwoStageExtendedIInterpolation(...)` configuration with independent factor and
product truncation and row limits. Explicit `ExtendedIInterpolation(...)` selects
the legacy distance-two path. See `AMGOptions` for these settings.

With `reuse=:partial_operators` or `:partial_sparsity`,
`n_levels_partial_keep` controls how many leading levels retain their
symbolic structure (default 3). `n_partial_keep` can shorten that prefix
at the first level whose matrix size is below the specified value. Its
default of -1 disables the size limit. The remaining levels rebuild their
symbolic data on each full update.
"""
mutable struct AMGPreconditioner{O} <: JutulPreconditioner
    options::O
    factor
    dim
    reuse::Symbol
    reuse_partial::Symbol
    n_levels_partial_keep::Int
    n_partial_keep::Int
end

function amg_coarsening(method::Symbol, theta; second_pass::Bool = true)
    if method == :ruge_stuben
        return KAPreconditioners.RugeStuben(theta; second_pass = second_pass)
    elseif method == :aggregation
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
    elseif method == :gauss_seidel
        return KAPreconditioners.GaussSeidel(steps, damping)
    elseif method == :ilu0
        return KAPreconditioners.ILU0(steps, damping)
    elseif method == :dilu
        return KAPreconditioners.DILU(steps, damping)
    elseif method == :vendor_ilu
        return KAPreconditioners.VendorILU(steps, damping)
    else
        throw(ArgumentError("Unsupported KA smoother: $method"))
    end
end

function AMGPreconditioner(
        method = :hmis;
        smoother_type::Symbol = :spai0,
        smoother = nothing,
        cycle = :V,
        npre::Int = 1,
        npost::Int = npre,
        theta = nothing,
        second_pass::Union{Nothing, Bool} = nothing,
        theta_agg = 0.25,
        coarse_size = 5,
        reuse::Symbol = :memory,
        reuse_partial::Symbol = :operators,
        n_levels_partial_keep::Integer = 2,
        n_partial_keep::Integer = -1,
        damping = 1.0,
        kwarg...
    )
    npre == npost || throw(
        ArgumentError(
            "KAPreconditioners currently requires equal pre- and post-smoothing steps"
        )
    )
    if isnothing(smoother)
        smoother = ka_smoother(smoother_type; steps = npre, damping = damping)
    end
    cycle in (:V, :W) || throw(ArgumentError("cycle must be :V or :W"))
    n_levels_partial_keep >= 0 ||
        throw(ArgumentError("n_levels_partial_keep must be non-negative"))
    (n_partial_keep == -1 || n_partial_keep > 0) ||
        throw(ArgumentError("n_partial_keep must be -1 or positive"))
    if method isa Jutul.KAPreconditioners.AbstractCoarsening
        coarsening = method
    else
        if method == :aggregation
            coarsening = amg_coarsening(method, theta_agg)
        else
            if isnothing(theta)
                theta = method == :ruge_stuben ? 0.25 : 0.5
            end
            coarsening = amg_coarsening(method, theta)
        end
    end
    if !isnothing(second_pass)
        coarsening isa KAPreconditioners.RugeStuben || throw(
            ArgumentError(
                "second_pass only applies to RugeStuben coarsening"
            )
        )
        coarsening = KAPreconditioners.RugeStuben(coarsening.theta; second_pass = second_pass)
    end
    options = KAPreconditioners.AMGOptions(;
        coarsening = coarsening,
        smoother = smoother,
        coarse_size = coarse_size,
        cycle = cycle,
        kwarg...
    )
    return AMGPreconditioner(
        options, nothing, nothing, reuse, reuse_partial,
        Int(n_levels_partial_keep), Int(n_partial_keep)
    )
end

function update_preconditioner!(amg::AMGPreconditioner, A, b, context, executor)
    if isnothing(amg.factor)
        amg.factor = setup_ka_amg(A, amg.options)
        amg.dim = (length(b), length(b))
    else
        update_ka_amg!(
            amg.factor, A, amg.reuse;
            n_levels_partial_keep = amg.n_levels_partial_keep,
            n_partial_keep = amg.n_partial_keep
        )
    end
    return amg
end

function partial_update_preconditioner!(
        amg::AMGPreconditioner,
        A, b, context, executor
    )
    isnothing(amg.factor) &&
        return update_preconditioner!(amg, A, b, context, executor)
    update_ka_amg!(
        amg.factor, A, amg.reuse_partial;
        n_levels_partial_keep = amg.n_levels_partial_keep,
        n_partial_keep = amg.n_partial_keep
    )
    return amg
end

operator_nrows(amg::AMGPreconditioner) = amg.dim[1]

function apply!(x, amg::AMGPreconditioner, y, alpha = 1.0, beta = 0.0)
    T = KAPreconditioners.matrix_scalar_type(eltype(x))
    alpha = convert(T, alpha)
    beta = convert(T, beta)
    if iszero(beta)
        apply_ka_amg!(x, amg.factor, y)
        isone(alpha) || lmul!(alpha, x)
    else
        previous = copy(x)
        apply_ka_amg!(x, amg.factor, y)
        @. x = alpha * x + beta * previous
    end
    return x
end

"""
    KASmootherPreconditioner(config = KAPreconditioners.SPAI0())

Wrap a backend-portable smoother in Jutul's preconditioner lifecycle. A symbol
(`:spai0`, `:gauss_seidel`, `:ilu0`, `:dilu`, or `:vendor_ilu`) can be supplied
instead of a smoother config.
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

function update_preconditioner!(
        smoother::KASmootherPreconditioner,
        A, b, context, executor
    )
    if isnothing(smoother.factor)
        smoother.factor = setup_ka_smoother(A, smoother.config)
        factor_type = eltype(smoother.factor)
        if factor_type <: StaticMatrix
            degrees_per_row = size(factor_type, 1)
        else
            degrees_per_row = 1
        end
        n = degrees_per_row * size(A, 1)
        smoother.dim = (n, n)
    else
        update_ka_smoother!(smoother.factor, A)
    end
    return smoother
end

function partial_update_preconditioner!(
        smoother::KASmootherPreconditioner,
        A, b, context, executor
    )
    return update_preconditioner!(smoother, A, b, context, executor)
end

operator_nrows(smoother::KASmootherPreconditioner) = smoother.dim[1]

function ka_smoother_vectors(smoother, x, y)
    factor_type = eltype(smoother.factor)
    if factor_type <: StaticMatrix && eltype(x) <: Real
        block_size = size(factor_type, 1)
        length(x) % block_size == 0 || throw(
            DimensionMismatch(
                "output length is not divisible by the smoother block size"
            )
        )
        length(y) == length(x) || throw(
            DimensionMismatch(
                "right-hand side and output must have equal lengths"
            )
        )
        scalar_type = eltype(factor_type)
        vector_type = SVector{block_size, scalar_type}
        x = unsafe_reinterpret(vector_type, x, length(x) ÷ block_size)
        y = unsafe_reinterpret(vector_type, y, length(y) ÷ block_size)
    end
    return x, y
end

function apply!(
        x, smoother::KASmootherPreconditioner,
        y, alpha = 1.0, beta = 0.0
    )
    T = KAPreconditioners.matrix_scalar_type(eltype(x))
    alpha = convert(T, alpha)
    beta = convert(T, beta)
    smoother_x, smoother_y = ka_smoother_vectors(smoother, x, y)
    if iszero(beta)
        apply_ka_smoother!(smoother_x, smoother.factor, smoother_y)
        isone(alpha) || lmul!(alpha, x)
    else
        previous = copy(x)
        apply_ka_smoother!(smoother_x, smoother.factor, smoother_y)
        @. x = alpha * x + beta * previous
    end
    return x
end
