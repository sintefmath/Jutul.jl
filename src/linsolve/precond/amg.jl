
"""
AMG on CPU (Julia native)
"""
mutable struct AMGPreconditioner{T} <: JutulPreconditioner
    method_kwarg
    cycle
    factor
    dim
    hierarchy
    smoothers
    smoother_type::Symbol
    npre::Int
    npost::Int
    function AMGPreconditioner(method::Symbol; smoother_type = :default, cycle = AlgebraicMultigrid.V(), npre = 1, npost = npre, kwarg...)
        @assert method == :smoothed_aggregation || method == :ruge_stuben || method == :aggregation
        new{method}(kwarg, cycle, nothing, nothing, nothing, nothing, smoother_type, npre, npost)
    end
end

function update_preconditioner!(amg::AMGPreconditioner{flavor}, A, b, context, executor) where flavor
    kw = amg.method_kwarg
    @debug string("Setting up preconditioner ", flavor)
    pre = GaussSeidel(iter = amg.npre)
    post = GaussSeidel(iter = amg.npost)
    sarg = (presmoother = pre, postsmoother = post)
    if flavor == :smoothed_aggregation
        gen = (A) -> smoothed_aggregation(A; sarg..., kw...)
    elseif flavor == :ruge_stuben
        gen = (A) -> ruge_stuben(A; sarg..., kw...)
    elseif flavor == :aggregation
        gen = (A) -> plain_aggregation(A; sarg..., kw...)
    end
    t_amg = @elapsed multilevel = gen(A)
    amg.hierarchy = (multilevel = multilevel, buffers = nothing)
    amg.dim = size(A)
    @debug "Set up AMG in $t_amg seconds."
    amg.factor = aspreconditioner(amg.hierarchy.multilevel, amg.cycle)
end

function solve_coarse_internal!(x, A, factor, b)
    x = ldiv!(x, factor, b)
    return x
end

function partial_update_preconditioner!(amg::AMGPreconditioner, A, b, context, executor)
    @tic "coarse update" amg.hierarchy = update_hierarchy!(amg, amg.hierarchy, A)
    @tic "smoother update" amg.smoothers = update_smoothers!(amg.smoothers, A, amg.hierarchy.multilevel)
    amg.factor = aspreconditioner(amg.hierarchy.multilevel, amg.cycle)
end

operator_nrows(amg::AMGPreconditioner) = amg.dim[1]

factorize_coarse(A) = lu(A)

function update_hierarchy!(amg, hierarchy, A)
    h = hierarchy.multilevel
    buffers = hierarchy.buffers
    levels = h.levels
    n = length(levels)
    for i = 1:n
        l = levels[i]
        P, R = l.P, l.R
        # Remake level in case A has been reallocated
        levels[i] = AlgebraicMultigrid.Level(A, P, R)
        if i == n
            A_c = h.final_A
        else
            A_c = levels[i+1].A
        end
        buf = isnothing(buffers) ? nothing : buffers[i]
        A = update_coarse_system!(A_c, R, A, P, buf, amg)
    end
    factor = factorize_coarse(A)
    coarse_solver = (x, b) -> solve_coarse_internal!(x, A, factor, b)
    S = amg.smoothers
    if isnothing(S)
        pre = h.presmoother
        post = h.postsmoother
    else
        pre = (A, x, b) -> apply_smoother!(x, A, b, S, amg.npre)
        post = (A, x, b) -> apply_smoother!(x, A, b, S, amg.npost)
    end
    multilevel = AlgebraicMultigrid.MultiLevel(levels, A, coarse_solver, pre, post, h.workspace)
    return (multilevel = multilevel, buffers = buffers)
end

function print_system(A)
    I, J, V = findnz(A)
    @info "Coarsest system"  size(A)
    for (i, j, v) in zip(I, J, V)
        @info "$i $j: $v"
    end
end

function update_coarse_system!(A_c, R, A, P, buffer, amg)
    # In place modification
    nz = nonzeros(A_c)
    A_c_next = R*A*P
    nz_next = nonzeros(A_c)
    if length(nz_next) == length(nz)
        nz .= nz_next
    else
        # Sparsity pattern has changed. Hope that the caller doesn't rely on
        # in-place updates.
        A_c = A_c_next
    end
    return A_c
end

function update_smoothers!(smoothers::Nothing, A, h)

end

function plain_aggregation(A::TA, 
                        ::Type{Val{bs}}=Val{1};
                        symmetry = HermitianSymmetry(),
                        strength = SymmetricStrength(),
                        aggregate = StandardAggregation(),
                        presmoother = GaussSeidel(),
                        postsmoother = GaussSeidel(),
                        max_levels = 10,
                        max_coarse = 10,
                        diagonal_dominance = false,
                        keep = false,
                        coarse_solver = AlgebraicMultigrid.Pinv, kwargs...) where {T,V,bs,TA<:SparseMatrixCSC{T,V}}

    n = size(A, 1)
    B = ones(T,n)

    levels = Vector{AlgebraicMultigrid.Level{TA, TA, Adjoint{T, TA}}}()
    bsr_flag = false
    w = AlgebraicMultigrid.MultiLevelWorkspace(Val{bs}, eltype(A))
    AlgebraicMultigrid.residual!(w, size(A, 1))

    while length(levels) + 1 < max_levels && size(A, 1) > max_coarse
        A, B, bsr_flag = extend_hierarchy!(levels, strength, aggregate,
                                            diagonal_dominance, keep, A, B, symmetry, bsr_flag)
                                            AlgebraicMultigrid.coarse_x!(w, size(A, 1))
        AlgebraicMultigrid.coarse_b!(w, size(A, 1))
        AlgebraicMultigrid.residual!(w, size(A, 1))
    end
    AlgebraicMultigrid.MultiLevel(levels, A, coarse_solver(A), presmoother, postsmoother, w)
end

struct HermitianSymmetry
end

function extend_hierarchy!(levels, strength, aggregate, diagonal_dominance, keep,
                            A, B,
                            symmetry, bsr_flag)

    # Calculate strength of connection matrix
    if symmetry isa HermitianSymmetry
        S, _T = strength(A, bsr_flag)
    else
        S, _T = strength(adjoint(A), bsr_flag)
    end

    # Aggregation operator
    P = copy(aggregate(S)')
    R = construct_R(symmetry, P)
    push!(levels, AlgebraicMultigrid.Level(A, P, R))

    A = R * A * P

    dropzeros!(A)

    bsr_flag = true

    A, B, bsr_flag
end
construct_R(::HermitianSymmetry, P) = P'
