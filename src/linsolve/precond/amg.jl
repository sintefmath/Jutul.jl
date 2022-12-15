
"""
AMG on CPU (Julia native)
"""
mutable struct AMGPreconditioner <: JutulPreconditioner
    method
    method_kwarg
    cycle
    factor
    dim
    hierarchy
    smoothers
    smoother_type::Symbol
    function AMGPreconditioner(method = ruge_stuben; smoother_type = :default, cycle = AlgebraicMultigrid.V(), kwarg...)
        if method == :ruge_stuben
            method = ruge_stuben
        elseif method == :smoothed_aggregation
            method = smoothed_aggregation
        end
        new(method, kwarg, cycle, nothing, nothing, nothing, nothing, smoother_type)
    end
end

matrix_for_amg(A) = A
matrix_for_amg(A::StaticSparsityMatrixCSR) = copy(A.At')

function update!(amg::AMGPreconditioner, A, b, context)
    kw = amg.method_kwarg
    A_amg = matrix_for_amg(A)
    @debug string("Setting up preconditioner ", amg.method)
    t_amg = @elapsed multilevel = amg.method(A_amg; kw...)
    amg = specialize_multilevel!(amg, multilevel, A, context)
    amg.dim = size(A)
    @debug "Set up AMG in $t_amg seconds."
    amg.factor = aspreconditioner(amg.hierarchy.multilevel, amg.cycle)
end

function specialize_multilevel!(amg, multilevel, A, context)
    amg.hierarchy = (multilevel = multilevel, buffers = nothing)
    return amg
end

function specialize_multilevel!(amg, h, A::StaticSparsityMatrixCSR, context)
    A_f = A
    nt = A.nthreads
    mb = A.minbatch
    to_csr(M) = StaticSparsityMatrixCSR(M, nthreads = nt, minbatch = mb)
    levels = h.levels
    new_levels = []
    n = length(levels)
    buffers = Vector{typeof(A.At)}()
    sizehint!(buffers, n)
    if n > 0
        for i = 1:n
            l = levels[i]
            P0, R0 = l.P, l.R
            # Convert P to a proper CSC matrix to get parallel
            # spmv for the wrapper type, trading some memory for
            # performance.
            if isa(P0, Adjoint)
                @assert R0 === P0'
                P0 = copy(P0)
            elseif isa(R0, Adjoint)
                @assert R0' === P0
                R0 = copy(R0)
            end
            R = to_csr(P0)
            P = to_csr(R0)
            if i > 1
                A = to_csr(copy(l.A'))
            end
            # Add first part of R*A*P to buffer as CSC
            push!(buffers, (A.At')*P0)
            lvl = AlgebraicMultigrid.Level(A, P, R)
            push!(new_levels, lvl)
        end
        typed_levels = Vector{typeof(new_levels[1])}(undef, n)
        for i = 1:n
            typed_levels[i] = new_levels[i]
        end
        levels = typed_levels
    else
        # There are no levels (case is tiny)
        T = typeof(A_f)
        levels = Vector{AlgebraicMultigrid.Level{T, T, T}}()
    end
    S = amg.smoothers = generate_amg_smoothers(amg.smoother_type, A_f, levels, context)
    smoother = (A, x, b) -> apply_smoother!(x, A, b, S)

    A_c = to_csr(h.final_A)
    factor = factorize_coarse(A_c)
    coarse_solver = (x, b) -> solve_coarse_internal!(x, A_c, factor, b)

    levels = AlgebraicMultigrid.MultiLevel(levels, A_c, coarse_solver, smoother, smoother, h.workspace)
    amg.hierarchy = (multilevel = levels, buffers = buffers)
    return amg
end

function solve_coarse_internal!(x, A, factor, b)
    x = ldiv!(x, factor, b)
    return x
end

function generate_amg_smoothers(t, A_fine, levels, context)
    sizes = Vector{Int64}()
    smoothers = []
    n = length(levels)
    for i = 1:n
        A = levels[i].A
        N = size(A, 1)
        b = zeros(N)
        if t == :default || t == :spai0
            prec = SPAI0Preconditioner()
        elseif t == :ilu0
            prec = ILUZeroPreconditioner()
        elseif t == :jacobi
            prec = JacobiPreconditioner()
        else
            error("Smoother :$t is not supported.")
        end
        update!(prec, A, b, context)
        push!(smoothers, (precond = prec, x = zeros(N), b = b, context = context))
        push!(sizes, N)
    end
    typed_smoothers = Tuple(smoothers)
    sizes = tuple(sizes...)
    return (n = sizes, smoothers = typed_smoothers)
end

function apply_smoother!(x, A, b, smoothers::NamedTuple)
    m = length(x)
    for (i, n) in enumerate(smoothers.n)
        if m == n
            smooth = smoothers.smoothers[i]
            S = get_factorization(smooth.precond)
            res = smooth.x
            B = smooth.b
            # In-place version of B = b - A*x
            B .= b
            mul!(B, A, x, -1, 1)
            ldiv!(res, S, B)
            @. x += res
            return x
        end
    end
    error("Unable to match smoother to matrix: Recieved $m by $m matrix, with smoother sizes $(smoothers.n)")
end

function partial_update!(amg::AMGPreconditioner, A, b, context)
    @timeit "coarse update" amg.hierarchy = update_hierarchy!(amg, amg.hierarchy, A)
    @timeit "smoother update" amg.smoothers = update_smoothers!(amg.smoothers, A, amg.hierarchy.multilevel)
end

operator_nrows(amg::AMGPreconditioner) = amg.dim[1]

factorize_coarse(A) = lu(A)

function factorize_coarse(A::StaticSparsityMatrixCSR)
    return lu(A.At)'
end

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
        A = update_coarse_system!(A_c, R, A, P, buf)
    end
    factor = factorize_coarse(A)
    coarse_solver = (x, b) -> solve_coarse_internal!(x, A, factor, b)
    S = amg.smoothers
    if isnothing(S)
        pre = h.presmoother
        post = h.postsmoother
    else
        pre = post = (A, x, b) -> apply_smoother!(x, A, b, S)
    end
    multilevel = AlgebraicMultigrid.MultiLevel(levels, A, coarse_solver, pre, post, h.workspace)
    return (multilevel = multilevel, buffers = buffers)
end

function print_system(A::StaticSparsityMatrixCSR)
    print_system(SparseMatrixCSC(A.At'))
end

function print_system(A)
    I, J, V = findnz(A)
    @info "Coarsest system"  size(A)
    for (i, j, v) in zip(I, J, V)
        @info "$i $j: $v"
    end
end

function update_coarse_system!(A_c, R, A, P, buffer)
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

function update_coarse_system!(A_c, R, A::StaticSparsityMatrixCSR, P, M)
    if false
        At = A.At
        Pt = R.At
        Rt = P.At
        # CSC <- CSR * CSC
        # in_place_mat_mat_mul!(M, A, Rt)
        # M = A*P -> M' = (AP)' = P'A' = RA'
        M_csr_transpose = StaticSparsityMatrixCSR(M, nthreads = nthreads(A), minbatch = minbatch(A))
        in_place_mat_mat_mul!(M_csr_transpose, R, A.At)
        # CSR <- CSR * CSC
        in_place_mat_mat_mul!(A_c, R, M)
    else
        At = A.At
        Pt = R.At
        Rt = P.At
        A_c_t = Rt*At*Pt
        nonzeros(A_c.At) .= nonzeros(A_c_t)
    end
    return A_c
end

function update_smoothers!(smoothers::Nothing, A, h)

end

function update_smoothers!(S::NamedTuple, A::StaticSparsityMatrixCSR, h)
    n = length(h.levels)
    for i = 1:n
        S_i = S.smoothers[i]
        update!(S_i.precond, A, S_i.b, S_i.context)
        if i < n
            A = h.levels[i+1].A
        end
    end
    return S
end
