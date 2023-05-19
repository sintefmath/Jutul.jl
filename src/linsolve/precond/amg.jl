
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

matrix_for_amg(A) = A
matrix_for_amg(A::StaticSparsityMatrixCSR) = copy(A.At')

function update_preconditioner!(amg::AMGPreconditioner{flavor}, A, b, context) where flavor
    kw = amg.method_kwarg
    A_amg = matrix_for_amg(A)
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
    t_amg = @elapsed multilevel = gen(A_amg)
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
    pre = (A, x, b) -> apply_smoother!(x, A, b, S, amg.npre)
    post = (A, x, b) -> apply_smoother!(x, A, b, S, amg.npost)

    A_c = to_csr(h.final_A)
    factor = factorize_coarse(A_c)
    coarse_solver = (x, b) -> solve_coarse_internal!(x, A_c, factor, b)

    levels = AlgebraicMultigrid.MultiLevel(levels, A_c, coarse_solver, pre, post, h.workspace)
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
        update_preconditioner!(prec, A, b, context)
        push!(smoothers, (precond = prec, x = zeros(N), b = b, context = context))
        push!(sizes, N)
    end
    typed_smoothers = Tuple(smoothers)
    sizes = tuple(sizes...)
    return (n = sizes, smoothers = typed_smoothers)
end

function apply_smoother!(x, A, b, smoothers::NamedTuple, nsmooth)
    if nsmooth == 0
        return x
    end
    m = length(x)
    for (i, n) in enumerate(smoothers.n)
        if m == n
            smooth = smoothers.smoothers[i]
            S = get_factorization(smooth.precond)
            res = smooth.x
            B = smooth.b
            # In-place version of B = b - A*x
            B .= b
            mul!(B, A, x, -1, true)
            for it = 1:nsmooth
                ldiv!(res, S, B)
                @. x += res
                if it < nsmooth
                    mul!(B, A, res, -1, true)
                end
            end
            return x
        end
    end
    error("Unable to match smoother to matrix: Recieved $m by $m matrix, with smoother sizes $(smoothers.n)")
end

function partial_update_preconditioner!(amg::AMGPreconditioner, A, b, context)
    @tic "coarse update" amg.hierarchy = update_hierarchy!(amg, amg.hierarchy, A)
    @tic "smoother update" amg.smoothers = update_smoothers!(amg.smoothers, A, amg.hierarchy.multilevel)
    amg.factor = aspreconditioner(amg.hierarchy.multilevel, amg.cycle)
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

function update_coarse_system!(A_c, R, A::StaticSparsityMatrixCSR, P, M, amg)
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

function update_coarse_system!(A_c, R, A::StaticSparsityMatrixCSR, P, M, amg::AMGPreconditioner{:aggregation})
    if false
        cols = colvals(A_c)
        nz = nonzeros(A_c)
        mb = minbatch(A_c)
        @batch minbatch = mb for row in axes(A_c, 1)
            @inbounds for ptr in nzrange(A_c, row)
                col = cols[ptr]
                nz[ptr] = aggregation_coarse_ij(A, R, row, col)
            end
        end
    elseif false
        At = A.At
        Pt = R.At
        Rt = P.At
        # CSC <- CSR * CSC
        # in_place_mat_mat_mul!(M, A, Rt)
        # M = A*P -> M' = (AP)' = P'A' = RA'
        M_csr_transpose = StaticSparsityMatrixCSR(M, nthreads = nthreads(A), minbatch = minbatch(A))
        agg_in_place_mat_mat_mul!(M_csr_transpose, R, A.At)
        # CSR <- CSR * CSC
        agg_in_place_mat_mat_mul!(A_c, R, M)
    else
        At = A.At
        Pt = R.At
        Rt = P.At
        A_c_t = Rt*At*Pt
        nonzeros(A_c.At) .= nonzeros(A_c_t)
    end
    return A_c
end

function aggregation_coarse_ij(A, R, row, col)
    A_rowptr = SparseArrays.getcolptr(A.At)
    R_rowptr = SparseArrays.getcolptr(R.At)

    @inbounds start_col = R_rowptr[col]
    @inbounds stop_col  = R_rowptr[col+1]-1
    R_cols = colvals(R)

    @inbounds start_R = R_rowptr[row]
    @inbounds stop_R = R_rowptr[row+1]-1
    # sum A_ij for i in I and j in J
    v = zero(eltype(A))
    @inbounds for pos in start_col:stop_col
        inner_row = R_cols[pos]

        start_A = A_rowptr[inner_row]
        stop_A = A_rowptr[inner_row+1]-1

        v += aggregation_coarse_ij_row(A, start_A, stop_A, nonzeros(A), colvals(A), start_R, stop_R, R_cols)
    end
    return v
end

function aggregation_coarse_ij_row(A, start_A, stop_A, val_A, ix_A, start_R, stop_R, ix_R)
    v = zero(eltype(A))
    pos_A = start_A
    pos_R = start_R

    @inbounds i_A = ix_A[pos_A]
    @inbounds i_R = ix_R[pos_R]
    @inbounds while true
        delta = i_A - i_R
        if i_A == i_R
            v += val_A[pos_A]
        end
        if delta >= 0
            if pos_R < stop_R
                pos_R += 1
                @inbounds i_R = ix_R[pos_R]
            else
                break
            end
        end
        if delta <= 0
            if pos_A < stop_A
                pos_A += 1
                @inbounds i_A = ix_A[pos_A]
            else
                break
            end
        end
    end
    return v
end

function update_smoothers!(smoothers::Nothing, A, h)

end

function update_smoothers!(S::NamedTuple, A::StaticSparsityMatrixCSR, h)
    n = length(h.levels)
    for i = 1:n
        S_i = S.smoothers[i]
        update_preconditioner!(S_i.precond, A, S_i.b, S_i.context)
        if i < n
            A = h.levels[i+1].A
        end
    end
    return S
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


function agg_in_place_mat_mat_mul!(M::CSR, A::CSR, B::CSC) where {CSR<:StaticSparsityMatrixCSR, CSC<:SparseMatrixCSC}
    columns = colvals(M)
    nz = nonzeros(M)
    n = size(M, 1)
    mb = max(n ÷ nthreads(A), minbatch(A))
    @batch minbatch = mb for row in 1:n
        for pos in nzrange(M, row)
            @inbounds col = columns[pos]
            @inbounds nz[pos] = rowcol_prod(A, B, row, col)
        end
    end
end

function agg_in_place_mat_mat_mul!(M::CSC, A::CSR, B::CSC) where {CSR<:StaticSparsityMatrixCSR, CSC<:SparseMatrixCSC}
    rows = rowvals(M)
    nz = nonzeros(M)
    n = size(M, 2)
    mb = max(n ÷ nthreads(A), minbatch(A))
    @batch minbatch = mb for col in 1:n
        for pos in nzrange(M, col)
            @inbounds row = rows[pos]
            @inbounds nz[pos] = rowcol_prod(A, B, row, col)
        end
    end
end

@inline function rowcol_prod(A::StaticSparsityMatrixCSR, B::SparseMatrixCSC, row, col)
    # We know both that this product is nonzero
    # First matrix, iterate over columns
    A_range = nzrange(A, row)
    nz_A = nonzeros(A)
    n_col = length(A_range)
    columns = colvals(A)
    @inline new_column(pos) = sparse_indirection(columns, A_range, pos)

    # Second matrix, iterate over row
    B_range = nzrange(B, col)
    nz_B = nonzeros(B)
    n_row = length(B_range)
    rows = rowvals(B)
    new_row(pos) = sparse_indirection(rows, B_range, pos)
    # Initialize
    pos_A = pos_B = 1
    current_col, A_idx = new_column(pos_A)
    current_row, B_idx = new_row(pos_B)
    v = zero(eltype(A))
    entries_remain = true
    while entries_remain
        delta = current_row - current_col
        if delta == 0
            @inbounds rv = nz_A[A_idx]
            @inbounds cv = nz_B[B_idx]
            v += rv*cv
            entries_remain = pos_A < n_col && pos_B < n_row
            if entries_remain
                pos_A += 1
                current_col, A_idx = new_column(pos_A)
                pos_B += 1
                current_row, B_idx = new_row(pos_B)
            end
        elseif delta > 0
            entries_remain = pos_A < n_col
            if entries_remain
                pos_A += 1
                current_col, A_idx = new_column(pos_A)
            end
        else
            entries_remain = pos_B < n_row
            if entries_remain
                pos_B += 1
                current_row, B_idx = new_row(pos_B)
            end
        end
    end
    return v
end

@inline function sparse_indirection(val, rng, pos)
    @inbounds ix = rng[pos]
    @inbounds v = val[ix]
    return (v, ix)
end

function rowcol_prod(A::SparseMatrixCSC, B::StaticSparsityMatrixCSR, row, col)
    v = zero(eltype(A))
    error()
end
