export ILUZeroPreconditioner, SPAI0Preconditioner, LUPreconditioner, GroupWisePreconditioner, TrivialPreconditioner, JacobiPreconditioner, AMGPreconditioner, JutulPreconditioner, apply!

abstract type JutulPreconditioner end
abstract type DiagonalPreconditioner <: JutulPreconditioner end

function update!(preconditioner::Nothing, arg...)
    # Do nothing.
end
function update!(preconditioner, lsys, model, storage, recorder)
    J = jacobian(lsys)
    r = residual(lsys)
    ctx = linear_system_context(model, lsys)
    update!(preconditioner, J, r, ctx)
end

export partial_update!
function partial_update!(p, A, b, context)
    update!(p, A, b, context)
end

function get_factorization(precond)
    return precond.factor
end

is_left_preconditioner(::JutulPreconditioner) = true
is_right_preconditioner(::JutulPreconditioner) = false

function linear_operator(precond::JutulPreconditioner, side::Symbol = :left, float_t = Float64)
    n = operator_nrows(precond)
    function local_mul!(res, x, α, β::T, type) where T
        if β == zero(T)
            apply!(res, precond, x, type)
            if α != one(T)
                lmul!(α, res)
            end
        else
            error("Not implemented yet.")
        end
    end

    if side == :left
        if is_left_preconditioner(precond)
            if is_right_preconditioner(precond)
                f! = (r, x, α, β) -> local_mul!(r, x, α, β, :left)
            else
                f! = (r, x, α, β) -> local_mul!(r, x, α, β, :both)
            end
            op = LinearOperator(float_t, n, n, false, false, f!)
        else
            op = opEye(n, n)
        end
    elseif side == :right
        if is_right_preconditioner(precond)
            f! = (r, x, α, β) -> local_mul!(r, x, α, β, :right)
            op = LinearOperator(float_t, n, n, false, false, f!)
        else
            op = opEye(n, n)
        end
    else
        error("Side must be :left or :right, was $side")
    end

    return op
end

function apply!(x, p::JutulPreconditioner, y, arg...)
    factor = get_factorization(p)
    if is_left_preconditioner(p)
        ldiv!(x, factor, y)
    elseif is_right_preconditioner(p)
        error("Not supported.")
    else
        error("Neither left or right preconditioner?")
    end
end

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
    function AMGPreconditioner(method = ruge_stuben; cycle = AlgebraicMultigrid.V(), kwarg...)
        if method == :ruge_stuben
            method = ruge_stuben
        elseif method == :smoothed_aggregation
            method = smoothed_aggregation
        end
        new(method, kwarg, cycle, nothing, nothing, nothing, nothing)
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
    S = amg.smoothers = generate_smoothers_csr(A_f, levels, nt, mb, context)
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

function generate_smoothers_csr(A_fine, levels, nthreads, min_batch, context)
    sizes = Vector{Int64}()
    smoothers = []
    n = length(levels)
    for i = 1:n
        A = levels[i].A
        max_t = max(size(A, 1) ÷ min_batch, 1)
        nt = min(nthreads, max_t)
        if nt == 1
            ilu_factor = ilu0_csr(A)
        else
            lookup = generate_lookup(context.partitioner, A, nt)
            ilu_factor = ilu0_csr(A, lookup)
        end
        N = size(A, 1)
        push!(smoothers, (factor = ilu_factor, x = zeros(N), b = zeros(N)))
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
            S = smooth.factor
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
        ilu0_csr!(S.smoothers[i].factor, A)
        if i < n
            A = h.levels[i+1].A
        end
    end
    return S
end

function update!(jac::DiagonalPreconditioner, A, b, context)
    if isnothing(jac.factor)
        n = size(A, 1)
        D = Vector{eltype(A)}(undef, n)
        for i in 1:n
            D[i] = A[i, i]
        end
        jac.factor = D
        d = length(b[1])
        jac.dim = d .* size(A, 1)
    end
    D = jac.factor
    mb = minbatch(A)
    jac.minbatch = mb
    diagonal_precond!(D, A, jac)
end

function diagonal_precond!(D, A, jac)
    mb = minbatch(A)
    @batch minbatch = mb for i in eachindex(D)
        @inbounds D[i] = diagonal_precond(A, i, jac)
    end
end

function apply!(x, jac::DiagonalPreconditioner, y, arg...)
    D = jac.factor

    s = D[1]
    N = size(s, 1)
    T = eltype(s)
    Vt = SVector{N, T}
    as_svec = (x) -> reinterpret(Vt, x)

    # Solve by reinterpreting vectors to block (=SVector) vectors
    diag_parmul!(as_svec(x), D, as_svec(y), minbatch(jac))
end

function operator_nrows(jac::DiagonalPreconditioner)
    return jac.dim
end

function diag_parmul!(x, D, y, mb)
    @batch minbatch = mb for i in eachindex(x, y, D)
        @inbounds x[i] = D[i]*x[i]
    end
end

"""
Damped Jacobi preconditioner on CPU
"""
mutable struct JacobiPreconditioner <: DiagonalPreconditioner
    factor
    dim::Int64
    w::Float64
    minbatch::Int64
    function JacobiPreconditioner(; w = 2.0/3.0, minbatch = 1000)
        new(nothing, 1, w, minbatch)
    end
end

function diagonal_precond(A, i, jac::JacobiPreconditioner)
    @inbounds A_ii = A[i, i]
    return jac.w*inv(A_ii)
end


mutable struct SPAI0Preconditioner <: DiagonalPreconditioner
    factor
    dim::Int64
    minbatch::Int64
    function SPAI0Preconditioner(; minbatch = 1000)
        new(nothing, 1, minbatch)
    end
end


function diagonal_precond!(D, A::SparseMatrixCSC, jac::SPAI0Preconditioner)
    for i in eachindex(D)
        D[i] = zero(eltype(D))
    end
    buf = zeros(length(D))
    rows = rowvals(A)
    vals = nonzeros(A)
    for col in axes(A, 2)
        for p in nzrange(A, col)
            row = rows[p]
            val = vals[p]
            nv = zero(eltype(val))
            for v in val
                nv += v^2
            end
            buf[row] += nv
        end
    end

    for i in eachindex(D)
        D[i] = inv(buf[i])*A[i, i]
    end
end
"""
ILU(0) preconditioner on CPU
"""
mutable struct ILUZeroPreconditioner <: JutulPreconditioner
    factor
    dim
    left::Bool
    right::Bool
    function ILUZeroPreconditioner(; left = true, right = false)
        @assert left || right "Left or right preconditioning must be enabled or it will have no effect."
        new(nothing, nothing, left, right)
    end
end

is_left_preconditioner(p::ILUZeroPreconditioner) = p.left
is_right_preconditioner(p::ILUZeroPreconditioner) = p.right

function set_dim!(ilu, A, b)
    T = eltype(b)
    if T<:AbstractFloat
        d = 1
    else
        d = length(T)
    end
    ilu.dim = d .* size(A)
end

function update!(ilu::ILUZeroPreconditioner, A, b, context)
    if isnothing(ilu.factor)
        ilu.factor = ilu0(A, eltype(b))
        set_dim!(ilu, A, b)
    else
        ilu0!(ilu.factor, A)
    end
end

function update!(ilu::ILUZeroPreconditioner, A::StaticSparsityMatrixCSR, b, context::ParallelCSRContext)
    if isnothing(ilu.factor)
        mb = A.minbatch
        max_t = max(size(A, 1) ÷ mb, 1)
        nt = min(nthreads(context), max_t)
        if nt == 1
            @debug "Setting up serial ILU(0)-CSR"
            F = ilu0_csr(A)
        else
            @debug "Setting up parallel ILU(0)-CSR with $(nthreads(td)) threads"
            # lookup = td.lookup
            part = context.partitioner
            lookup = generate_lookup(part, A, nt)
            F = ilu0_csr(A, lookup)
        end
        ilu.factor = F
        set_dim!(ilu, A, b)
    else
        ilu0_csr!(ilu.factor, A)
    end
end

function apply!(x, ilu::ILUZeroPreconditioner, y, arg...)
    factor = get_factorization(ilu)
    ilu_apply!(x, factor, y, arg...)
end

function ilu_f(type::Symbol)
    # Why must this be qualified?
    if type == :left
        f = forward_substitution!
    elseif type == :right
        f = backward_substitution!
    else
        f = ldiv!
    end
end

function ilu_apply!(x::AbstractArray{F}, f::AbstractILUFactorization, y::AbstractArray{F}, type::Symbol = :both) where {F<:Real}
    T = eltype(f)
    N = size(T, 1)
    T = eltype(T)
    Vt = SVector{N, T}
    as_svec = (x) -> reinterpret(Vt, x)

    ldiv!(as_svec(x), f, as_svec(y))
    return x
end

function ilu_apply!(x, f::AbstractILUFactorization, y, type::Symbol = :both)
    ldiv!(x, f, y)
end

function ilu_apply!(x::AbstractArray{F}, f::ILU0Precon{F}, y::AbstractArray{F}, type::Symbol = :both) where {F<:Real}
    f! = ilu_f(type)
    f!(x, f, y)
end

# function ilu_apply!(x::AbstractArray{F}, f::CuSparseMatrix{F}, y::AbstractArray{F}, type::Symbol = :both) where {F<:Real}
#     x .= y
#     ix = 'O'
#     sv2!('N', 'L', 'N', 1.0, f, x, ix)
#     sv2!('N', 'U', 'U', 1.0, f, x, ix)
# end

function ilu_apply!(x, ilu::ILU0Precon, y, type::Symbol = :both)
    T = eltype(ilu.l_nzval)
    N = size(T, 1)
    T = eltype(T)
    Vt = SVector{N, T}
    as_svec = (x) -> reinterpret(Vt, x)

    # Solve by reinterpreting vectors to block (=SVector) vectors
    f! = ilu_f(type)
    f!(as_svec(x), ilu, as_svec(y))
    return x
end

function operator_nrows(ilu::ILUZeroPreconditioner)
    return ilu.dim[1]
end

mutable struct TrivialPreconditioner <: JutulPreconditioner
    dim
    function TrivialPreconditioner()
        new(nothing)
    end
end

"""
Full LU factorization as preconditioner (intended for smaller subsystems)
"""
mutable struct LUPreconditioner <: JutulPreconditioner
    factor
    function LUPreconditioner()
        new(nothing)
    end
end

function update!(lup::LUPreconditioner, A, b, context)
    if isnothing(lup.factor)
        lup.factor = lu(A)
    else
        lu!(lup.factor, A)
    end
end

export operator_nrows
function operator_nrows(lup::LUPreconditioner)
    f = get_factorization(lup)
    return size(f.L, 1)
end

# LU factor as precond for wells?

"""
Trivial / identity preconditioner with size for use in subsystems.
"""
# Trivial precond
function update!(tp::TrivialPreconditioner, lsys, arg...)
    A = jacobian(lsys)
    b = residual(lsys)
    tp.dim = size(A).*length(b[1])
end

function linear_operator(id::TrivialPreconditioner, ::Symbol)
    return opEye(id.dim...)
end

"""
Multi-model preconditioners
"""
mutable struct GroupWisePreconditioner <: JutulPreconditioner
    preconditioners::AbstractVector
    function GroupWisePreconditioner(preconditioners)
        new(preconditioners)
    end
end

function update!(prec::GroupWisePreconditioner, lsys::MultiLinearizedSystem, arg...)
    s = lsys.subsystems
    n = size(s, 1)
    @assert n == length(prec.preconditioners)
    for i in 1:n
        update!(prec.preconditioners[i], s[i, i], arg...)
    end
end

function linear_operator(precond::GroupWisePreconditioner, side::Symbol = :left)
    d = Vector{LinearOperator}(map((x) -> linear_operator(x, side), precond.preconditioners))
    D = BlockDiagonalOperator(d...)
    return D
end
