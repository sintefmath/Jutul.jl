export ILUZeroPreconditioner, LUPreconditioner, GroupWisePreconditioner, TrivialPreconditioner, DampedJacobiPreconditioner, AMGPreconditioner, JutulPreconditioner, apply!

abstract type JutulPreconditioner end

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
        new(method, kwarg, cycle, nothing, nothing, nothing, nothing)
    end
end

matrix_for_amg(A) = A
matrix_for_amg(A::StaticSparsityMatrixCSR) = A.At

function update!(amg::AMGPreconditioner, A, b, context)
    kw = amg.method_kwarg
    A_amg = matrix_for_amg(A)
    @debug string("Setting up preconditioner ", amg.method)
    t_amg = @elapsed hierarchy = amg.method(A_amg; kw...)
    amg = specialize_hierarchy!(amg, hierarchy, A, context)
    amg.dim = size(A)
    @debug "Set up AMG in $t_amg seconds."
    amg.factor = aspreconditioner(amg.hierarchy, amg.cycle)
end

function specialize_hierarchy!(amg, hierarchy, A, context)
    amg.hierarchy = hierarchy
    return amg
end

function create_transposed(x)
    i, j, v = findnz(x)
    n, m = size(x)
    return sparse(j, i, v, m, n)
end

function specialize_hierarchy!(amg, h, A::StaticSparsityMatrixCSR, context)
    A_f = A
    nt = A.nthreads
    mb = A.minbatch
    to_csr(M) = StaticSparsityMatrixCSR(M, nthreads = nt, minbatch = mb)
    levels = h.levels
    new_levels = []
    n = length(levels)
    for i = 1:n
        l = levels[i]
        P0, R0 = l.P, l.R
        # Convert P to a proper CSC matrix to get parallel
        # spmv for the wrapper type, trading some memory for
        # performance.
        if isa(P0, Adjoint)
            P0 = create_transposed(R0)
        elseif isa(R0, Adjoint)
            R0 = create_transposed(P0)
        end
        A = to_csr(l.A)
        R = to_csr(P0)
        P = to_csr(R0)
        lvl = AlgebraicMultigrid.Level(A, P, R)
        push!(new_levels, lvl)
    end

    typed_levels = Vector{typeof(new_levels[1])}(undef, n)
    for i = 1:n
        typed_levels[i] = new_levels[i]
    end
    levels = typed_levels
    S = amg.smoothers = generate_smoothers_csr(A_f, levels, nt, context)
    smoother = (A, x, b) -> apply_smoother!(x, A, b, S)

    A_c = to_csr(h.final_A)
    factor = factorize_coarse(A_c)
    coarse_solver = (x, b) -> ldiv!(x, factor, b)

    amg.hierarchy = AlgebraicMultigrid.MultiLevel(typed_levels, A_c, coarse_solver, smoother, smoother, h.workspace)
    return amg
end

function generate_smoothers_csr(A_fine, levels, nt, context)
    sizes = Vector{Int64}()
    smoothers = []
    n = length(levels)
    for i = 1:n
        A = levels[i].A
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
    typed_smoothers = Vector{typeof(smoothers[1])}(undef, n)
    for i = 1:n
        typed_smoothers[i] = smoothers[i]
    end
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
    @timeit "coarse update" amg.hierarchy = update_hierarchy!(amg.hierarchy, A)
    @timeit "smoother update" amg.smoothers = update_smoothers!(amg.smoothers, A, amg.hierarchy)
end

operator_nrows(amg::AMGPreconditioner) = amg.dim[1]

factorize_coarse(A) = lu(A)
factorize_coarse(A::StaticSparsityMatrixCSR) = lu(A.At)'

function update_hierarchy!(h, A)
    levels = h.levels
    n = length(levels)
    for i = 1:n
        l = levels[i]
        P, R = l.P, l.R
        levels[i] = AlgebraicMultigrid.Level(A, P, R)
        if i == n
            A_c = h.final_A
        else
            A_c = levels[i+1].A
        end
        A = update_coarse_system!(A_c, R, A, P)
    end
    factor = factorize_coarse(A)
    coarse_solver = (x, b) -> ldiv!(x, factor, b)
    # print_system(A)
    # error()

    return AlgebraicMultigrid.MultiLevel(levels, A, coarse_solver, h.presmoother, h.postsmoother, h.workspace)
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

function update_coarse_system!(A_c, R, A, P)
    return R*A*P
end

function update_coarse_system!(A_c, R, A::StaticSparsityMatrixCSR, P)
    if false
        A_c = coarse_product!(A_c, A, R)
    else
        At = A.At
        Pt = R.At
        Rt = Pt'
        A_c = Rt*At*Pt
        A_c = StaticSparsityMatrixCSR(A_c, nthreads = A.nthreads, minbatch = A.minbatch)
    end
    return A_c
end

function update_smoothers!(smoothers::Nothing, A, h)

end

function update_smoothers!(S::NamedTuple, A::StaticSparsityMatrixCSR, h)
    n = length(h.levels)
    for i = 1:(n-1)
        ilu0_csr!(S.smoothers[i].factor, A)
        A = h.levels[i+1].A
    end
end

"""
Damped Jacobi preconditioner on CPU
"""
mutable struct DampedJacobiPreconditioner <: JutulPreconditioner
    factor
    dim
    w
    function DampedJacobiPreconditioner(; w = 1)
        new(nothing, nothing, w)
    end
end


function update!(jac::DampedJacobiPreconditioner, A, b, context)
    ω = jac.w
    if isnothing(jac.factor)
        D = Diagonal(A)
        for i in 1:size(D, 1)
            D.diag[i] = ω*inv(D.diag[i])
        end
        jac.factor = D
        d = length(b[1])
        jac.dim = d .* size(A)
    else
        D = jac.factor
        for i in 1:size(D, 1)
            D.diag[i] = ω*inv(A[i, i])
        end
    end
end

function apply!(x, jac::DampedJacobiPreconditioner, y, arg...)
    # Very hacky.
    D = jac.factor

    s = D.diag[1]
    N = size(s, 1)
    T = eltype(s)
    Vt = SVector{N, T}
    as_svec = (x) -> reinterpret(Vt, x)

    # Solve by reinterpreting vectors to block (=SVector) vectors
    tmp = D*as_svec(y)
    xv = as_svec(x)
    xv .= tmp
    # f! = ilu_f(type)
    # f!(as_svec(x), ilu, as_svec(y))
end

function operator_nrows(ilu::DampedJacobiPreconditioner)
    return ilu.dim[1]
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
        nt = nthreads(context)
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

function update!(ilu::ILUZeroPreconditioner, A::CuSparseMatrix, b::CuArray, context)
    if isnothing(ilu.factor)
        set_dim!(ilu, A, b)
    end
    ilu.factor = ilu02(A, 'O')
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

function ilu_apply!(x::AbstractArray{F}, f::CuSparseMatrix{F}, y::AbstractArray{F}, type::Symbol = :both) where {F<:Real}
    x .= y
    ix = 'O'
    sv2!('N', 'L', 'N', 1.0, f, x, ix)
    sv2!('N', 'U', 'U', 1.0, f, x, ix)
end

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
