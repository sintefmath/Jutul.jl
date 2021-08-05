export ILUZeroPreconditioner, LUPreconditioner, GroupWisePreconditioner, TrivialPreconditioner, DampedJacobiPreconditioner, AMGPreconditioner
using ILUZero

abstract type TervPreconditioner end

function update!(preconditioner, lsys, model, storage)
    J = jacobian(lsys)
    r = residual(lsys)
    update!(preconditioner, J, r)
end

function get_factorization(precond)
    return precond.factor
end

is_left_preconditioner(::TervPreconditioner) = true
is_right_preconditioner(::TervPreconditioner) = false

function linear_operator(precond::TervPreconditioner, side::Symbol = :left)
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
            op = LinearOperator(Float64, n, n, false, false, f!)
        else
            op = opEye(n, n)
        end
    elseif side == :right
        if is_right_preconditioner(precond)
            f! = (r, x, α, β) -> local_mul!(r, x, α, β, :right)
            op = LinearOperator(Float64, n, n, false, false, f!)
        else
            op = opEye(n, n)
        end
    else
        error("Side must be :left or :right, was $side")
    end

    return op
end

function apply!(x, p::TervPreconditioner, y, arg...)
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
mutable struct AMGPreconditioner <: TervPreconditioner
    method
    factor
    hierarchy
    function AMGPreconditioner(method = ruge_stuben)
        new(method, nothing, nothing)
    end
end

function update!(amg::AMGPreconditioner, A, b)
    @debug string("Setting up preconditioner ", amg.method)
    t_amg = @elapsed amg.hierarchy = amg.method(A)
    @debug "Set up AMG in $t_amg seconds."
    amg.factor = aspreconditioner(amg.hierarchy)
end

"""
Damped Jacobi preconditioner on CPU
"""
mutable struct DampedJacobiPreconditioner <: TervPreconditioner
    factor
    dim
    w
    function DampedJacobiPreconditioner(; w = 1)
        new(nothing, nothing, w)
    end
end


function update!(jac::DampedJacobiPreconditioner, A, b)
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
mutable struct ILUZeroPreconditioner <: TervPreconditioner
    factor
    dim
    left::Bool
    right::Bool
    function ILUZeroPreconditioner(; left = true, right = true)
        @assert left || right "Left or right preconditioning must be enabled or it will have no effect."
        new(nothing, nothing, left, right)
    end
end

is_left_preconditioner(p::ILUZeroPreconditioner) = p.left
is_right_preconditioner(p::ILUZeroPreconditioner) = p.right

function update!(ilu::ILUZeroPreconditioner, A, b)
    if isnothing(ilu.factor)
        ilu.factor = ilu0(A, eltype(b))
        d = length(b[1])
        ilu.dim = d .* size(A)
    else
        ilu0!(ilu.factor, A)
    end
end

function apply!(x, ilu::ILUZeroPreconditioner, y, arg...)
    factor = get_factorization(ilu)
    ilu_apply!(x, factor, y, arg...)
end

function ilu_f(type::Symbol)
    # Why must this be qualified?
    if type == :left
        f = ILUZero.forward_substitution!
    elseif type == :right
        f = ILUZero.backward_substitution!
    else
        f = ILUZero.ldiv!
    end
end

function ilu_apply!(x::AbstractArray{F}, f::ILU0Precon{F}, y::AbstractArray{F}, type::Symbol = :both) where {F<:Real}
    f! = ilu_f(type)
    f!(x, f, y)
end

function ilu_apply!(x, ilu::ILU0Precon, y, type::Symbol = :both)
    # Very hacky.
    s = ilu.l_nzval[1]
    N = size(s, 1)
    T = eltype(s)
    Vt = SVector{N, T}
    as_svec = (x) -> reinterpret(Vt, x)

    # Solve by reinterpreting vectors to block (=SVector) vectors
    f! = ilu_f(type)
    f!(as_svec(x), ilu, as_svec(y))
end

function operator_nrows(ilu::ILUZeroPreconditioner)
    return ilu.dim[1]
end

mutable struct TrivialPreconditioner <: TervPreconditioner
    dim
    function TrivialPreconditioner()
        new(nothing)
    end
end
"""
Full LU factorization as preconditioner (intended for smaller subsystems)
"""
mutable struct LUPreconditioner <: TervPreconditioner
    factor
    function LUPreconditioner()
        new(nothing)
    end
end

function update!(lup::LUPreconditioner, A, b)
    if isnothing(lup.factor)
        lup.factor = lu(A)
    else
        lu!(lup.factor, A)
    end
end

function operator_nrows(lup::LUPreconditioner)
    f = get_factorization(lup)
    return size(f.L, 1)
end

# LU factor as precond for wells?

"""
Trivial / identity preconditioner with size for use in subsystems.
"""
# Trivial precond
function update!(tp::TrivialPreconditioner, A, b)
    tp.dim = size(A).*length(b[1])
end

function linear_operator(id::TrivialPreconditioner, ::Symbol)
    return opEye(id.dim...)
end

"""
Multi-model preconditioners
"""
mutable struct GroupWisePreconditioner <: TervPreconditioner
    preconditioners::AbstractVector
    function GroupWisePreconditioner(preconditioners)
        new(preconditioners)
    end
end

function update!(prec::GroupWisePreconditioner, lsys::MultiLinearizedSystem)
    s = lsys.subsystems
    n = size(s, 1)
    @assert n == length(prec.preconditioners)
    for i in 1:n
        update!(prec.preconditioners[i], s[i, i])
    end
end

function linear_operator(precond::GroupWisePreconditioner, side::Symbol = :left)
    d = Vector{LinearOperator}(map((x) -> linear_operator(x, side), precond.preconditioners))
    D = BlockDiagonalOperator(d...)
    return D
end
