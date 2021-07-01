export ILUZeroPreconditioner
using ILUZero

abstract type TervPreconditioner end

function update!(preconditioner, A, b)

end

function get_factorization(precond)
    return precond.factor
end

function linear_operator(precond::TervPreconditioner, side::Symbol = :left)
    n = matrix_dim(precond)
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
        if precond.left
            if precond.right
                f! = (r, x, α, β) -> local_mul!(r, x, α, β, :left)
            else
                f! = (r, x, α, β) -> local_mul!(r, x, α, β, :both)
            end
            op = LinearOperator(Float64, n, n, false, false, f!)
        else
            op = I
        end
    elseif side == :right
        if precond.right
            f! = (r, x, α, β) -> local_mul!(r, x, α, β, :right)
            op = LinearOperator(Float64, n, n, false, false, f!)
        else
            op = I
        end
    else
        error("Side must be :left or :right, was $side")
    end

    return op
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
    factor = ilu.factor
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

function ilu_apply!(x::Vector{F}, f::ILU0Precon{F}, y::Vector{F}, type::Symbol = :both) where {F<:Real}
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

function matrix_dim(ilu::ILUZeroPreconditioner)
    return ilu.dim[1]
end
