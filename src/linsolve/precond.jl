export ILUZeroPreconditioner
using ILUZero

abstract type TervPreconditioner end

function update!(preconditioner, A, b)

end

function get_factorization(precond)
    return precond.factor
end

function linear_operator(precond::TervPreconditioner)
    # f! = (x, b) -> ldiv!(x, precond, b)
    function f!(res, x, α, β::T) where T
        if β == zero(T)
            apply!(res, precond, x)
            if α != one(T)
                lmul!(α, res)
            end
        else
            error("Not implemented yet.")
        end
    end
    n = get_n(precond)
    return LinearOperator(Float64, n, n, false, false, f!)
end

"""
ILU(0) preconditioner on CPU
"""
mutable struct ILUZeroPreconditioner <: TervPreconditioner
    factor
    dim
    function ILUZeroPreconditioner()
        new(nothing, nothing)
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

function apply!(x, ilu::ILUZeroPreconditioner, y)
    factor = ilu.factor
    ilu_apply!(x, factor, y)
end

function ilu_apply!(x::Vector{F}, f::ILU0Precon{F}, y::Vector{F}) where {F<:Real}
    # Why must this be qualified?
    ILUZero.ldiv!(x, f, y)
end

function ilu_apply!(x, ilu::ILU0Precon, y)
    s = ilu.l_nzval[1]
    N = size(s, 1)
    T = eltype(s)
    Vt = SVector{N, T}
    as_svec = (x) -> reinterpret(Vt, x)
    x_b = as_svec(x)
    y_b = as_svec(y)
    ILUZero.ldiv!(x_b, ilu, y_b)
end

function get_n(ilu::ILUZeroPreconditioner)
    return ilu.dim[1]
end
