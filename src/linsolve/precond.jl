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
    function ILUZeroPreconditioner()
        new(nothing)
    end
end

function update!(ilu::ILUZeroPreconditioner, A, b)
    if isnothing(ilu.factor)
        ilu.factor = ilu0(A)
    else
        ilu0!(ilu.factor, A)
    end
end
function apply!(x, ilu::ILUZeroPreconditioner, y)
    # Why must this be qualified?
    ILUZero.ldiv!(x, ilu.factor, y)
end

function get_n(ilu::ILUZeroPreconditioner)
    return ilu.factor.n
end
