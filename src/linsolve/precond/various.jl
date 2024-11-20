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

function update_preconditioner!(lup::LUPreconditioner, A, b, context, executor)
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

#function operator_nrows(prec::TrivialPreconditioner)
#    return prec.dim[1]
#end
# LU factor as precond for wells?

"""
Trivial / identity preconditioner with size for use in subsystems.
"""
function apply!(x,tp::TrivialPreconditioner,r, args...)
    x = copy(r)
end


# Trivial precond
function update_preconditioner!(tp::TrivialPreconditioner, lsys, model, storage, recorder, executor)
    A = jacobian(lsys)
    b = residual(lsys)
    tp.dim = size(A).*length(b[1])
end
export linear_operator
function linear_operator(id::TrivialPreconditioner, ::Symbol, args...)
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

function update_preconditioner!(prec::GroupWisePreconditioner, lsys::MultiLinearizedSystem, arg...)
    s = lsys.subsystems
    n = size(s, 1)
    @assert n == length(prec.preconditioners)
    for i in 1:n
        update_preconditioner!(prec.preconditioners[i], s[i, i], arg...)
    end
end

function linear_operator(precond::GroupWisePreconditioner, float_t = Float64)
    d = Vector{LinearOperator}(map((x) -> linear_operator(x, float_t), precond.preconditioners))
    D = BlockDiagonalOperator(d...)
    return D
end
