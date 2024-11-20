function update_preconditioner!(preconditioner::Nothing, arg...)
    # Do nothing.
end
function update_preconditioner!(preconditioner::JutulPreconditioner, lsys::JutulLinearSystem, context, model, storage, recorder, executor)
    J = jacobian(lsys)
    r = residual(lsys)
    update_preconditioner!(preconditioner, J, r, context, executor)
end

function partial_update_preconditioner!(p, A, b, context, executor)
    update_preconditioner!(p, A, b, context, executor)
end

function get_factorization(precond)
    return precond.factor
end

is_left_preconditioner(::JutulPreconditioner) = true
is_right_preconditioner(::JutulPreconditioner) = false

function linear_operator(precond::JutulPreconditioner)
    return linear_operator(precond, Float64, nothing, nothing, nothing, nothing, nothing)
end

function linear_operator(precond::JutulPreconditioner, float_t, sys, context, model, storage, recorder)
    n = operator_nrows(precond)
    function precond_apply!(res, x, α, β::T) where T
        if β == zero(T)
            apply!(res, precond, x)
            if α != one(T)
                lmul!(α, res)
            end
        else
            error("Preconditioner not implemented for β ≠ 0.")
        end
    end
    op = LinearOperator(float_t, n, n, false, false, precond_apply!)
    return op
end

#nead to be spesilized on type not all JutulPreconditioners has get_factor
function apply!(x, p::JutulPreconditioner, y)
    factor = get_factorization(p)
    ldiv!(x, factor, y)
end
