
function update_preconditioner!(preconditioner::Nothing, arg...)
    # Do nothing.
end
function update_preconditioner!(preconditioner, lsys, model, storage, recorder, executor)
    J = jacobian(lsys)
    r = residual(lsys)
    ctx = linear_system_context(model, lsys)
    update_preconditioner!(preconditioner, J, r, ctx, executor)
end

function partial_update_preconditioner!(p, A, b, context, executor)
    update_preconditioner!(p, A, b, context, executor)
end

function get_factorization(precond)
    return precond.factor
end

is_left_preconditioner(::JutulPreconditioner) = true
is_right_preconditioner(::JutulPreconditioner) = false

function linear_operator(precond::JutulPreconditioner, float_t = Float64)
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
