
function vector_residual(sys::Matrix{LinearizedSystem})
    r = map(vector_residual, diag(sys))
    return vcat(r...)
end

function linear_operator(sys::Matrix{LinearizedSystem})

    
    function block_mul!(res, x, α, β::T) where T
        res_v = as_svec(res)
        x_v = as_svec(x)
        if β == zero(T)
            mul!(res_v, jac, x_v)
            if α != one(T)
                lmul!(α, res_v)
            end
        else
            # TODO: optimize me like the three argument version.
            res_v .= α.*jac*x_v + β.*res_v
        end
    end


    apply! = get_mul!(sys)
    n = length(sys.r_buffer)
    op = LinearOperator(Float64, n, n, false, false, apply!)
    return op
end
