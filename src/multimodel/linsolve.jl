
struct MultiLinearizedSystem{L} <: TervLinearSystem
    subsystems::Matrix{LinearizedType}
    r
    dx
    r_buffer
    dx_buffer
    matrix_layout::L
    function MultiLinearizedSystem(subsystems, context, layout; r = nothing, dx = nothing)
        n = 0
        for i = 1:size(subsystems, 1)
            ni, mi = size(subsystems[i, i].jac)
            @assert ni == mi
            n += ni
        end
        dx, dx_buf = get_jacobian_vector(n, context, layout, dx)
        r, r_buf = get_jacobian_vector(n, context, layout, r)
        new{typeof(layout)}(subsystems, r, dx, r_buf, dx_buf, layout)
    end
end

function getindex(ls::MultiLinearizedSystem, i, j = i)
    return ls.subsystems[i, j]
end

function vector_residual(sys::Matrix{LinearizedSystem})
    r = map(vector_residual, diag(sys))
    return vcat(r...)
end

function linear_operator(sys::MultiLinearizedSystem)

    
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
