export get_1d_interpolator

function get_1d_interpolator(xs, ys; method = DataInterpolations.LinearInterpolation, cap_endpoints = true)
    if cap_endpoints
        ϵ = 100*sum(abs.(xs))
        xs = vcat(xs[1] - ϵ, xs, xs[end] + ϵ);
        ys = vcat(ys[1], ys, ys[end]);
    end
    return method(ys, xs)
end


"""
"""
struct UnaryTabulatedVariable <: GroupedVariables
    x
    F
    interpolators
    x_symbol
    function UnaryTabulatedVariable(x::AbstractVector, F::AbstractMatrix, x_s::Symbol; kwarg...)
        nt, n = size(F)
        @assert nt > 0
        if eltype(x)<:AbstractVector
            # We got a set of different vectors that correspond to rows of kr
            @assert all(map(length, x) .== n)
            interpolators = map((ix) -> get_1d_interpolator(x[ix], F[ix, :]; kwarg...), 1:nt)
        else
            # We got a single vector that is used for all rows
            @assert length(x) == n
            interpolators = map((ix) -> get_1d_interpolator(x, F[ix, :]; kwarg...), 1:nt)
        end
        new(x, F, interpolators, x_s)
    end
end

function get_dependencies(var::UnaryTabulatedVariable, model)
    return [var.x_symbol]
 end
 
 function update_secondary_variable!(array_target, var::UnaryTabulatedVariable, model, parameters, state)
     update_as_secondary!(array_target, var, model, parameters, state[var.x_symbol])
 end

function update_as_secondary!(F_v, tbl::UnaryTabulatedVariable, model, param, x_v::AbstractVector)
    I = tbl.interpolators
    @tullio F_v[ph, i] = I[k](x_v[i])
end

function update_as_secondary!(F_v, tbl::UnaryTabulatedVariable, model, param, x_v::AbstractMatrix)
    I = tbl.interpolators
    @tullio F_v[k, i] = I[k](x_v[k, i])
end
