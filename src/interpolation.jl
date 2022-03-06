export get_1d_interpolator

first_lower(tab, x) = clamp(searchsortedfirst(tab, x) - 1, 1, length(tab) - 1)

function interval_weight(t, x, ix)
    @inbounds first = t[ix]
    @inbounds last = t[ix+1]
    Δx = last - first
    w = (x - first)/Δx
    return w
end

function linear_interp(X, F, x)
    ix = first_lower(X, x)
    return linear_interp_internal(F, X, ix, x)
end

function linear_interp_internal(F, X, ix, x)
    @inbounds X_0 = X[ix]
    @inbounds F_0 = F[ix]
    @inbounds ∂F∂X = (F[ix+1] - F_0)/(X[ix+1] - X_0)
    Δx = x - X_0
    return F_0 + ∂F∂X*Δx
end

struct LinearInterpolant{V}
    X::V
    F::V
    function LinearInterpolant(X::T, F::T) where T<:AbstractVector
        @assert length(X) == length(F)
        @assert issorted(X) "Interpolation inputs must be sorted"
        new{T}(X, F)
    end
end

interpolate(I::LinearInterpolant, x) = linear_interp(I.X, I.F, x)
(I::LinearInterpolant)(x) = interpolate(I, x)

function get_1d_interpolator(xs, ys; method = LinearInterpolant, cap_endpoints = true)
    if cap_endpoints
        ϵ = 100*sum(abs.(xs))
        # Add perturbed points
        xs = vcat(xs[1] - ϵ, xs)
        xs = vcat(xs, xs[end] + ϵ)
        # Repeat start and end
        ys = vcat(ys[1], ys)
        ys = vcat(ys, ys[end])
    end
    return method(xs, ys)
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
