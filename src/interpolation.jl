export get_1d_interpolator

first_lower(tab, x) = clamp(searchsortedfirst(tab, x) - 1, 1, length(tab) - 1)

function first_lower_linear(tab, x)
    for i in eachindex(tab)
        X = @inbounds tab[i]
        if X >= x
            return i-1
        end
    end
    return length(tab)
end

@inline function interval_weight(t, x, ix)
    @inbounds first = t[ix]
    @inbounds last = t[ix+1]
    Δx = last - first
    w = (x - first)/Δx
    return w
end

@inline function linear_interp(X, F, x)
    ix = first_lower(X, value(x))
    return linear_interp_internal(F, X, ix, x)
end

@inline function linear_interp_internal(F, X, ix, x)
    @inbounds X_0 = X[ix]
    @inbounds F_0 = F[ix]
    @inbounds ∂F∂X = (F[ix+1] - F_0)/(X[ix+1] - X_0)
    Δx = x - X_0
    return F_0 + ∂F∂X*Δx
end

struct LinearInterpolant{V}
    X::V
    F::V
    function LinearInterpolant(X::T, F::T; static = false) where T<:AbstractVector
        @assert length(X) == length(F)
        @assert issorted(X) "Interpolation inputs must be sorted: X = $X"
        if static
            n = length(X)
            T_el = eltype(X)
            X = SVector{n, T_el}(X)
            F = SVector{n, T_el}(F)
        end
        new{T}(X, F)
    end
end

interpolate(I::LinearInterpolant, x) = linear_interp(I.X, I.F, x)
(I::LinearInterpolant)(x) = interpolate(I, x)

"""
    get_1d_interpolator(xs, ys; <keyword arguments>)

Get a 1D interpolator `F(x) ≈ y` for a table `xs, ys` that by default does constant extrapolation

# Arguments

- `xs`: sorted list of parameter points.
- `ys`: list of function values with equal length to `xs`
- `method=LinearInterpolant`: constructor for the interpolation. Defaults to `LinearInterpolant` which does simple linear interpolation.
- `cap_endpoints = true`: Add values so that the endpoints are capped (constant extrapolation). Otherwise, the extrapolation will match the method.
- `cap_start = cap_endpoints`: Fine-grained version of cap_endpoints for the start of the interval only (extrapolation for `x < xs[1]`)
- `cap_end = cap_endpoints`:Fine-grained version of cap_endpoints for the end of the interval only (extrapolation for `x > xs[end]`)
"""
function get_1d_interpolator(xs, ys; method = LinearInterpolant, cap_endpoints = true, cap_end = cap_endpoints, cap_start = cap_endpoints)
    if cap_endpoints
        ϵ = 100*sum(abs, xs)
        # Add perturbed points
        # Repeat start and end
        if cap_start
            xs = vcat(xs[1] - ϵ, xs)
            ys = vcat(ys[1], ys)
        end
        if cap_end
            xs = vcat(xs, xs[end] + ϵ)
            ys = vcat(ys, ys[end])
        end
    end
    return method(xs, ys)
end


struct UnaryTabulatedVariable <: VectorVariables
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
 
 function update_secondary_variable!(V, var::UnaryTabulatedVariable, model, state)
    update_unary_tabulated!(V, var, model, state[var.x_symbol], entity_eachindex(V))
 end

function update_unary_tabulated!(F_v, tbl::UnaryTabulatedVariable, model, x_v::AbstractVector, ix)
    I = tbl.interpolators
    for i in ix
        F_v[ph, i] = I[k](x_v[i])
    end
end

function update_unary_tabulated!(F_v, tbl::UnaryTabulatedVariable, model, x_v::AbstractMatrix, ix)
    I = tbl.interpolators
    for i in ix
        F_v[k, i] = I[k](x_v[k, i])
    end
end
