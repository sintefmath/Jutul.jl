export get_1d_interpolator

@inline function first_lower(tab, x, ::Missing)
    return first_lower(tab, x)
end

@inline function first_lower(tab, x)
    return clamp(searchsortedfirst(tab, x) - 1, 1, length(tab) - 1)
end

@inline function first_lower(tab, x, lookup)
    x0, dx, n = lookup
    if x <= x0 + dx
        pos = 1
    else
        m = floor(Int, (x-x0)/dx)+1
        pos = min(m, n-1)
    end
    return pos::Int
end

@inline function interval_weight(t, x, ix)
    @inbounds first = t[ix]
    @inbounds last = t[ix+1]
    Δx = last - first
    w = (x - first)/Δx
    return w
end

@inline function linear_interp(X, F, x, lookup = missing)
    ix = first_lower(X, value(x), lookup)
    return linear_interp_internal(F, X, ix, x)
end

@inline function linear_interp_internal(F, X, ix, x)
    @inbounds X_0 = X[ix]
    @inbounds F_0 = F[ix]
    @inbounds ∂F∂X = (F[ix+1] - F_0)/(X[ix+1] - X_0)
    Δx = x - X_0
    return F_0 + ∂F∂X*Δx
end

"""
    interpolation_constant_lookup(X, constant_dx = missing)

Generate a lookup table for linear interpolation when dx is evenly spaced.

Note: Setting `constant_dx=true` can lead to incorrect interpolations if the
data is not evenly spaced.
"""
function interpolation_constant_lookup(X, constant_dx = missing)
    Δx = X[2] - X[1]
    if ismissing(constant_dx)
        constant_dx = true
        for i in 2:length(X)
            constant_dx = constant_dx && Δx ≈ X[i] - X[i-1]
        end
    end
    constant_dx::Bool

    if constant_dx
        lookup = (x0 = X[1], dx = Δx, n = length(X))
    else
        lookup = missing
    end
    return lookup
end

struct LinearInterpolant{V, T, L}
    X::V
    F::T
    lookup::L
end

function LinearInterpolant(X::V, F::T; static = false, constant_dx = missing) where {T<:AbstractVector, V<:AbstractVector}
    length(X) == length(F) || throw(ArgumentError("X and F values must have equal length."))
    if !issorted(X)
        ix = sortperm(X)
        X = X[ix]
        F = F[ix]
    end
    lookup = interpolation_constant_lookup(X, constant_dx)
    if static
        n = length(X)
        T_el = eltype(X)
        X = SVector{n, T_el}(X)
        F = SVector{n, T_el}(F)
    end
    L = typeof(lookup)
    return LinearInterpolant{V, T, L}(X, F, lookup)
end

interpolate(I::LinearInterpolant, x) = linear_interp(I.X, I.F, x, I.lookup)
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

Additional keyword arguments are passed onto the interpolator constructor.
"""
function get_1d_interpolator(xs, ys;
        method = LinearInterpolant,
        cap_endpoints = true,
        cap_end = cap_endpoints,
        cap_start = cap_endpoints,
        kwarg...
    )
    if cap_endpoints && (cap_start || cap_end)
        xs = copy(xs)
        ys = copy(ys)
        # Add perturbed points, repeat start and end value
        if cap_start
            ϵ = xs[2] - xs[1]
            pushfirst!(xs, xs[1] - ϵ)
            pushfirst!(ys, ys[1])
        end
        if cap_end
            ϵ = xs[end] - xs[end-1]
            push!(xs, xs[end] + ϵ)
            push!(ys, ys[end])
        end
    end
    nx = length(xs)
    ny = length(ys)
    nx > 1 || throw(ArgumentError("xs values must have more than one entry."))
    nx == ny || throw(ArgumentError("Number of xs ($nx) and ys(x) ($ny) must match"))
    return method(xs, ys; kwarg...)
end
struct BilinearInterpolant{V, T, LX, LY}
    X::V
    Y::V
    F::T
    lookup_x::LX
    lookup_y::LY
    function BilinearInterpolant(xs::T, ys::T, fs::M;
            constant_dx = missing,
            constant_dy = missing
        ) where {T, M}
        issorted(xs) || throw(ArgumentError("xs must be sorted."))
        issorted(ys) || throw(ArgumentError("ys must be sorted."))
        lookup_x = interpolation_constant_lookup(xs, constant_dx)
        lookup_y = interpolation_constant_lookup(ys, constant_dy)
        nx = length(xs)
        ny = length(ys)
        size(fs) == (nx, ny) || throw(ArgumentError("f(x, y) must match lengths of xs (as rows) and xy (as columns) = ($nx,$ny)"))
        return new{T, M, typeof(lookup_x), typeof(lookup_y)}(xs, ys, fs, lookup_x, lookup_y)
    end
end

function bilinear_interp(X, Y, F, x, y, lookup_x = missing, lookup_y = missing)
    function interp_local(X_0, F_0, X_1, F_1, X)
        ∂F∂X = (F_1 - F_0)/(X_1 - X_0)
        ΔX = X - X_0
        return F_0 + ∂F∂X*ΔX
    end

    x_pos = first_lower(X, value(x), lookup_x)
    y_pos = first_lower(Y, value(y), lookup_y)
    @inbounds begin
        x_1 = X[x_pos]
        x_2 = X[x_pos+1]
        Δx = x_2 - x_1

        y_1 = Y[y_pos]
        y_2 = Y[y_pos+1]
        Δy = y_2 - y_1

        F_11 = F[x_pos, y_pos]
        F_12 = F[x_pos, y_pos+1]
        F_21 = F[x_pos+1, y_pos]
        F_22 = F[x_pos+1, y_pos+1]
    end
    F_upper = interp_local(x_1, F_12, x_2, F_22, x)
    F_lower = interp_local(x_1, F_11, x_2, F_21, x)
    w_lower = (y_2 - y)/Δy
    w_upper = (y - y_1)/Δy
    F = w_lower*F_lower + w_upper*F_upper
    return F
end

function interpolate(I::BilinearInterpolant, x, y)
    return bilinear_interp(I.X, I.Y, I.F, x, y, I.lookup_x, I.lookup_y)
end

function (I::BilinearInterpolant)(x, y)
    return interpolate(I, x, y)
end

export get_2d_interpolator
"""
    get_2d_interpolator(xs, ys, fs; method = BilinearInterpolant, cap_endpoints = true)

For `xs` of length `nx` and `ys` of length `ny` generate a 2D interpolation for
values given as a `nx` by `ny` matrix. By default `cap_endpoints=true`, and
constant extrapolation is used. Fine-grined control over extrapolation can be
achieved by setting the keywords arguments `cap_x = (cap_low_x, cap_high_x)` and
analogously for `cap_y`.
"""
function get_2d_interpolator(xs, ys, fs;
        method = BilinearInterpolant,
        cap_endpoints = true,
        cap_x = (cap_endpoints, cap_endpoints),
        cap_y = (cap_endpoints, cap_endpoints),
        kwarg...
    )
    cap_xlo, cap_xhi = cap_x
    cap_ylo, cap_yhi = cap_y

    cap_any_x = cap_xlo || cap_xhi
    cap_any_y = cap_ylo || cap_yhi
    if cap_any_x || cap_any_y
        if cap_any_x
            xs = copy(xs)
        end
        if cap_any_y
            ys = copy(ys)
        end

        F_t = eltype(fs)
        nx, ny = size(fs)
        fs_new = zeros(F_t, nx + cap_xlo + cap_xhi, ny + cap_ylo + cap_yhi)
        xoffset = cap_xlo
        yoffset = cap_ylo
        fs_new[(1+xoffset):(nx+xoffset), (1+yoffset):(ny+yoffset)] .= fs
        if cap_xlo
            fs_new[1, :] .= fs_new[2, :]
            ϵ = xs[2] - xs[1]
            pushfirst!(xs, xs[1] - ϵ)
        end
        if cap_xhi
            fs_new[end, :] .= fs_new[end-1, :]
            ϵ = xs[end] - xs[end-1]
            push!(xs, xs[end] + ϵ)
        end
        if cap_ylo
            fs_new[:, 1] .= fs_new[:, 2]
            ϵ = ys[2] - ys[1]
            pushfirst!(ys, ys[1] - ϵ)
        end
        if cap_yhi
            fs_new[:, end] .= fs_new[:, end-1]
            ϵ = ys[end] - ys[end-1]
            push!(ys, ys[end] + ϵ)
        end
        fs = fs_new
    end

    return method(xs, ys, fs; kwarg...)
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

function update_secondary_variable!(V, var::UnaryTabulatedVariable, model, state, ix = entity_eachindex(V))
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
