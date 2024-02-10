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

struct LinearInterpolant{V, T}
    X::V
    F::T
end

function LinearInterpolant(X::T, F::T; static = false) where T<:AbstractVector
    @assert length(X) == length(F)
    @assert issorted(X) "Interpolation inputs must be sorted: X = $X"
    if static
        n = length(X)
        T_el = eltype(X)
        X = SVector{n, T_el}(X)
        F = SVector{n, T_el}(F)
    end
    return LinearInterpolant{T, T}(X, F)
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
    if cap_endpoints && (cap_start || cap_end)
        ϵ = 100*sum(abs, xs)
        xs = copy(xs)
        ys = copy(ys)
        # Add perturbed points, repeat start and end value
        if cap_start
            pushfirst!(xs, xs[1] - ϵ)
            pushfirst!(ys, ys[1])
        end
        if cap_end
            push!(xs, xs[end] + ϵ)
            push!(ys, ys[end])
        end
    end
    nx = length(xs)
    ny = length(ys)
    nx > 1 || throw(ArgumentError("xs values must have more than one entry."))
    nx == ny || throw(ArgumentError("Number of xs ($nx) and ys(x) ($ny) must match"))
    return method(xs, ys)
end
struct BilinearInterpolant{V, T}
    X::V
    Y::V
    F::T
    function BilinearInterpolant(xs::T, ys::T, fs::M) where {T, M}
        issorted(xs) || throw(ArgumentError("xs must be sorted."))
        issorted(ys) || throw(ArgumentError("ys must be sorted."))
        nx = length(xs)
        ny = length(ys)
        size(fs) == (nx, ny) || throw(ArgumentError("f(x, y) must match lengths of xs (as rows) and xy (as columns) = ($nx,$ny)"))
        return new{T, M}(xs, ys, fs)
    end
end

function bilinear_interp(X, Y, F, x, y)
    function interp_local(X_0, F_0, X_1, F_1, X)
        ∂F∂X = (F_1 - F_0)/(X_1 - X_0)
        ΔX = X - X_0
        return F_0 + ∂F∂X*ΔX
    end

    x_pos = Jutul.first_lower(X, value(x))
    y_pos = Jutul.first_lower(Y, value(y))
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
    w_lower = (y - y_2)/Δy
    w_upper = (y_1 - y)/Δy
    F = w_lower*F_lower + w_upper*F_upper

    return F
end

function interpolate(I::BilinearInterpolant, x, y)
    return bilinear_interp(I.X, I.Y, I.F, x, y)
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
        cap_y = (cap_endpoints, cap_endpoints)
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
        ϵ_x = 100*sum(abs, xs)
        ϵ_y = 100*sum(abs, ys)

        F_t = eltype(fs)
        nx, ny = size(fs)
        fs_new = zeros(F_t, nx + cap_xlo + cap_xhi, ny + cap_ylo + cap_yhi)
        xoffset = cap_xlo
        yoffset = cap_ylo
        fs_new[(1+xoffset):(nx+xoffset), (1+yoffset):(ny+yoffset)] .= fs
        if cap_xlo
            fs_new[1, :] .= fs_new[2, :]
            pushfirst!(xs, xs[1] - ϵ_x)
        end
        if cap_xhi
            fs_new[end, :] .= fs_new[end-1, :]
            push!(xs, xs[end] + ϵ_x)
        end
        if cap_ylo
            fs_new[:, 1] .= fs_new[:, 2]
            pushfirst!(ys, ys[1] - ϵ_y)
        end
        if cap_yhi
            fs_new[:, end] .= fs_new[:, end-1]
            push!(ys, ys[end] + ϵ_y)
        end
        fs = fs_new
    end

    return method(xs, ys, fs)
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
