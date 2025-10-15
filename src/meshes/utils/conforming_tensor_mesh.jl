using IntervalSets

function tensor_mesh(bounding_box::Vector{Vector{Float64}};
    cell_size = [missing, missing, missing],
    constraints = [nothing, nothing, nothing],
)

    dim = length(bounding_box)
    x = []
    for d in 1:dim
        xb = bounding_box[d]
        @assert length(xb) == 2 "Bounding box in dimension $d must have length 2"
        @assert xb[1] < xb[2] "Bounding box in dimension $d must be increasing"

        hfun = missing
        if ismissing(cell_size[d])
            hb = (xb[2] - xb[1])/20
        elseif isa(cell_size[d], Number)
            hb = cell_size[d]
        else
            hfun = cell_size[d]
        end

        constraints_d = isnothing(constraints[d]) ? [] : constraints[d]
        # Add boundary constraints if not already present
        for x in xb
            if !any(c -> c[1] == x && c[3] == :boundary, constraints_d)
                h = ismissing(hfun) ? hb : hfun(x)
                push!(constraints_d, (x, h, :boundary, Inf))
            end
        end

        # Set cell size function if not provided
        if ismissing(hfun)
            hfun = interpolate_cell_size(constraints_d, hb)
        end
        @assert hfun isa Function "cell_size in dimension $d must be missing, a scalar, or a function"

        c0 = copy(constraints_d)
        constraints_d = []
        for c in c0
            add_constraint!(constraints_d, c)
        end

        constraints_d, removed_d = process_constraints(constraints_d)
        xd = sample_coordinates(constraints_d, hfun)

        push!(x, xd)
    end

    x = Tuple(x)
    sizes = map(x->diff(x), x)
    dims = Tuple([length(s) for s in sizes])
    msh = CartesianMesh(dims, sizes, origin = minimum.(x))

    return msh

end

function interpolate_cell_size(constraints, background_size)

    x = [c[1] for c in constraints]
    h = [c[2] for c in constraints]
    hb = background_size

    n = 100
    xv, hv = Float64[], Float64[]
    for k in 1:(length(x)-1)
        
        xm = (x[k] + x[k+1])/2
        
        xl = range(x[k], stop = xm, length = n)
        hl = range(h[k], stop = hb, length = n)
        push!(xv, xl...)
        push!(hv, hl...)

        xr = range(xm, stop = x[k+1], length = n)
        hr = range(hb, stop = h[k+1], length = n)
        push!(xv, xr...)
        push!(hv, hr...)

    end

    h = x -> hv[last(findmin(abs.(xv .- x)))]

    return h

end

function add_constraint!(constraints, new)

    (x, h, type, priority) = new
    @assert type ∈ [:boundary, :face, :cell] "Constraint type must be :face or :cell"
    @assert priority >= 0 "Constraint priority must be a non-negative integer"
    α = type == :cell ? 2 : 1
    xA, xB = x - h/α, x + h/α
    x = (xA, x, xB)
    push!(constraints, (x, type, priority))
    return constraints

end

function process_constraints(constraints)

    sort!(constraints, by = x -> x[end])
    processed_constraints = Vector{Any}(undef, 0)
    removed_constraints = Vector{Any}(undef, 0)
    while !isempty(constraints)
        curr = popfirst!(constraints)
        cc, tc, pc = curr
        if isempty(processed_constraints)
            push!(processed_constraints, curr)
            continue
        end
        for (k, other) in enumerate(processed_constraints)
            if isnothing(other)
                continue
            end
            co, to, po = other
            ic = cc[1]..cc[end]
            io = co[1]..co[end]
            intersection = ic ∩ io
            if !isempty(intersection)
                if pc == po
                    cc, co = shrink_constraints(cc, co)
                    processed_constraints[k] = (co, to, po)
                else
                    @assert pc > po
                    keep = co[2] ∈ intersection ? false : true
                    if keep
                        co = shrink_constraint(co, cc)
                        processed_constraints[k] = (co, to, po)
                    else
                        processed_constraints[k] = nothing
                        push!(removed_constraints, other)
                    end
                end
            end
        end
        push!(processed_constraints, (cc, tc, pc))

    end

    filter!(!isnothing, processed_constraints)
    sort!(processed_constraints, by = x -> x[1][2])


    return processed_constraints, removed_constraints

end

function shrink_constraints(ca, cb)
    if ca[2] > cb[2]
        return reverse(shrink_constraints(cb, ca))
    end

    _, xa_mid, xa_right = ca
    xb_left, xb_mid, _ = cb

    x_mid = (xa_mid + xb_mid)/2
    ha = x_mid - xa_mid
    hb = xb_mid - x_mid

    if ha > xa_right - xa_mid
        cb = shrink_constraint(cb, ca)
    elseif hb > xb_mid - xb_left
        ca = shrink_constraint(ca, cb)
    else
        ca = (xa_mid - ha, xa_mid, x_mid)
        cb = (x_mid, xb_mid, xb_mid + hb)
    end

    return ca, cb
end

function shrink_constraint(c, cp)

    x_left, x_mid, x_right = c
    xp_left, xp_mid, xp_right = cp

    if x_mid < xp_mid
        @assert x_right >= xp_left
        c = (x_mid - (xp_left - x_mid), x_mid, xp_left)
    else
        @assert x_left <= xp_right
        c = (xp_right, x_mid, x_mid + (x_mid - xp_right))
    end

    return c
end

function sample_coordinates(constraints, hfun, interpolation = :default)

    x = Float64[]
    n = length(constraints)
    if interpolation isa Symbol
        interpolation = fill(interpolation, n)
    end
    for (k, constraint) in enumerate(constraints)

        (c, type, priority) = constraint
        if type == :boundary
            xc = k == 1 ? c[2:end] : c[1:end-1]
        elseif type == :face
            xc = c
        elseif type == :cell
            xc = c[[1, end]]
        end

        if k == 1
            push!(x, xc...)
            continue
        end
        xl = x[end]
        xr = xc[1]
        xl_t, xr_t = xl, xr
        Δx = xr - xl

        @assert Δx >= 0 "Constraints must be ordered and non-overlapping"

        hl = x[end] - x[end-1]
        hc = hfun((xr + xl)/2)
        hr = xc[2] - xc[1]

        itp = interpolation[k]
        gt_tol = (x, y) -> isapprox(x, y; rtol=1e-2) ? false : x > y
        if itp == :default
            if gt_tol(hc, hl) && gt_tol(hc, hr)
               itp = :both
            elseif gt_tol(hc, hl)
               itp = :left
            elseif gt_tol(hc, hr)
               itp = :right
            else
               itp = :none
            end
        end

        Δxl_t, Δxr_t = 0.0, 0.0
        if itp == :left
            Δxl_t = Δx/2
        elseif itp == :right
            Δxr_t = Δx/2
        elseif itp == :both
            Δxl_t = Δx/3
            Δxr_t = Δx/3
        end

        force_interpolation = true
        if itp != :none && force_interpolation
            hl = Δxl_t > 0 ? min(hl, Δxl_t/2) : hl
            hc = min(hc, (Δx-Δxl_t-Δxr_t)/2)
            hr =  Δxr_t > 0 ? min(hr, Δxr_t/2) : hr
        end

        xl_t = xl + Δxl_t
        xr_t = xr - Δxr_t

        xl = interpolate(xl, xl_t, hl, hc)
        xm = interpolate(xl_t, xr_t, hc, hc)
        xr = interpolate(xr_t, xr, hc, hr)
        xk = unique(vcat(xl[1:end-1], xm[1:end-1], xr[1:end], xc...))

        push!(x, xk...)
        unique!(x)

    end

    return x

end

function interpolate(xa::Float64, xb::Float64, da::Float64, db::Float64)

    L = xb - xa
    if isapprox(L, 0.0)
        z = [xa, xb]

    elseif isapprox(da, db)
        n = max(Int(round(L/da))+1,2)
        z = collect(range(xa, xb, length=n))

    else
        α = (L-da)/(L-db)
        K = Int(round(log(db/da)/log(α)))
        α = (db/da)^(1/K)
        dz = da*α.^(0:K)
        rem = sum(dz) - L
        dz .-= rem.*dz./sum(dz)
        z = xa .+ cumsum(vcat(0, dz))

    end

    @assert isapprox(z[1], xa) && isapprox(z[end], xb)
    z[1] = xa
    z[end] = xb

    return z

end