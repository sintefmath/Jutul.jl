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

        if ismissing(cell_size[d])
            hd = x -> (xb[2] - xb[1])/20
        elseif isa(cell_size[d], Number)
            hd = x -> cell_size[d]
        else
            hd = cell_size[d]
            @assert isa(hd, Function) "cell_size in dimension $d must be a number or a function"
        end
        constraints_d = isnothing(constraints[d]) ? [] : constraints[d]
        for x in xb
            if !any(c -> c[1] == x && c[3] == :boundary, constraints_d)
                push!(constraints_d, (x, hd(x), :boundary, Inf))
            end
        end

        c0 = copy(constraints_d)
        constraints_d = []
        for c in c0
            add_constraint!(constraints_d, c)
        end

        constraints_d, removed_d = process_constraints(constraints_d)
        
        xd = sample_coordinates(constraints_d, hd)

        push!(x, xd)
    end
  
    x = Tuple(x)
    sizes = map(x->diff(x), x)
    dims = Tuple([length(s) for s in sizes])
    msh = CartesianMesh(dims, sizes, origin = minimum.(x))

    return msh

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

function sample_coordinates(constraints, h)

    x = Float64[]
    n = length(constraints)
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
        dx = xr - xl
        
        hl = x[end] - x[end-1]
        hc = h((xr + xl)/2)
        hr = xc[2] - xc[1]

        # frac = (1 < k < n-1) ? 1/3 : 2/3
        frac = 1/3
        dx_t = frac*dx
        do_interpolation = true#interpolation[i] != :nothing
        force_interpolation = true
        interpolation = fill(:both, n)
        if do_interpolation && hc > dx_t
            msg = "Transition in layer $(k) not feasible (hz = $hc > dz = $dx_t)"
            if force_interpolation
                hc = dx_t/2
                @warn msg * ". Forcing interpolation by reducing hz to $hc"
            else
                do_interpolation = false
                @warn msg * ". No interpolation will be done."
            end
        end
        if do_interpolation
            if interpolation[k] ∈ [:top, :both]
                xl_t = xl + dx_t
                hl = min(hl, dx_t/2)
            end
            if interpolation[k] ∈ [:bottom, :both]
                xr_t = xr - dx_t
                hr = min(hr, dx_t/2)
            end
        end

        xl = interpolate(xl, xl_t, hl, hc)
        xm = interpolate(xl_t, xr_t, hc, hc)
        xr = interpolate(xr_t, xr, hc, hr)
        xk = unique(vcat(xl[1:end-1], xm[1:end-1], xr[1:end], xc...))

        push!(x, xk...)
        unique!(x)
    end

    return x

end

function interpolate(z_a::Float64, z_b::Float64, dz_a::Float64, dz_b::Float64)

    L = z_b - z_a
    if isapprox(L, 0.0)
        z = [z_a, z_b]

    elseif isapprox(dz_a, dz_b)
        n = max(Int(round(L/dz_a))+1,2)
        z = collect(range(z_a, z_b, length=n))

    else
        α = (L-dz_a)/(L-dz_b)
        K = Int(round(log(dz_b/dz_a)/log(α)))
        α = (dz_b/dz_a)^(1/K)
        dz = dz_a*α.^(0:K)
        rem = sum(dz) - L
        dz .-= rem.*dz./sum(dz)
        z = z_a .+ cumsum(vcat(0, dz))

    end

    @assert isapprox(z[1], z_a) && isapprox(z[end], z_b)

    return z

end