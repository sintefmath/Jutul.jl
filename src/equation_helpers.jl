export ConservationLaw, setup_conservationlaw
export allocate_array_ad, get_ad_unit_scalar, update_values!
export allocate_residual, allocate_jacobian
export value

struct ConservationLaw <: TervEquation
    accumulation::AbstractArray
    half_face_flux::AbstractArray
    accumulation_jac_pos::AbstractArray   # Usually diagonal entries
    half_face_flux_jac_pos::AbstractArray # Equal length to half face flux
end

function ConservationLaw(G::TervGrid, lsys, nder::Integer = 0; jacobian_row_offset = 0, context = DefaultContext())
    F = float_type(context)
    I = index_type(context)
    # Create conservation law for a given grid with a number of partials
    nc = number_of_cells(G)
    nf = number_of_half_faces(G)

    accpos = zeros(I, nder, nc)
    fluxpos = zeros(I, nder, nf)
    # Note: jacobian_row_offset needs to be added somewhere for multiphase
    jac = lsys.jac
    # Note: We copy this back to host if it is on GPU to avoid rewriting these functions for CuArrays
    conn_data = Array(G.conn_data)
    accumulation_sparse_pos!(accpos, jac)
    half_face_flux_sparse_pos!(fluxpos, jac, nc, conn_data)

    # Once positions are figured out (in a CPU context since Jacobian is not yet transferred)
    # we copy over the data to target device (or do nothing if we are on CPU)
    accpos = transfer(context, accpos)
    fluxpos = transfer(context, fluxpos)

    ConservationLaw(nc, nf, accpos, fluxpos, nder, context = context)
end

function ConservationLaw(nc::Integer, nhf::Integer, 
                         accpos::AbstractArray, fluxpos::AbstractArray, 
                         npartials::Integer = 0; context = DefaultContext(), units = 1)
    acc = allocate_array_ad(nc, units, context = context, npartials = npartials)
    flux = allocate_array_ad(nhf, units, context = context, npartials = npartials)
    ConservationLaw(acc, flux, accpos, fluxpos)
end

function allocate_array_ad(n::R...; context::TervContext = DefaultContext(), diag_pos = nothing, npartials = 1) where {R<:Integer}
    # allocate a n length zero vector with space for derivatives
    T = float_type(context)
    if npartials == 0
        A = allocate_array(context, T(0), n...)
    else
        if isa(diag_pos, AbstractVector)
            @assert n[1] == length(diag_pos)
            d = map(x -> get_ad_unit_scalar(T(0.0), npartials, x), diag_pos)
            A = allocate_array(context, d, 1, n[2:end]...)
        else
            d = get_ad_unit_scalar(T(0.0), npartials, diag_pos)
            A = allocate_array(context, d, n...)
        end
    end
    return A
end

function allocate_array_ad(v::AbstractVector; context = DefaultContext(), diag_pos = nothing, npartials = 1)
    # create a copy of a vector as AD
    v_AD = allocate_array_ad(length(v), context = context, diag_pos = diag_pos, npartials = npartials)
    update_values!(v_AD, v)
end

function allocate_array_ad(v::AbstractMatrix; context = DefaultContext(), diag_pos = nothing, npartials = 1)
    # create a copy of a vector as AD
    v_AD = allocate_array_ad(size(v)..., context = context, diag_pos = diag_pos, npartials = npartials)
    update_values!(v_AD, v)
end



function get_ad_unit_scalar(v::T, npartials, diag_pos = nothing) where {T<:Real}
    # Get a scalar, with a given number of zero derivatives. A single entry can be specified to be non-zero
    if npartials > 0
        v = ForwardDiff.Dual{T}(v, ntuple(x -> T.(x == diag_pos), npartials))
    end
    return v
end

function update_values!(v::AbstractArray, next::AbstractArray)
    @. v = v - value(v) + next
end


function accumulation_sparse_pos!(accpos, jac)
    nder = size(accpos, 1)
    nc = size(accpos, 2)
    for i in 1:nc
        for derno = 1:nder
            # Diagonal positions
            global_pos = (derno-1)*nc + i
            pos = jac.colptr[global_pos]:jac.colptr[global_pos+1]-1
            accpos[derno, i] = pos[jac.rowval[pos] .== global_pos][1]
        end
    end
end

function half_face_flux_sparse_pos!(fluxpos, jac, nc, conn_data)
    nder = size(fluxpos, 1)
    nf = size(fluxpos, 2)

    for i in 1:nf
        # Off diagonal positions
        cd = conn_data[i]
        self = cd.self
        other = cd.other
        for derno = 1:nder
            global_pos = (derno-1)*nc + self
            pos = jac.colptr[global_pos]:jac.colptr[global_pos+1]-1
            fluxpos[derno, i] = pos[jac.rowval[pos] .== other + (derno-1)*nc][1]
        end
    end
end

@inline function value(x)
    return ForwardDiff.value(x)
end

function value(d::Dict)
    v = copy(d)
    for key in keys(v)
        v[key] = value.(v[key])
    end
    return v
end