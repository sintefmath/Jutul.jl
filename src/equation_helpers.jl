export ConservationLaw, setup_conservationlaw
export allocate_vector_ad, get_ad_unit_scalar, update_values!
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
    accumulation_sparse_pos!(accpos, jac)
    half_face_flux_sparse_pos!(fluxpos, jac, nc, G.conn_data)

    # Once positions are figured out (in a CPU context since Jacobian is not yet transferred)
    # we copy over the data to target device (or do nothing if we are on CPU)
    accpos = transfer(context, accpos)
    fluxpos = transfer(context, fluxpos)

    ConservationLaw(nc, nf, accpos, fluxpos, nder, context = context)
end

function ConservationLaw(nc::Integer, nhf::Integer, 
                         accpos::AbstractArray, fluxpos::AbstractArray, 
                         nder::Integer = 0; context = DefaultContext())
    acc = allocate_vector_ad(nc, nder, context = context)
    flux = allocate_vector_ad(nhf, nder, context = context)
    ConservationLaw(acc, flux, accpos, fluxpos)
end

#function ConservationLaw(context::TervContext, nc, nhf, accpos, fluxpos, nder)
#    F = float_type(context)
#    fluxpos = transfer(context, fluxpos)
#    accpos = transfer(context, accpos)
#    acc = adapt(CuArray, allocate_vector_ad(nc, nder, T = F))
#    flux = adapt(CuArray, allocate_vector_ad(nhf, nder, T = F))
#
#    return ConservationLaw(acc, flux, accpos, fluxpos)
# end

function allocate_residual(G::TervGrid, Law::ConservationLaw)
    r = similar(Law.accumulation, number_of_cells(G))
    return r
end

function allocate_jacobian(G::TervGrid, Law::ConservationLaw)
    jac = get_incomp_matrix(G) # Just a hack for the time being
    return jac
end


function allocate_vector_ad(n::R, nder = 0; context::TervContext = DefaultContext(), diag_pos = nothing) where {R<:Integer}
    # allocate a n length zero vector with space for derivatives
    T = float_type(context)
    if nder == 0
        return zeros(T, n)
    else
        d = get_ad_unit_scalar(T(0.0), nder, diag_pos)
        return allocate_vector(context, d, n)
    end
end

function allocate_vector_ad(v::AbstractVector, nder = 0; context = DefaultContext(), diag_pos = nothing)
    # create a copy of a vector as AD
    v_AD = allocate_vector_ad(length(v), nder, context = context, diag_pos = diag_pos)
    update_values!(v_AD, v)
end

function get_ad_unit_scalar(v::T, nder, diag_pos = nothing) where {T<:Real}
    # Get a scalar, with a given number of zero derivatives. A single entry can be specified to be non-zero
    if nder > 0
        v = ForwardDiff.Dual{T}(v, ntuple(x -> T.(x == diag_pos), nder))
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