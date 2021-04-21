export ConservationLaw, setup_conservationlaw
export allocate_vector_ad, get_ad_unit_scalar, update_values!
export allocate_residual, allocate_jacobian

struct ConservationLaw <: TervEquation
    accumulation::AbstractArray
    half_face_flux::AbstractArray
    accumulation_jac_pos::AbstractArray   # Usually diagonal entries
    half_face_flux_jac_pos::AbstractArray # Equal length to half face flux
end

function ConservationLaw(G::TervGrid, lsys, nder::Integer = 0, hasAcc = true; T=Float64, jacobian_row_offset = 0)
    # Create conservation law for a given grid with a number of partials
    nc = number_of_cells(G)
    nf = number_of_half_faces(G)

    accpos = zeros(Int64, nder, nc)
    fluxpos = zeros(Int64, nder, nf)
    # Note: jacobian_row_offset needs to be added somewhere for multiphase
    jac = lsys.jac
    for i in 1:nc
        for derno = 1:nder
            # Diagonal positions
            global_pos = (derno-1)*nc + i
            pos = jac.colptr[global_pos]:jac.colptr[global_pos+1]-1
            accpos[derno, i] = pos[jac.rowval[pos] .== global_pos][1]
        end
    end
    for i in 1:nf
        # Off diagonal positions
        cd = G.conn_data[i]
        self = cd.self
        other = cd.other
        for derno = 1:nder
            global_pos = (derno-1)*nc + self
            pos = jac.colptr[global_pos]:jac.colptr[global_pos+1]-1
            fluxpos[derno, i] = pos[jac.rowval[pos] .== other + (derno-1)*nc][1]
        end
    end
    ConservationLaw(nc, nf, accpos, fluxpos, nder, hasAcc, T = T)
end

function ConservationLaw(nc::Integer, nf::Integer, 
                         accpos::AbstractArray, fluxpos::AbstractArray, 
                         nder::Integer = 0, hasAcc = true; T=Float64)
    if hasAcc
        acc = allocate_vector_ad(nc, nder, T=T)
    else
        acc = nothing
    end
    flux = allocate_vector_ad(2*nf, nder, T=T)
    ConservationLaw(acc, flux, accpos, fluxpos)
end

function allocate_residual(G::TervGrid, Law::ConservationLaw)
    r = similar(Law.accumulation, number_of_cells(G))
    return r
end

function allocate_jacobian(G::TervGrid, Law::ConservationLaw)
    jac = get_incomp_matrix(G) # Just a hack for the time being
    return jac
end


function allocate_vector_ad(n::R, nder = 0; T = Float64, diag_pos = nothing) where {R<:Integer}
    # allocate a n length zero vector with space for derivatives
    if nder == 0
        return Vector{T}(0, n)
    else
        d = get_ad_unit_scalar(T(0.0), nder, diag_pos)
        return Vector{typeof(d)}(undef, n)
    end
end

function allocate_vector_ad(v::AbstractVector, nder = 0; T = Float64, diag_pos = nothing)
    # create a copy of a vector as AD
    v_AD = allocate_vector_ad(length(v), nder, T = T, diag_pos = diag_pos)
    update_values!(v_AD, v)
end

function get_ad_unit_scalar(v::T, nder, diag_pos = nothing) where {T<:Real}
    # Get a scalar, with a given number of zero derivatives. A single entry can be specified to be non-zero
    if nder > 0
        v = ForwardDiff.Dual{T}(v, ntuple(x -> T(x == diag_pos), nder))
    end
    return v
end

function update_values!(v::AbstractArray, next::AbstractArray)
    v .= v - value(v) + next
end
