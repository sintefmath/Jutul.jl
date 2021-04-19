export ConservationLaw, setup_conservationlaw
export allocate_vector_ad, get_ad_unit_scalar, update_values!
export allocate_residual, allocate_jacobian

struct ConservationLaw
    accumulation
    half_face_flux
end

function setup_conservationlaw(G::TervGrid, nder = 0, hasAcc = true; T=Float64)
    # Should be made into a normal constructor when I figure out the syntax
    nc = number_of_cells(G)
    nf = number_of_half_faces(G)
    if hasAcc
        acc = allocate_vector_ad(nc, nder, T=T)
    else
        acc = nothing
    end
    flux = allocate_vector_ad(2*nf, nder, T=T)
    ConservationLaw(acc, flux)
    # ConservationLaw{T}(acc, flux)
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
