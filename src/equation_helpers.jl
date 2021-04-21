export ConservationLaw, setup_conservationlaw
export allocate_vector_ad, get_ad_unit_scalar, update_values!
export allocate_residual, allocate_jacobian

struct ConservationLaw
    accumulation::AbstractArray
    half_face_flux::AbstractArray
end

function ConservationLaw(G::TervGrid, nder::Integer = 0, hasAcc = true; T=Float64)
    # Create conservation law for a given grid with a number of partials
    nc = number_of_cells(G)
    nf = number_of_half_faces(G)
    ConservationLaw(nc, nf, nder, hasAcc, T = T)
end

function ConservationLaw(nc::Integer, nf::Integer, nder::Integer = 0, hasAcc = true; T=Float64)
    if hasAcc
        acc = allocate_vector_ad(nc, nder, T=T)
    else
        acc = nothing
    end
    flux = allocate_vector_ad(2*nf, nder, T=T)
    ConservationLaw(acc, flux)
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
