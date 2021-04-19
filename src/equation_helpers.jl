export ConservationLaw, setup_conservationlaw, allocate_vector_ad, get_ad_unit_scalar

struct ConservationLaw
    accumulation
    half_face_flux
end

function setup_conservationlaw(G::TervGrid, nder = 0, hasAcc = true; T=Float64)
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

function allocate_vector_ad(n::R, nder = 0; T = Float64, diag_pos = nothing) where {R<:Integer}
    # allocate a n length vector with space for derivatives
    if nder == 0
        return Vector{T}(undef, n)
    else
        # d = ForwardDiff.Dual{T}(zeros(nder+1)...)
        d = get_ad_unit_scalar(T(0.0), nder, diag_pos)
        return Vector{typeof(d)}(undef, n)
    end
end

function get_ad_unit_scalar(v::T, nder, diag_pos = nothing, diag_val = 1) where {T<:Real}
    # Get a scalar, with a given number of zero derivatives. A single entry can be specified to be non-zero
    if nder > 0
        if isnothing(diag_pos)
            v = ForwardDiff.Dual{T}([v, zeros(nder+1)...]...)
        else
            v = ForwardDiff.Dual{T}([v, zeros(diag_pos-1)..., diag_val, zeros(nder-diag_pos)...]...)
        end
    end
    return v
end