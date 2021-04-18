export ConservationLaw, setup_conservationlaw, allocate_vector_ad

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

function allocate_vector_ad(n::R, nder = 0; T=Float64) where {R<:Integer}
    # allocate a n length vector with space for derivatives
    if nder == 0
        return Vector{T}(undef, n)
    else
        d = ForwardDiff.Dual{T}(zeros(nder+1)...)
        return Vector{typeof(d)}(undef, n)
    end
    
end