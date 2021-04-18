export ConservationLaw, setup_conservationlaw

struct ConservationLaw
    accumulation
    half_face_flux
end

function setup_conservationlaw(G::TervGrid, nder = 0, hasAcc = true; T=Float64)
    nc = number_of_cells(G)
    nf = number_of_half_faces(G)
    acc = zeros(T, nc)
    if nder == 0
        dt = T
    else
        d = ForwardDiff.Dual(zeros(nder+1)...)
        dt = typeof(d)
    end
    if hasAcc
        acc = Vector{dt}(undef, nc)
    else
        acc = nothing
    end
    flux = Vector{dt}(undef, 2*nf)
    ConservationLaw(acc, flux)
    # ConservationLaw{T}(acc, flux)
end
