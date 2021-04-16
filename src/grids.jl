export TervGrid, MRSTGrid, MinimalTPFAGrid, TPFAHalfFaceData

abstract type TervGrid end


# Helpers follow
"Minimal struct for TPFA connections (transmissibility + dz + cell pair)"
struct TPFAHalfFaceData{R<:Real,I<:Integer}
    T::R
    dz::R
    self::I
    other::I
end

# TPFA grid
struct MinimalTPFAGrid{R<:AbstractFloat, I<:Integer} <: TervGrid
    conn_data::AbstractArray{TPFAHalfFaceData{R,I}}
    pv::AbstractArray{R}
end


# Member functions, TPFA grid
function number_of_cells(G::MinimalTPFAGrid)
    return length(G.pv)
end

function number_of_faces(G::MinimalTPFAGrid)
    return number_of_half_faces(G)/2
end

function number_of_half_faces(G::MinimalTPFAGrid)
    return length(G.TPFAHalfFaceData)
end

function get_pore_volume(G::MinimalTPFAGrid)
    return G.pore_volume
end
