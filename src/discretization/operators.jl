function local_divergence(F, local_half_face_map)
    (; cells, faces, signs) = local_half_face_map
    v = -F(cells[1], faces[1], signs[1])
    for i = 2:length(cells)
        v -= F(cells[i], faces[i], signs[i])
    end
    return v
end

function divergence(F, local_half_face_map)
    (; faces, signs) = local_half_face_map

    @inbounds v = -signs[1]*F(faces[1])
    @inbounds for i in 2:length(faces)
        f = faces[i]
        s = signs[i]
        v -= s*F(f)
    end
    return v
end

function divergence!(x, F, local_half_face_map)
    (; cells, faces, signs) = local_half_face_map
    for i in eachindex(cells)
        x -= F(cells[i], faces[i], signs[i])
    end
    return x
end
