function local_divergence(F, local_half_face_map)
    (; cells, faces, signs) = local_half_face_map
    v = F(cells[1], faces[1], signs[1])
    for i = 2:length(cells)
        v += F(cells[i], faces[i], signs[i])
    end
    return v
end
