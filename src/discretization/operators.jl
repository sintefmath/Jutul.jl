function local_divergence(F, local_half_face_map)
    (; cells, faces, signs) = local_half_face_map
    v = sum((cell, face, sign) -> F(cell, face, sgn), zip(cells, faces, signs))
    return sum(v)
end
