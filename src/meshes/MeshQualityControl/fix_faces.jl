function fix_nonpositive_areas!(msh, bad_faces; print::Bool = true, boundary::Bool = false)
    if length(bad_faces) > 0
        # We always print this since the issue cannot be fixed
        if boundary
            ts = "boundary"
        else
            ts = "internal"
        end
        jutul_print("Mesh fixing", "Mesh has $ts faces with non-positive or non-finite area. Automatic fixing is not implemented for this issue.", color = :yellow)
    end
    return msh
end

function fix_normal_orientation!(msh::UnstructuredMesh, bad_faces; print::Bool = true, boundary::Bool = false)
    if boundary
        dest = msh.boundary_faces.faces_to_nodes
    else
        dest = msh.faces.faces_to_nodes
    end
    for f in bad_faces
        subview = dest[f]
        reverse!(subview)
    end
    return msh
end
