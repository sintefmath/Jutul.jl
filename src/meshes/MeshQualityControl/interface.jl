function check_and_fix_mesh(msh; print = true)
    return check_and_fix_mesh!(deepcopy(msh); print = print)
end

function check_and_fix_mesh!(msh; print = true, recheck = true)
    ok, bad = check_mesh(msh; print = print, extra_out = true)
    if !ok
        if print
            jutul_message("Mesh fixing", "Mesh quality issues detected. Attempting automatic fixes for detected issues...", color = :yellow)
        end
        msh isa UnstructuredMesh || error("Automatic fixing of mesh is only implemented for unstructured meshes.")
        fix_nonpositive_cell_volumes!(msh, bad["bad_volumes"], print = print)
        fix_nonpositive_areas!(msh, bad["bad_internal_areas"], boundary = false, print = print)
        fix_nonpositive_areas!(msh, bad["bad_boundary_areas"], boundary = true, print = print)
        fix_normal_orientation!(msh, bad["bad_internal_faces"], boundary = false, print = print)
        fix_normal_orientation!(msh, bad["bad_boundary_faces"], boundary = true, print = print)
        if recheck
            jutul_message("Mesh fixing", "Re-checking mesh quality after attempted fixes...", color = :yellow)
            ok_after = check_mesh(msh; print = print)
            if ok_after
                jutul_message("Mesh fixing", "Mesh quality issues appear to have been resolved after attempted fixes.", color = :green)
            else
                jutul_message("Mesh fixing", "Mesh still has quality issues after attempted fixes. Manual intervention may be required.", color = :yellow)
            end
        end
    end
    return msh
end

function check_mesh(msh; print = true, extra_out = false)
    geo = tpfv_geometry(msh)
    if print
        jutul_message("Mesh quality control", "Checking mesh for issues...")
    end
    # Check volumes
    bad_volumes = check_volumes(msh, geo, print = print)
    # Check normals
    bad_int_faces = check_normals(msh, geo, boundary = false, print = print)
    bad_bnd_faces = check_normals(msh, geo, boundary = true, print = print)
    # Check areas
    bad_int_areas = check_areas(msh, geo, print = print, boundary = false)
    bad_bnd_areas = check_areas(msh, geo, print = print, boundary = true)

    bad = Dict(
        "bad_volumes" => bad_volumes,
        "bad_internal_faces" => bad_int_faces,
        "bad_boundary_faces" => bad_bnd_faces,
        "bad_internal_areas" => bad_int_areas,
        "bad_boundary_areas" => bad_bnd_areas
    )
    ok = sum(k -> length(bad[k]), keys(bad)) == 0
    if print
        if ok
            jutul_message("Mesh quality control", "No mesh quality issues detected.")
        else
            jutul_message("Mesh quality control", "Mesh quality issues detected. See details above.")
        end
    end
    if extra_out
        out = (ok, bad)
    else
        out = ok
    end
    return out
end