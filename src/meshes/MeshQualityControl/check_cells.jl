function check_volumes(m::JutulMesh, geo = tpfv_geometry(m); print = false)
    negative = findall(v -> v <= 0.0, geo.volumes)
    nonfinite = findall(v -> !isfinite(v), geo.volumes)
    bad = union(negative, nonfinite)
    unique!(bad)

    if print
        ncells = length(geo.volumes)
        nbad = length(bad)
        if nbad == 0
            jutul_message("Cell volumes", "All $ncells cells have positive volume.", color = :green)
        else
            jutul_message("Cell volumes", "$nbad out of $ncells cells have non-positive or non-finite volume.", color = :yellow)
        end
    end
    return bad
end
