
function check_normal(left_center, face_center, normal)
    to_face = face_center .- left_center
    return dot(to_face, normal) >= 0.0
end

function check_normals(m::JutulMesh, geo = tpfv_geometry(m); boundary::Bool = false, print = false)
    bad = Int[]
    cell_centroids = geo.cell_centroids
    if boundary
        neighbors = geo.boundary_neighbors
        normals = geo.boundary_normals
        face_centroids = geo.boundary_centroids
    else
        # Take left
        neighbors = geo.neighbors[1, :]
        normals = geo.normals
        face_centroids = geo.face_centroids
    end
    nf = size(normals, 2)
    for f in 1:nf
        l = neighbors[f]
        left_center = cell_centroids[:, l]
        ok = check_normal(left_center, face_centroids[:, f], normals[:, f])
        if !ok
            push!(bad, f)
        end
    end
    if print
        if boundary
            ts = "boundary"
        else
            ts = "internal"
        end
        nbad = length(bad)
        if nbad == 0
            jutul_message("Normal orientation", "All $nf $ts faces appear to be oriented correctly.", color = :green)
        else
            jutul_message("Normal orientation", "$nbad out of $nf $ts faces appear to be oriented the wrong way.", color = :yellow)
        end
    end
    return bad
end

function check_areas(m::JutulMesh, geo = tpfv_geometry(m); print = false, boundary::Bool = false)
    if boundary
        areas = geo.boundary_areas
        ts = "boundary"
    else
        areas = geo.areas
        ts = "internal"
    end
    negative = findall(a -> a <= 0.0, areas)
    nonfinite = findall(a -> !isfinite(a), areas)
    bad = union(negative, nonfinite)
    unique!(bad)
    if print
        nfaces = length(areas)
        nbad = length(bad)
        if nbad == 0
            jutul_message("Face areas", "All $nfaces $ts faces have positive area.", color = :green)
        else
            jutul_message("Face areas", "$nbad out of $nfaces $ts faces have non-positive or non-finite area.", color = :yellow)
        end
    end
    return bad
end