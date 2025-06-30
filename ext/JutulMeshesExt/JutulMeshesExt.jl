module JutulMeshesExt
    using Jutul, Meshes
    function meshes_fv_geometry_3d(grid::Mesh)
        nc = nelements(grid)
        nf = nfacets(grid)
        topo = topology(grid)
        bnd = Boundary{3,2}(topo)
        # Global neighborship
        N = zeros(Int64, 2, nf)
        # Normals and centroids
        normals = Vector{Meshes.Vec3}(undef, nf)
        cell_centroids = centroid.(grid)
        face_centroids = Vector{Point3}(undef, nf)
        # Measures
        areas = zeros(nf)
        volumes = volume.(grid)
        # Mask found/not found for each face to get signed normals
        found = BitArray(undef, nf)
        found .= false
        for cell_ix in 1:nc
            hexa = grid[cell_ix]
            # Next, deal with faces
            boundary_faces = bnd(cell_ix)
            mesh = boundary(hexa)
            nhf = length(mesh)
            @assert nhf == length(boundary_faces)
            for j in 1:nhf
                face_ix = boundary_faces[j]
                if found[face_ix]
                    # Already processed, only need to add neighborship
                    lr = 2
                else
                    lr = 1
                    face_mesh = mesh[j]
                    face_centroids[face_ix] = centroid(face_mesh)
                    areas[face_ix] = area(face_mesh)
                    # Simplexify to get normals
                    trimesh = simplexify(face_mesh)
                    normals[face_ix] = first(normal.(trimesh))
                    found[face_ix] = true
                end
                N[lr, face_ix] = cell_ix
            end
        end
        # Convert Point3 arrays to Jutul bare array format for geometry
        d = 3
        float_normals = zeros(d, nf)
        float_face_centroids = zeros(d, nf)
        for f in 1:nf
            for j in 1:d
                float_normals[j, f] = normals[f][j]
                float_face_centroids[j, f] = face_centroids[f].coords[j]
            end
        end
        float_cell_centroids = zeros(d, nc)
        for c in 1:nc
            for j in 1:d
                float_cell_centroids[j, c] = cell_centroids[c].coords[j]
            end
        end
        # Jutul currently expects the interior faces
        active = vec(all(x -> x > 0, N, dims=1))
        return (
                N=N[:, active], 
                areas=areas[active],
                volumes=volumes,
                normals=float_normals[:, active],
                cell_centroids = float_cell_centroids,
                face_centroids = float_face_centroids[:, active]
                )
    end


    function Jutul.tpfv_geometry(g::T) where T<:Meshes.Mesh{3, <:Any}
        N, A, V, Nv, Cc, Fc = meshes_fv_geometry_3d(g)
        geo = TwoPointFiniteVolumeGeometry(N, A, V, Nv, Cc, Fc)
        return geo
    end

    function Jutul.add_default_domain_data!(Ω::DataDomain, m::Meshes.Mesh; geometry = missing)
        # TODO: Fix this code duplication
        if ismissing(geometry)
            geometry = tpfv_geometry(m)
        end
        fv = geometry
        geom_pairs = (
            Pair(Faces(), [:neighbors, :areas, :normals, :face_centroids]),
            Pair(Cells(), [:cell_centroids, :volumes]),
            Pair(HalfFaces(), [:half_face_cells, :half_face_faces]),
            Pair(BoundaryFaces(), [:boundary_areas, :boundary_centroids, :boundary_normals, :boundary_neighbors])
        )
        for (entity, names) in geom_pairs
            if hasentity(Ω, entity)
                for name in names
                    Ω[name, entity] = getproperty(fv, name)
                end
            end
        end
    end
end
