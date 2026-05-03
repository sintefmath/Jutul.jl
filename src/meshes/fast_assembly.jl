struct FastAssemblyData{dim, num_type}
    node_points::Vector{SVector{dim, num_type}}
    face_centroids::Vector{SVector{dim, num_type}}
    cell_centroids::Vector{SVector{dim, num_type}}
    neighbors::Vector{Tuple{Int, Int}}
    normals::Vector{SVector{dim, num_type}}
    areas::Vector{num_type}
    volumes::Vector{num_type}
    cells_to_faces::IndirectionMap{Int}
    cells_to_nodes::IndirectionMap{Int}
    faces_to_nodes::IndirectionMap{Int}
    nodes_to_faces::IndirectionMap{Int}
    nodes_to_cells::IndirectionMap{Int}
    node_is_boundary::Vector{Bool}
    face_is_boundary::Vector{Bool}
    function FastAssemblyData(;
            node_points::Vector{SVector{dim, num_type}},
            face_centroids,
            cell_centroids,
            neighbors,
            normals,
            areas,
            volumes,
            cells_to_faces,
            cells_to_nodes,
            faces_to_nodes,
            nodes_to_faces,
            nodes_to_cells,
            node_is_boundary,
            face_is_boundary
        ) where {dim, num_type}
        nc = length(cell_centroids)
        nf = length(face_centroids)
        nn = length(node_points)

        # Cells
        length(cell_centroids) == nc || error("Length of cell_centroids must match number of cells")
        length(volumes) == nc || error("Length of volumes must match number of cells")
        length(cells_to_faces) == nc || error("Length of cells_to_faces must match number of cells")
        length(cells_to_nodes) == nc || error("Length of cells_to_nodes must match number of cells")
        # Nodes
        nn > 0 || error("node_points cannot be empty")
        length(node_is_boundary) == nn || error("Length of node_is_boundary must match number of nodes")
        length(nodes_to_faces) == nn || error("Length of nodes_to_faces must match number of nodes")
        length(nodes_to_cells) == nn || error("Length of nodes_to_cells must match number of nodes")
        # Faces
        length(neighbors) == nf || error("Length of neighbors must match number of faces")
        length(areas) == nf || error("Length of areas must match number of faces")
        length(face_is_boundary) == nf || error("Length of face_is_boundary must match number of faces")
        length(faces_to_nodes) == nf || error("Length of faces_to_nodes must match number of faces")
        length(face_centroids) == nf || error("Length of face_centroids must match number of faces")
        length(normals) == nf || error("Length of normals must match number of faces")
        new{dim, num_type}(
            node_points,
            face_centroids,
            cell_centroids,
            neighbors,
            normals,
            areas,
            volumes,
            cells_to_faces,
            cells_to_nodes,
            faces_to_nodes,
            nodes_to_faces,
            nodes_to_cells,
            node_is_boundary,
            face_is_boundary
        )
    end
end

"""
    FastAssemblyData(mesh::UnstructuredMesh)

Set up data structures for fast assembly of of discretizations. This includes
geometric information such as cell volumes, face areas, and normals, as well as
connectivity information such as which faces belong to which cells and which
nodes belong to which faces. The data is stored in a way that is optimized for
fast access with statically sized vectors for vectors and points.
"""
function FastAssemblyData(
        mesh::UnstructuredMesh;
        num_type = Jutul.float_type(mesh)
    )
    mesh_dim = dim(mesh)
    nf = number_of_faces(mesh)
    nbf = number_of_boundary_faces(mesh)
    nc = number_of_cells(mesh)
    nn = length(mesh.node_points)

    geo = tpfv_geometry(mesh)
    Vec_t = SVector{mesh_dim, num_type}
    function to_svec(x)
        if eltype(x) != num_type
            x = num_type.(x)
        end
        return collect(vec(reinterpret(Vec_t, x)))
    end

    cell_centroids = to_svec(geo.cell_centroids)
    volumes = geo.volumes
    areas = copy(geo.areas)
    # Faces
    face_centroids = to_svec(geo.face_centroids)
    normals = to_svec(geo.normals)
    is_boundary_face = [false for _ in 1:nf]

    for i in 1:nbf
        push!(is_boundary_face, true)
        push!(areas, geo.boundary_areas[i])
    end

    for bc in to_svec(geo.boundary_centroids)
        push!(face_centroids, bc)
    end
    for bn in to_svec(geo.boundary_normals)
        push!(normals, bn)
    end
    nf_total = nf + nbf
    @assert length(face_centroids) == nf_total
    @assert length(areas) == nf_total
    @assert length(is_boundary_face) == nf_total

    function setup_vector_of_vector(n)
        vov = Vector{Vector{Int}}()
        sizehint!(vov, n)
        for _ in 1:n
            push!(vov, Int[])
        end
        return vov
    end

    # Construct various mappings
    # cells -> faces and cells -> nodes
    c2f = setup_vector_of_vector(nc)
    c2n = setup_vector_of_vector(nc)
    n2c = setup_vector_of_vector(nn)
    for c in 1:nc
        for f in mesh.faces.cells_to_faces[c]
            for n in mesh.faces.faces_to_nodes[f]
                push!(c2n[c], n)
                push!(n2c[n], c)
            end
            push!(c2f[c], f)
        end
        for bf in mesh.boundary_faces.cells_to_faces[c]
            for n in mesh.boundary_faces.faces_to_nodes[bf]
                push!(c2n[c], n)
                push!(n2c[n], c)
            end
            push!(c2f[c], bf + nf)
        end
        unique!(c2n[c])
        # unique!(c2f[c])
    end
    # faces -> nodes
    f2n = setup_vector_of_vector(nf_total)
    n2f = setup_vector_of_vector(nn)
    for f in 1:nf
        for n in mesh.faces.faces_to_nodes[f]
            push!(f2n[f], n)
            push!(n2f[n], f)
        end
    end
    for bf in 1:nbf
        f = bf + nf
        for n in mesh.boundary_faces.faces_to_nodes[bf]
            push!(f2n[f], n)
            push!(n2f[n], f)
        end
    end
    for n in 1:nn
        unique!(n2f[n])
        unique!(n2c[n])
    end
    neighbors = Vector{Tuple{Int, Int}}()
    for f in 1:nf
        push!(neighbors, mesh.faces.neighbors[f])
    end
    for bf in 1:nbf
        push!(neighbors, (mesh.boundary_faces.neighbors[bf], 0))
    end

    node_is_boundary = Bool[]
    for n in 1:nn
        push!(node_is_boundary, any(is_boundary_face[f] for f in n2f[n]))
    end

    return FastAssemblyData(;
        node_points = mesh.node_points,
        face_centroids = face_centroids,
        cell_centroids = cell_centroids,
        normals = normals,
        neighbors = neighbors,
        areas = areas,
        volumes = volumes,
        cells_to_faces = IndirectionMap(c2f),
        cells_to_nodes = IndirectionMap(c2n),
        faces_to_nodes = IndirectionMap(f2n),
        nodes_to_faces = IndirectionMap(n2f),
        nodes_to_cells = IndirectionMap(n2c),
        node_is_boundary = node_is_boundary,
        face_is_boundary = is_boundary_face
    )
end

function Base.show(io::IO, t::MIME"text/plain", data::FastAssemblyData{dim, num_type}) where {dim, num_type}
    print(io, "FastAssemblyData with dimension $dim and numerical type $num_type")
end
