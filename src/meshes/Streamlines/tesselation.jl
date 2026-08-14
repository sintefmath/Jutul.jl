"""
Cell tesselation and velocity reconstruction
"""

"""
    tesselate_cell(mesh, cell, geometry)

Subdivide a cell into sub-cells using centroid tesselation.
This creates the geometric structure without velocity information.
"""
function tesselate_cell(mesh::UnstructuredMesh{D}, cell::Int, 
                       geometry::TwoPointFiniteVolumeGeometry) where D
    T = Jutul.float_type(mesh)
    subcells = SubCell{D, T}[]
    
    # Get cell centroid
    cell_centroid = SVector{D, T}(geometry.cell_centroids[:, cell])
    
    for face_idx in mesh.faces.cells_to_faces[cell]
        face = mesh.faces.faces_to_nodes[face_idx]
        append!(subcells, tesselate_face(mesh, cell, face_idx, false, face, cell_centroid))
    end
    for face_idx in mesh.boundary_faces.cells_to_faces[cell]
        face = mesh.boundary_faces.faces_to_nodes[face_idx]
        append!(subcells, tesselate_face(mesh, cell, face_idx, true, face, cell_centroid))
    end
    
    return subcells
end

function tesselate_face(mesh::UnstructuredMesh{D}, cell::Int, face_idx::Int,
                        is_boundary::Bool, face, cell_centroid::SVector{D, T}) where {D, T}
    subcells = SubCell{D, T}[]
    zero_velocity = zero(SVector{D, T})

    if D == 3
        for triangle in triangulate_face_nodes(mesh, face)
            vertices = [
                mesh.node_points[triangle[1]],
                mesh.node_points[triangle[2]],
                mesh.node_points[triangle[3]],
                cell_centroid
            ]
            subcell_centroid, measure = compute_tet_centroid_and_volume(vertices)
            bbox_min, bbox_max = compute_subcell_bbox(vertices)
            push!(subcells, SubCell(cell, face_idx, is_boundary, vertices, subcell_centroid, zero_velocity, measure, bbox_min, bbox_max))
        end
    elseif D == 2
        @assert length(face) == 2 "2D faces should have 2 nodes"
        vertices = [
            mesh.node_points[face[1]],
            mesh.node_points[face[2]],
            cell_centroid
        ]
        subcell_centroid, measure = compute_tri_centroid_and_area(vertices)
        bbox_min, bbox_max = compute_subcell_bbox(vertices)
        push!(subcells, SubCell(cell, face_idx, is_boundary, vertices, subcell_centroid, zero_velocity, measure, bbox_min, bbox_max))
    else
        error("Only 2D and 3D meshes are supported")
    end

    return subcells
end

"""
    update_subcell_velocities!(subcells, mesh, fluxes, geometry)

Update velocities in existing subcells based on face fluxes.
This is the second phase of setup, allowing velocity updates without rebuilding geometry.
"""
function update_subcell_velocities!(subcells::Vector{SubCell{D, T}}, mesh::UnstructuredMesh{D},
                                   fluxes::AbstractVector,
                                   geometry::TwoPointFiniteVolumeGeometry;
                                   boundary_fluxes = nothing) where {D, T}
    nc = Jutul.number_of_cells(mesh)
    nbf = Jutul.number_of_boundary_faces(mesh)
    if isnothing(boundary_fluxes)
        boundary_fluxes = zeros(T, nbf)
    else
        length(boundary_fluxes) == nbf || throw(ArgumentError("Expected $nbf boundary face fluxes, got $(length(boundary_fluxes))."))
    end
    length(fluxes) == Jutul.number_of_faces(mesh) || throw(ArgumentError("Expected $(Jutul.number_of_faces(mesh)) face fluxes, got $(length(fluxes))."))

    cell_velocities = Vector{SVector{D, T}}(undef, nc)
    for cell in 1:nc
        cell_velocities[cell] = reconstruct_cell_velocity(mesh, geometry, fluxes, boundary_fluxes, cell, T)
    end

    for subcell in subcells
        subcell.velocity = cell_velocities[subcell.parent_cell]
    end
    
    return subcells
end

"""
    triangulate_face_nodes(mesh, face)

Triangulate a face (polygon) into triangles for tesselation.
Uses simple fan triangulation from the first node.
"""
function triangulate_face_nodes(mesh::UnstructuredMesh, face::AbstractVector{Int})
    if length(face) == 3
        # Already a triangle
        return [face]
    else
        # Fan triangulation from first vertex
        triangles = Vector{Vector{Int}}()
        for i in 2:(length(face)-1)
            push!(triangles, [face[1], face[i], face[i+1]])
        end
        return triangles
    end
end

"""
    compute_tet_centroid_and_volume(vertices)

Compute centroid and volume of a tetrahedron.
"""
function compute_tet_centroid_and_volume(vertices::Vector{SVector{3, T}}) where T
    @assert length(vertices) == 4
    # Centroid is average of vertices
    centroid = (vertices[1] + vertices[2] + vertices[3] + vertices[4]) / 4
    
    # Volume using determinant formula
    v0 = vertices[1]
    v1 = vertices[2] - v0
    v2 = vertices[3] - v0
    v3 = vertices[4] - v0
    
    volume = abs(dot(v1, cross(v2, v3))) / 6
    
    return (centroid, volume)
end

"""
    compute_tri_centroid_and_area(vertices)

Compute centroid and area of a triangle.
"""
function compute_tri_centroid_and_area(vertices::Vector{SVector{2, T}}) where T
    @assert length(vertices) == 3
    # Centroid is average of vertices
    centroid = (vertices[1] + vertices[2] + vertices[3]) / 3
    
    # Area using cross product
    v1 = vertices[2] - vertices[1]
    v2 = vertices[3] - vertices[1]
    
    area = abs(v1[1] * v2[2] - v1[2] * v2[1]) / 2
    
    return (centroid, area)
end

"""
    compute_subcell_bbox(vertices)

Compute axis-aligned bounding box for a sub-cell.
"""
function compute_subcell_bbox(vertices::Vector{SVector{D, T}}) where {D, T}
    bbox_min = MVector{D, T}(vertices[1])
    bbox_max = MVector{D, T}(vertices[1])
    for v in vertices[2:end]
        for d in 1:D
            bbox_min[d] = min(bbox_min[d], v[d])
            bbox_max[d] = max(bbox_max[d], v[d])
        end
    end
    return (SVector{D, T}(bbox_min), SVector{D, T}(bbox_max))
end

"""
    reconstruct_cell_velocity(mesh, geometry, fluxes, boundary_fluxes, cell, T)

Reconstruct a constant velocity in a cell from face fluxes using a least-squares fit.
"""
function reconstruct_cell_velocity(mesh::UnstructuredMesh{D},
                                   geometry::TwoPointFiniteVolumeGeometry,
                                   fluxes::AbstractVector,
                                   boundary_fluxes::AbstractVector,
                                   cell::Int,
                                   ::Type{T}) where {D, T}
    A = zeros(T, D, D)
    b = zeros(T, D)

    for face_idx in mesh.faces.cells_to_faces[cell]
        neighbors = mesh.faces.neighbors[face_idx]
        sign = neighbors[1] == cell ? one(T) : -one(T)
        normal = sign * SVector{D, T}(geometry.normals[:, face_idx])
        area = T(geometry.areas[face_idx])
        row = area * normal
        flux = sign * T(fluxes[face_idx])
        for i in 1:D, j in 1:D
            A[i, j] += row[i] * row[j]
        end
        for i in 1:D
            b[i] += flux * row[i]
        end
    end

    for face_idx in mesh.boundary_faces.cells_to_faces[cell]
        normal = SVector{D, T}(geometry.boundary_normals[:, face_idx])
        area = T(geometry.boundary_areas[face_idx])
        row = area * normal
        flux = T(boundary_fluxes[face_idx])
        for i in 1:D, j in 1:D
            A[i, j] += row[i] * row[j]
        end
        for i in 1:D
            b[i] += flux * row[i]
        end
    end

    scale = max(sum(abs, A), one(T))
    λ = sqrt(eps(T)) * scale
    for i in 1:D
        A[i, i] += λ
    end
    return SVector{D, T}(A \ b)
end
