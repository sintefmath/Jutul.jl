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
    
    # Get faces of this cell
    cell_faces = mesh.faces.cells_to_faces[cell]
    
    # For each face, create sub-cells connecting the face to the cell centroid
    for face_idx in cell_faces
        face = mesh.faces.faces_to_nodes[face_idx]
        
        if D == 3
            # 3D: Create tetrahedra from face triangles
            # First, triangulate the face from its nodes
            face_triangles = triangulate_face_nodes(mesh, face)
            
            for triangle in face_triangles
                # Create a tetrahedron from triangle + cell centroid
                vertices = [
                    mesh.node_points[triangle[1]],
                    mesh.node_points[triangle[2]],
                    mesh.node_points[triangle[3]],
                    cell_centroid
                ]
                
                subcell_centroid, measure = compute_tet_centroid_and_volume(vertices)
                
                # Initialize with zero velocity (will be set later)
                zero_velocity = zero(SVector{D, T})
                
                push!(subcells, SubCell(cell, face_idx, vertices, subcell_centroid, zero_velocity, measure))
            end
        elseif D == 2
            # 2D: Create triangles from edge + cell centroid
            @assert length(face) == 2 "2D faces should have 2 nodes"
            vertices = [
                mesh.node_points[face[1]],
                mesh.node_points[face[2]],
                cell_centroid
            ]
            
            subcell_centroid, measure = compute_tri_centroid_and_area(vertices)
            
            # Initialize with zero velocity (will be set later)
            zero_velocity = zero(SVector{D, T})
            
            push!(subcells, SubCell(cell, face_idx, vertices, subcell_centroid, zero_velocity, measure))
        else
            error("Only 2D and 3D meshes are supported")
        end
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
                                   geometry::TwoPointFiniteVolumeGeometry) where {D, T}
    # Update velocity for each subcell
    for subcell in subcells
        cell = subcell.parent_cell
        face_idx = subcell.parent_face
        
        # Determine flux orientation (into or out of cell)
        neighbors = mesh.faces.neighbors[face_idx]
        if neighbors[1] == cell
            flux_sign = T(1.0)
        else
            flux_sign = T(-1.0)
        end
        flux = flux_sign * fluxes[face_idx]
        
        # Get face area and normal
        area = geometry.areas[face_idx]
        normal = SVector{D, T}(geometry.normals[:, face_idx])
        
        # Reconstruct velocity
        subcell.velocity = reconstruct_subcell_velocity(flux, area, normal, subcell.measure, geometry.volumes[cell])
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
    reconstruct_subcell_velocity(flux, face_area, normal, subcell_measure, cell_volume)

Reconstruct velocity in a sub-cell from face flux.
The velocity is computed as the flux per unit area in the normal direction.
All subcells from the same face share the same velocity.
"""
function reconstruct_subcell_velocity(flux::T, face_area::T, normal::SVector{D, T}, 
                                       subcell_measure::T, cell_volume::T) where {D, T}
    # Velocity is simply flux per unit area in the normal direction
    # All subcells from the same face should have the same velocity
    flux_velocity = (flux / face_area) * normal
    
    return flux_velocity
end
