struct EmbeddedMesh <: Jutul.FiniteVolumeMesh
    unstructured_mesh::UnstructuredMesh
    intersection_faces::Vector{Vector{Int}}
end

function EmbeddedMesh(mesh::UnstructuredMesh, faces)
    embedded_mesh, intersections = make_mesh_from_faces(mesh, faces)
    return EmbeddedMesh(embedded_mesh, intersections)
end

function Jutul.UnstructuredMesh(mesh::EmbeddedMesh)
    return mesh.unstructured_mesh
end

function make_mesh_from_faces(mesh, faces)

    # Make edges
    face_edges, num_edges_per_face, face_edge_signs, neighbors = get_face_edges(mesh, faces)
    edge_nodes, num_nodes_per_edge, node_points = get_edge_nodes(mesh, keys(neighbors))

    # Renumber nodes
    nodes = unique(edge_nodes)
    node_map = make_mapping(nodes)
    nodes = [node_map[n] for n in nodes]
    @assert nodes[1] == 1 && all(diff(nodes) .== 1)

    edge_nodes = [node_map[n] for n in edge_nodes]

    # Renumber edges
    edges = vcat(keys(neighbors)...)
    edge_map = make_mapping(edges)
    edges = [edge_map[e] for e in edges]
    @assert edges[1] == 1 && all(diff(edges) .== 1)

    face_edges = map(e -> edge_map[e], face_edges)

    # Renumber faces
    face_map = make_mapping(faces)
    faces = [face_map[f] for f in faces]
    @assert faces[1] == 1 && all(diff(faces) .== 1)

    neighbors = [[face_map[f] for f in n] for n in values(neighbors)]

    # Make edge_pos and face_pos vectors
    face_edge_pos = cumsum([1; num_edges_per_face])
    # edge_node_pos = cumsum([1; num_nodes_per_edge])

    # Handle intersection
    neighbors, num_ix_faces, edge_nodes, num_nodes_per_edge, face_edges, 
    face_edge_signs, face_edge_pos, intersection_faces = 
    split_intersections(
        neighbors, face_edges, face_edge_signs, face_edge_pos, length(faces), 
        edge_nodes, num_nodes_per_edge)
    edge_node_pos = cumsum([1; num_nodes_per_edge])

    # Fix orientation of edges
    N = fix_edge_orientation(neighbors, face_edges, face_edge_signs, face_edge_pos)
    
    # Create unstructured mesh
    mesh_2d = UnstructuredMesh(face_edges, face_edge_pos, edge_nodes, edge_node_pos, node_points, N)

    return mesh_2d, intersection_faces

end

function get_face_edges(mesh, faces)

    edge_neighbors = Dict{Tuple{Int, Int}, Vector{Int}}()
    face_edges = Vector{Tuple{Int, Int}}()
    face_edge_signs = Vector{Int}()
    num_edges_per_face = Vector{Int}(undef, length(faces))

    for (i, face) in enumerate(faces)
        nodes = mesh.faces.faces_to_nodes[face]

        num_nodes = length(nodes)

        for j = 1:num_nodes
            
            # Get edge nodes
            node_a = nodes[j]
            node_b = nodes[mod(j, num_nodes) + 1]
            # Make edge with consistent ordering
            edge, sgn = (node_a < node_b) ? ((node_a, node_b), 1) : ((node_b, node_a), -1)

            if haskey(edge_neighbors, edge)
                push!(edge_neighbors[edge], face)
            else
                edge_neighbors[edge] = [face]
            end
            push!(face_edges, edge)
            push!(face_edge_signs, sgn)
            
        end
        num_edges_per_face[i] = num_nodes

    end

    return face_edges, num_edges_per_face, face_edge_signs, edge_neighbors

end

function get_edge_nodes(mesh, edges)

    edge_nodes = vcat([vcat(n...) for n in edges]...)
    num_nodes_per_edge = [length(n) for n in edges]

    nodes = unique(edge_nodes)
    node_points = mesh.node_points[nodes]

    return edge_nodes, num_nodes_per_edge, node_points
    
end


function split_intersections(neighbors, face_edges, face_edge_signs, face_edge_pos, num_faces, edge_nodes, num_nodes_per_edge)

    new_neighbors = Vector{Vector{Int}}()
    new_edge_nodes = Vector{Int}()
    new_num_nodes_per_edge = Vector{Int}()
    
    current_edge_node_idx = 1
    
    # Store replacements: old_edge_idx -> Dict(face_idx => [new_edge_indices])
    # We use a Vector of Dicts
    replacements = [Dict{Int, Vector{Int}}() for _ in 1:length(neighbors)]

    intersection_faces = Vector{Vector{Int}}()

    # Iterate over original edges
    for (old_edge_idx, faces) in enumerate(neighbors)
        
        n_nodes = num_nodes_per_edge[old_edge_idx]
        nodes = edge_nodes[current_edge_node_idx : current_edge_node_idx + n_nodes - 1]
        current_edge_node_idx += n_nodes

        if length(faces) > 2
            # Intersection: create pairwise connections
            # Connect every pair (f_i, f_j)
            for i in 1:length(faces)
                for j in (i+1):length(faces)
                    f1 = faces[i]
                    f2 = faces[j]
                    
                    push!(new_neighbors, [f1, f2])
                    append!(new_edge_nodes, nodes)
                    push!(new_num_nodes_per_edge, n_nodes)
                    
                    new_edge_idx = length(new_neighbors)
                    
                    # Register replacement for f1
                    if !haskey(replacements[old_edge_idx], f1)
                        replacements[old_edge_idx][f1] = Int[]
                    end
                    push!(replacements[old_edge_idx][f1], new_edge_idx)
                    
                    # Register replacement for f2
                    if !haskey(replacements[old_edge_idx], f2)
                        replacements[old_edge_idx][f2] = Int[]
                    end
                    push!(replacements[old_edge_idx][f2], new_edge_idx)
                end
            end
            # Store faces that are part of intersection
            push!(intersection_faces, faces)
        else
            # Normal edge or boundary
            push!(new_neighbors, faces)
            append!(new_edge_nodes, nodes)
            push!(new_num_nodes_per_edge, n_nodes)
            
            new_edge_idx = length(new_neighbors)
            
            for f in faces
                if !haskey(replacements[old_edge_idx], f)
                    replacements[old_edge_idx][f] = Int[]
                end
                push!(replacements[old_edge_idx][f], new_edge_idx)
            end
        end
    end

    # Rebuild face_edges, face_edge_signs, face_edge_pos
    new_face_edges = Vector{Int}()
    new_face_edge_signs = Vector{Int}()
    new_face_edge_pos = Vector{Int}()
    push!(new_face_edge_pos, 1)

    for f in 1:num_faces
        # Iterate over old edges of this face
        start_pos = face_edge_pos[f]
        end_pos = face_edge_pos[f+1]-1
        
        for k in start_pos:end_pos
            old_edge = face_edges[k]
            sgn = face_edge_signs[k]
            
            # Look up replacements
            if haskey(replacements[old_edge], f)
                new_edges = replacements[old_edge][f]
                for new_e in new_edges
                    push!(new_face_edges, new_e)
                    push!(new_face_edge_signs, sgn) # Keep original sign
                end
            else
                error("Edge $old_edge not found in neighbors for face $f")
            end
        end
        push!(new_face_edge_pos, length(new_face_edges) + 1)
    end
    
    num_ix_faces = 0

    return new_neighbors, num_ix_faces, new_edge_nodes, new_num_nodes_per_edge, new_face_edges, new_face_edge_signs, new_face_edge_pos, intersection_faces

end

function fix_edge_orientation(neighbors, face_edges, face_edge_signs, face_edge_pos)

    function get_face_edge_sign(face, edge, face_edges, edge_signs, edge_pos)

        if face == 0
            return 0
        end

        # Handle virtual faces (index out of bounds of edge_pos)
        if face >= length(edge_pos)
            return 0
        end

        pos = edge_pos[face]:edge_pos[face+1]-1
        mask = face_edges[pos] .== edge
        
        if !any(mask)
             error("Edge $edge not found in face $face")
        end
        sgn = edge_signs[pos][mask]

        return sgn[1]

    end

    N = Matrix{Int}(undef, 2, length(neighbors))
    for (i, n) in enumerate(neighbors)
        if length(n) <= 2

            n = (length(n) == 1) ? vcat(n, 0) : n
            @assert length(n) == 2
            
            f1, f2 = n[1], n[2]
            
            sgn_l = get_face_edge_sign(f1, i, face_edges, face_edge_signs, face_edge_pos)
            sgn_r = get_face_edge_sign(f2, i, face_edges, face_edge_signs, face_edge_pos)
            
            # Determine orientation
            if sgn_l != 0
                if sgn_l == 1
                    N[:, i] .= [f1, f2]
                else
                    N[:, i] .= [f2, f1]
                end
            elseif sgn_r != 0
                if sgn_r == 1
                    N[:, i] .= [f2, f1]
                else
                    N[:, i] .= [f1, f2]
                end
            else
                 # Should not happen if at least one face is real
                 error("Unable to determine orientation for edge $i neighbors $n")
            end

        else
            @error "Face $i has more than two neighbors. Not implemented yet"

        end

    end

    return N

end

function make_mapping(v)

    mapping = Dict{typeof(v[1]), Int}()
    for (i, vi) in enumerate(v)
        mapping[vi] = i
    end
    return mapping

end

