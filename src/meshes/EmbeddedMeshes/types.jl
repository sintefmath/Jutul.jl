"""
    EmbeddedMesh <: FiniteVolumeMesh

A mesh type for representing embedded lower-dimensional features (such as
fractures or faults) within a computational domain. The mesh consists of
interconnected faces.

# Fields
- `unstructured_mesh::UnstructuredMesh`: The underlying unstructured mesh representation
- `intersection_neighbors::Vector{Vector{Int}}`: Embedded mesh cells grouped per intersection
- `intersection_faces::Vector{Vector{Int}}`: Face indices per intersection, aligned with
  `intersection_neighbors`. For `:remove`, these are boundary face indices; for `:star_delta`
  and `:keep`, these are internal face indices.
- `intersection_cells::Vector{Int}`: Embedded mesh cell indices created for kept intersections
"""
struct EmbeddedMesh <: Jutul.FiniteVolumeMesh
    unstructured_mesh::UnstructuredMesh
    parent_faces::Vector{Int}
    intersection_neighbors::Vector{Vector{Int}}
    intersection_faces::Vector{Vector{Int}}
    intersection_cells::Vector{Int}
end

"""
    EmbeddedMesh(mesh::UnstructuredMesh, faces)

Construct an embedded mesh from selected faces of an unstructured mesh.

# Arguments
- `mesh::UnstructuredMesh`: Parent mesh containing the faces
- `faces`: Collection of face indices to include in the embedded mesh

# Keyword arguments
- `intersection_strategy::Symbol = :keep`: Strategy for handling intersections.
    - `:keep` (default): Intersection edges are kept disconnected in the
      neighborship and stored in `intersections`.
    - `:star_delta`: Intersections with three or more faces are split into pairwise internal connections.

# Returns
- `EmbeddedMesh`: Embedded mesh made up of the specified faces.

"""
function EmbeddedMesh(mesh::UnstructuredMesh, faces; intersection_strategy = :star_delta)
    embedded_mesh, intersection_neighbors, intersection_faces, intersection_cells = make_mesh_from_faces(
        mesh,
        faces;
        intersection_strategy = intersection_strategy
    )
    return EmbeddedMesh(embedded_mesh, faces, intersection_neighbors, intersection_faces, intersection_cells)
end

function Jutul.UnstructuredMesh(mesh::EmbeddedMesh)
    return mesh.unstructured_mesh
end

"""
    make_mesh_from_faces(mesh, faces)

Helper for EmbeddedMesh constructor.
"""
function make_mesh_from_faces(mesh, faces; intersection_strategy = :star_delta)

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
    face_edge_signs, face_edge_pos, intersection_neighbors, intersection_edges,
    intersection_cells =
    split_intersections(
        neighbors,
        face_edges,
        face_edge_signs,
        face_edge_pos,
        length(faces),
        edge_nodes,
        num_nodes_per_edge;
        strategy = intersection_strategy
    )
    edge_node_pos = cumsum([1; num_nodes_per_edge])

    # Fix orientation of edges
    N = fix_edge_orientation(neighbors, face_edges, face_edge_signs, face_edge_pos)

    # Remap raw edge indices to boundary-local or internal face indices
    intersection_edges = remap_intersection_edges(intersection_edges, N, intersection_strategy)
    
    # Create unstructured mesh
    mesh_2d = UnstructuredMesh(face_edges, face_edge_pos, edge_nodes, edge_node_pos, node_points, N)

    return mesh_2d, intersection_neighbors, intersection_edges, intersection_cells

end

function remap_intersection_edges(intersection_edges, N, strategy)
    nfaces = size(N, 2)
    if strategy == :remove
        # Boundary edges: remap to boundary-local face indices
        boundary_mesh_faces = findall(i -> N[1, i] == 0 || N[2, i] == 0, 1:nfaces)
        face_map = Dict{Int, Int}(f => i for (i, f) in enumerate(boundary_mesh_faces))
        label = "boundary"
    else
        # Internal edges: remap to internal face indices
        internal_mesh_faces = findall(i -> N[1, i] != 0 && N[2, i] != 0, 1:nfaces)
        face_map = Dict{Int, Int}(f => i for (i, f) in enumerate(internal_mesh_faces))
        label = "internal"
    end

    mapped = Vector{Vector{Int}}(undef, length(intersection_edges))
    for i in eachindex(intersection_edges)
        ix = intersection_edges[i]
        mapped_ix = Int[]
        sizehint!(mapped_ix, length(ix))
        for f in ix
            haskey(face_map, f) || error("Intersection edge $f is not a $label face in constructed mesh.")
            push!(mapped_ix, face_map[f])
        end
        mapped[i] = mapped_ix
    end
    return mapped
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


function split_intersections(
    neighbors,
    face_edges,
    face_edge_signs,
    face_edge_pos,
    num_faces,
    edge_nodes,
    num_nodes_per_edge;
    strategy = :star_delta,
)

    new_neighbors = Vector{Vector{Int}}()
    new_edge_nodes = Vector{Int}()
    new_num_nodes_per_edge = Vector{Int}()
    
    current_edge_node_idx = 1
    
    # Store replacements: old_edge_idx -> Dict(face_idx => [new_edge_indices])
    # We use a Vector of Dicts
    replacements = [Dict{Int, Vector{Int}}() for _ in 1:length(neighbors)]

    intersection_neighbors = Vector{Vector{Int}}()
    intersection_edges = Vector{Vector{Int}}()
    intersection_cells = Int[]
    intersection_cell_edges = Vector{Vector{Int}}()
    intersection_cell_edge_signs = Vector{Vector{Int}}()

    function register_replacement!(old_edge_idx, face_idx, new_edge_idx)
        if !haskey(replacements[old_edge_idx], face_idx)
            replacements[old_edge_idx][face_idx] = Int[]
        end
        push!(replacements[old_edge_idx][face_idx], new_edge_idx)
    end

    # Iterate over original edges
    for (old_edge_idx, faces) in enumerate(neighbors)
        
        n_nodes = num_nodes_per_edge[old_edge_idx]
        nodes = edge_nodes[current_edge_node_idx : current_edge_node_idx + n_nodes - 1]
        current_edge_node_idx += n_nodes

        if length(faces) > 2
            if strategy == :star_delta
                # Intersection: create pairwise internal connections.
                ix_edges = Int[]
                for i in 1:length(faces)
                    for j in (i+1):length(faces)
                        f1 = faces[i]
                        f2 = faces[j]

                        push!(new_neighbors, [f1, f2])
                        append!(new_edge_nodes, nodes)
                        push!(new_num_nodes_per_edge, n_nodes)

                        new_edge_idx = length(new_neighbors)
                        push!(ix_edges, new_edge_idx)
                        register_replacement!(old_edge_idx, f1, new_edge_idx)
                        register_replacement!(old_edge_idx, f2, new_edge_idx)
                    end
                end
                push!(intersection_edges, ix_edges)
            elseif strategy == :remove
                # Intersection: duplicate as boundary edge per face.
                ix_boundary_edges = Int[]
                for f in faces
                    push!(new_neighbors, [f])
                    append!(new_edge_nodes, nodes)
                    push!(new_num_nodes_per_edge, n_nodes)

                    new_edge_idx = length(new_neighbors)
                    push!(ix_boundary_edges, new_edge_idx)
                    register_replacement!(old_edge_idx, f, new_edge_idx)
                end
                push!(intersection_edges, ix_boundary_edges)
            elseif strategy == :keep
                ix_cell = num_faces + length(intersection_cells) + 1
                ix_edges = Int[]
                ix_edge_signs = Int[]

                for f in faces
                    push!(new_neighbors, [f, ix_cell])
                    append!(new_edge_nodes, nodes)
                    push!(new_num_nodes_per_edge, n_nodes)

                    new_edge_idx = length(new_neighbors)
                    push!(ix_edges, new_edge_idx)
                    register_replacement!(old_edge_idx, f, new_edge_idx)

                    face_positions = face_edge_pos[f]:(face_edge_pos[f + 1] - 1)
                    face_pos = findfirst(face_edges[face_positions] .== old_edge_idx)
                    isnothing(face_pos) && error("Edge $old_edge_idx not found in face $f")
                    push!(ix_edge_signs, -face_edge_signs[face_positions[face_pos]])
                end

                push!(intersection_edges, ix_edges)
                push!(intersection_cells, ix_cell)
                push!(intersection_cell_edges, ix_edges)
                push!(intersection_cell_edge_signs, ix_edge_signs)
            else
                error("Unknown intersection strategy: $strategy")
            end
            # Store faces that are part of intersection
            push!(intersection_neighbors, faces)
        else
            # Normal edge or boundary
            push!(new_neighbors, faces)
            append!(new_edge_nodes, nodes)
            push!(new_num_nodes_per_edge, n_nodes)
            
            new_edge_idx = length(new_neighbors)
            
            for f in faces
                register_replacement!(old_edge_idx, f, new_edge_idx)
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

    for (ix_edges, ix_signs) in zip(intersection_cell_edges, intersection_cell_edge_signs)
        append!(new_face_edges, ix_edges)
        append!(new_face_edge_signs, ix_signs)
        push!(new_face_edge_pos, length(new_face_edges) + 1)
    end

    num_ix_faces = length(intersection_cells)

    return new_neighbors, num_ix_faces, new_edge_nodes, new_num_nodes_per_edge, new_face_edges, new_face_edge_signs, new_face_edge_pos, intersection_neighbors, intersection_edges, intersection_cells

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

