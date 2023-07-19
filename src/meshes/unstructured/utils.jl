function count_entities(g::UnstructuredMesh, ::Cells)
    return number_of_cells(g)
end

function count_entities(g::UnstructuredMesh, ::Faces)
    return number_of_faces(g)
end

function count_entities(g::UnstructuredMesh, ::BoundaryFaces)
    return number_of_boundary_faces(g)
end

function number_of_boundary_faces(g::UnstructuredMesh)
    return length(g.boundary_faces.faces_to_nodes)
end

function number_of_cells(g::UnstructuredMesh)
    return length(g.faces.cells_to_faces)
end

function number_of_faces(g::UnstructuredMesh)
    return length(g.faces.faces_to_nodes)
end

function Base.show(io::IO, t::MIME"text/plain", g::UnstructuredMesh)
    nc = number_of_cells(g)
    nf = number_of_faces(g)
    nb = number_of_boundary_faces(g)
    print(io, "UnstructuredMesh $nc cells, $nf faces and $nb boundary faces")
end
