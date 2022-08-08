export number_of_cells, number_of_faces, number_of_half_faces, count_entities, get_entities, declare_entities, get_neighborship


function declare_entities(G::JutulGrid)
    return [(entity = Cells(), count = 1)]
end

function get_entities(D::JutulDomain)
    return [Cells()]
end

function count_entities(D::JutulDomain, ::Cells)
    # The default implementation yields a single cell and nothing else.
    1
end

function get_entities(D::DiscretizedDomain)
    return keys(D.entities)
end

function select_variables_domain_helper!(S, domain::DiscretizedDomain, model, f!)
    d = domain.discretizations
    if !isnothing(d)
        for k in keys(d)
            f!(S, d[k], model)
        end
    end
end

function select_primary_variables!(S, domain::DiscretizedDomain, model)
    select_variables_domain_helper!(S, domain, model, select_primary_variables!)
end

function select_secondary_variables!(S, domain::DiscretizedDomain, model)
    select_variables_domain_helper!(S, domain, model, select_secondary_variables!)
end

function select_parameters!(S, domain::DiscretizedDomain, model)
    select_variables_domain_helper!(S, domain, model, select_parameters!)
end

function select_equations!(S, domain::DiscretizedDomain, model)
    select_variables_domain_helper!(S, domain, model, select_equations!)
end

count_entities(D::DiscretizedDomain, entity::Cells) = D.entities[entity]
count_entities(D::DiscretizedDomain, entity) = D.entities[entity]

count_active_entities(D, entity; kwarg...) = count_entities(D, entity)
count_active_entities(D::DiscretizedDomain, entity; kwarg...) = count_active_entities(D, D.global_map, entity; kwarg...)

function number_of_cells(D::DiscretizedDomain)
    return count_entities(D, Cells())
end

function number_of_faces(D::DiscretizedDomain)
    return count_entities(D, Faces())
end

function number_of_half_faces(D::DiscretizedDomain)
    return 2*number_of_faces(D)
end

function positional_map(domain::JutulDomain, source_entity::JutulUnit, target_entity::JutulUnit)
    positional_map(domain.grid, source_entity, target_entity)
end

function positional_map(grid::JutulGrid, source_entity, target_entity)
    error("Not implemented.")
end

function positional_map(grid::JutulGrid, ::Cells, ::Faces)
    faces, facePos = get_facepos(grid.neighborship)
    return (indices = faces, pos = facePos)
end

function half_face_map(N, nc)
    faces, face_pos = get_facepos(N, nc)
    signs = similar(faces)
    cells = similar(faces)
    for i in 1:nc
        for j in face_pos[i]:(face_pos[i+1]-1)
            f = faces[j]
            l = N[1, f]
            r = N[2, f]
            if l == i
                signs[j] = 1
                c = r
            else
                @assert r == i
                c = l
                signs[j] = -1
            end
            cells[j] = c
        end
    end
    return (cells = cells, faces = faces, face_pos = face_pos, face_sign = signs)
end
