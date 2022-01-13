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


function select_secondary_variables_domain!(S, domain::DiscretizedDomain, system, formulation)
    d = domain.discretizations
    if !isnothing(d)
        for k in keys(d)
            select_secondary_variables_discretization!(S, domain, system, formulation, d[k])
        end
    end
end

function select_secondary_variables_discretization!(S, domain, system, formulation, disc)

end


function select_primary_variables_domain!(S, domain::DiscretizedDomain, system, formulation)
    d = domain.discretizations
    if !isnothing(d)
        for k in keys(d)
            select_primary_variables_domain!(S, domain, system, formulation, d[k])
        end
    end
end

function select_primary_variables_domain!(S, domain, system, formulation, disc)

end

count_entities(D::DiscretizedDomain, entity::Cells) = D.entities[entity]
count_entities(D::DiscretizedDomain, entity) = D.entities[entity]

count_active_entities(D, entity) = count_entities(D, entity)
count_active_entities(D::DiscretizedDomain, entity) = count_active_entities(D, D.global_map, entity)

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
    (indices = faces, pos = facePos)
end


