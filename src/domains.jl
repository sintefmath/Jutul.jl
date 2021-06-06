function declare_units(G::TervGrid)
    return [(unit = Cells(), count = 1)]
end

function get_units(D::TervDomain)
    return [Cells()]
end

function count_units(D::TervDomain, ::Cells)
    # The default implementation yields a single cell and nothing else.
    1
end

function get_units(D::DiscretizedDomain)
    return keys(D.units)
end


function select_secondary_variables_domain!(S, domain::DiscretizedDomain, system, formulation)
    d = domain.discretizations
    for k in keys(d)
        select_secondary_variables_discretization!(S, domain, system, formulation, d[k])
    end
end

function select_secondary_variables_discretization!(S, domain, system, formulation, disc)

end


function select_primary_variables_domain!(S, domain::DiscretizedDomain, system, formulation)
    d = domain.discretizations
    for k in keys(d)
        select_primary_variables_domain!(S, domain, system, formulation, d[k])
    end
end

function select_primary_variables_domain!(S, domain, system, formulation, disc)

end

function count_units(D::DiscretizedDomain, unit::Cells)
    D.units[unit]
end

function count_units(D::DiscretizedDomain, unit)
    D.units[unit]
end

function number_of_cells(D::DiscretizedDomain)
    return count_units(D, Cells())
end

function number_of_faces(D::DiscretizedDomain)
    return count_units(D, Faces())
end

function number_of_half_faces(D::DiscretizedDomain)
    return 2*number_of_faces(D)
end

function positional_map(domain::TervDomain, source_unit::TervUnit, target_unit::TervUnit)
    positional_map(domain.grid, source_unit, target_unit)
end

function positional_map(grid::TervGrid, source_unit, target_unit)
    error("Not implemented.")
end

function positional_map(grid::TervGrid, ::Cells, ::Faces)
    faces, facePos = get_facepos(grid.neighborship)
    (indices = faces, pos = facePos)
end


