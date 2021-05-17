function declare_units(G::TervGrid)
    return [(Cells(), 1)]
end

function count_units(D::TervDomain, ::Cells)
    # The default implementation yields a single cell and nothing else.
    1
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
