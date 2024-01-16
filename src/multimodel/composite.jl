function cross_term_entities(ct::CrossTerm, eq::Pair{Symbol, E}, model) where E<:JutulEquation
    e = cross_term_entities(ct, eq[2], composite_submodel(model, eq[1]))
    return e
end

function cross_term_entities_source(ct::CrossTerm, eq::Pair{Symbol, E}, model) where E<:JutulEquation
    e = cross_term_entities_source(ct, eq[2], composite_submodel(model, eq[1]))
    return e
end
