function cross_term_entities(ct::CrossTerm, eq::Pair{Symbol, E}, model) where E<:JutulEquation
    e = cross_term_entities(ct, eq[2], composite_submodel(model, eq[1]))
    return e
end

function cross_term_entities_source(ct::CrossTerm, eq::Pair{Symbol, E}, model) where E<:JutulEquation
    e = cross_term_entities_source(ct, eq[2], composite_submodel(model, eq[1]))
    return e
end

# function declare_sparsity(target_model, source_model, eq::Pair{Symbol, E}, x::CrossTerm, x_storage, entity_indices, target_entity, source_entity, row_layout, col_layout) where E<:JutulEquation
#     k, eq = eq
#     target_model = composite_submodel(target_model, k)
#     if source_model.system isa CompositeSystem
#         source_model = composite_submodel(source_model, k)
#     end
#     return declare_sparsity(target_model, source_model, eq, x, x_storage, entity_indices, target_entity, source_entity, row_layout, col_layout)
# end
