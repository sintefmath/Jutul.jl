
function get_model_intersection(u, target_model, source_model, target, source)
    return get_domain_intersection(u, target_model.domain, source_model.domain, target, source)
end

"""
For a given unit in domain target_d, find any indices into that unit that is connected to
any units in source_d. The interface is limited to a single unit-unit impact.
The return value is a tuple of indices and the corresponding unit
"""
function get_domain_intersection(u, target_d, source_d, target_symbol, source_symbol)
    source_symbol::Union{Nothing, Symbol}
    (target = nothing, source = nothing, target_unit = u, source_unit = Cells())
end

function update_cross_term!(ct::InjectiveCrossTerm, eq, target_storage, source_storage, target_model, source_model, target, source, dt)
    error("Cross term must be specialized for your equation and models. Did not understand how to specialize $target ($(typeof(target_model))) to $source ($(typeof(source_model)))")
end

function update_cross_term!(::Nothing, arg...)
    # Do nothing.
end
