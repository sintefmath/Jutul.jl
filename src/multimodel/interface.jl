function update_cross_term_in_entity!

end

function cross_term_entities_source

end

function cross_term_entities

end

function has_symmetry

end

"""
    prepare_cross_term(cross_term, target_model, source_model)

Configure a cross term once when constructing a simulator, after the variable
definitions are finalized. Applications can capture typed constitutive
evaluators here instead of inspecting model definitions in entity kernels.
The default returns the cross term unchanged.
"""
prepare_cross_term(ct::CrossTerm, target_model, source_model) = ct


function prepare_cross_term_in_entity!(
        i,
        state_target, state0_target,
        state_source, state0_source,
        target_model, source_model,
        ct::CrossTerm, eq, dt, ldisc = local_discretization(ct, i)
    )
    return nothing
end

export subcrossterm

function subcrossterm(ct, ctp, m_t, m_s, ::TrivialGlobalMap, ::TrivialGlobalMap, partition)
    # All maps are trivial, reuse cross term directly
    return ct
end
