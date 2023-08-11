function setup_helper_equation_storage!(storage, r, mm::MultiModel; offset = 0)
    for (k, model) in pairs(mm.models)
        offset = setup_helper_equation_storage!(storage[k], r, model, offset = offset)
    end
    T = eltype(r)
    for (ct_p, ct_s) in zip(mm.cross_terms, storage.cross_terms)
        ct = ct_p.cross_term
        model_t = mm[ct_p.target]
        eq_t = model_t.equations[ct_p.target_equation]

        active = cross_term_entities(ct, eq_t, model_t)
        N = length(active)
        n = number_of_equations_per_entity(model_t, eq_t)
        active = cross_term_entities(ct, eq_t, model_t)
        N = length(active)
        x = zeros(T, n, N)
        ct_s[:target] = x
        ct_s[:source] = x
        ct_s[:helper_mode] = true
    end
end
