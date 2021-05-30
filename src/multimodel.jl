export MultiModel

struct MultiModel <: TervModel
    models::NamedTuple
    groups::Vector
    context::TervContext
    number_of_degrees_of_freedom
    function MultiModel(models; groups = nothing, context = DefaultContext())
        nm = length(models)
        if isnothing(groups)
            groups = ones(Int64, nm)
        end
        @assert maximum(groups) <= nm
        @assert minimum(groups) > 0
        @assert length(groups) == nm
        for m in models
            @assert context == m.context
        end
        ndof = map(number_of_degrees_of_freedom, models)
        new(models, groups, context, ndof)
    end
end

abstract type CrossTerm end

"""
A cross model term where the dependency is injective and the term is additive:
(each addition to a unit in the target only depends one unit from the source, 
and is added into that position upon application)
"""
struct InjectiveCrossTerm <: CrossTerm
    impact                 # 2 by N - first row is target, second is source
    units                  # tuple - first tuple is target, second is source
    crossterm_target       # The cross-term, with AD values taken relative to the targe
    crossterm_source       # Same cross-term, AD values taken relative to the source
    crossterm_source_cache # The cache that holds crossterm_source together with the entries.
    equations_per_unit     # Number of equations per impact
    npartials_target       # Number of partials per equation (in target)
    npartials_source       # (in source)
    function InjectiveCrossTerm(target_eq, target_model, source_model, intersection = nothing; target = nothing, source = nothing)
        context = target_model.context
        target_unit = associated_unit(target_eq)
        if isnothing(intersection)
            intersection = get_domain_intersection(target_unit, target_model, source_model, source)
        end
        target_impact, source_impact, source_unit = intersection
        @assert !isnothing(target_impact) "Cannot declare cross term when there is no overlap between domains."
        noverlap = length(target_impact)
        @assert noverlap == length(source_impact) "Injective source must have one to one mapping between impact and source."
        # Infer Unit from target_eq
        equations_per_unit = number_of_equations_per_unit(target_eq)

        npartials_target = number_of_partials_per_unit(target_model, target_unit)
        npartials_source = number_of_partials_per_unit(source_model, source_unit)

        target_tag = get_unit_tag(target, target_unit)
        c_term_target = allocate_array_ad(equations_per_unit, noverlap, context = context, npartials = npartials_target, tag = target_tag)
        c_term_source_c = CompactAutoDiffCache(equations_per_unit, noverlap, npartials_source, context = context, tag = source)
        c_term_source = c_term_source_c.entries

        # Units and overlap - target, then source
        units = (target = target_unit, source = source_unit)
        overlap = (target = target_impact, source = source_impact)
        new(overlap, units, c_term_target, c_term_source, c_term_source_c, equations_per_unit, npartials_target, npartials_target)
    end
end

function align_to_jacobian!(ct::InjectiveCrossTerm, jac, target::TervModel, source::TervModel; row_offset = 0, col_offset = 0)
    cs = ct.crossterm_source_cache

    layout = matrix_layout(source.context)

    impact_target = ct.impact[1]
    impact_source = ct.impact[2]

    injective_alignment!(cs, jac, layout, target_index = impact_target, source_index = impact_source, target_offset = row_offset, source_offset = col_offset)
end

function apply_cross_term!(eq, ct, model_t, model_s, arg...)
    ix = ct.impact.target
    d = get_diagonal_part(eq)
    # TODO: Why is this allocating?
    d[:, ix] += ct.crossterm_target
end


function update_linearized_system_subset!(jac, model_t, model_s, ct::InjectiveCrossTerm)
    update_linearized_system_subset!(jac, nothing, model_s, ct.crossterm_source_cache)
end

function declare_pattern(target_model, source_model, x::InjectiveCrossTerm, unit)
    source_unit = x.units.source
    if unit == source_unit
        nunits_source = count_units(source_model.domain, source_unit)

        target_impact = x.impact.target
        source_impact = x.impact.source

        n_impact = length(target_impact)

        out = (target_impact, source_impact, n_impact, nunits_source)
    else
        out = nothing
    end
    return out
end

function declare_sparsity(target_model, source_model, x::CrossTerm, unit, layout::EquationMajorLayout)
    primitive = declare_pattern(target_model, source_model, x, unit)
    if isnothing(primitive)
        out = nothing
    else
        target_impact = primitive[1]
        source_impact = primitive[2]
        n_impact = primitive[3]
        nunits_source = primitive[4]

        n_partials = x.npartials_source
        n_eqs = x.equations_per_unit
        I = []
        J = []
        for eqno in 1:n_eqs
            for derno in 1:n_partials
                push!(I, target_impact .+ (eqno-1)*n_impact)
                push!(J, source_impact .+ (derno-1)*nunits_source)
            end
        end
        I = vcat(I...)
        J = vcat(J...)

        n = n_eqs*n_impact
        m = n_partials*nunits_source
        out = (I, J, n, m)
    end
    return out
end


function get_domain_intersection(u::TervUnit, target_model::TervModel, source_model::TervModel, arg...)
    return get_domain_intersection(u, target_model.domain, source_model.domain, arg...)
end

"""
For a given unit in domain target_d, find any indices into that unit that is connected to
any units in source_d. The interface is limited to a single unit-unit impact.
The return value is a tuple of indices and the corresponding unit
"""
function get_domain_intersection(u::TervUnit, target_d::TervDomain, source_d::TervDomain, source_symbol)
    source_symbol::Union{Nothing, Symbol}
    (target = nothing, source = nothing, source_unit = Cells())
end

function number_of_models(model::MultiModel)
    return length(model.models)
end

function get_primary_variable_names(model::MultiModel)

end

function setup_state!(state, model::MultiModel, init_values)
    error("Mutating version of setup_state not supported for multimodel.")
end

function setup_storage(model::MultiModel; state0 = setup_state(model), parameters = setup_parameters(model))
    storage = Dict()
    for key in keys(model.models)
        m = model.models[key]
        storage[key] = setup_storage(m,  state0 = state0[key], 
                                            parameters = parameters[key], 
                                            setup_linearized_system = false,
                                            tag = key)
    end
    setup_cross_terms(storage, model)
    setup_linearized_system!(storage, model)
    align_equations_to_linearized_system!(storage, model)
    align_cross_terms_to_linearized_system!(storage, model)
    return storage
end

function setup_cross_terms(storage, model::MultiModel)
    crossd = Dict{Symbol, Any}()
    models = model.models
    for target in keys(models)
        sources = Dict{Symbol, Any}()
        for source in keys(models)
            if target == source
                continue
            end
            target_model = models[target]
            source_model = models[source]
            d = Dict()
            for (key, eq) in storage[target][:equations]
                ct = declare_cross_term(eq, target_model, source_model, target = target, source = source)
                d[key] = ct
            end
            sources[source] = d
        end
        crossd[target] = sources
    end
    storage[:cross_terms] = crossd
end

function declare_cross_term(eq::TervEquation, target_model, source_model; source = nothing, kwarg...)
    target_unit = associated_unit(eq)
    intersection = get_domain_intersection(target_unit, target_model, source_model, source)
    if isnothing(intersection.target)
        # Declare nothing, so we can easily spot no overlap
        ct = nothing
    else
        ct = InjectiveCrossTerm(eq, target_model, source_model, intersection; source = source, kwarg...)
    end
    return ct
end

function align_equations_to_linearized_system!(storage, model::MultiModel; row_offset = 0, col_offset = 0)
    models = model.models
    ndofs = model.number_of_degrees_of_freedom
    lsys = storage[:LinearizedSystem]
    for key in keys(models)
        submodel = models[key]
        eqs = storage[key][:equations]
        nrow_end = align_equations_to_jacobian!(eqs, lsys.jac, submodel, row_offset = row_offset, col_offset = col_offset)
        nrow = nrow_end - row_offset
        ndof = ndofs[key]
        @assert nrow == ndof "Submodels must have equal number of equations and degrees of freedom. Found $nrow equations and $ndof variables for submodel $key"
        row_offset += ndof
        col_offset += ndof # Assuming that each model by itself forms a well-posed, square Jacobian...
    end
end

function align_cross_terms_to_linearized_system!(storage, model::MultiModel; row_offset = 0, col_offset = 0)
    models = model.models
    ndofs = model.number_of_degrees_of_freedom

    lsys = storage[:LinearizedSystem]
    cross_terms = storage[:cross_terms]

    base_col_offset = col_offset
    # Iterate over targets (= rows)
    for target in keys(models)
        target_model = models[target]
        col_offset = base_col_offset
        # Iterate over sources (= columns)
        for source in keys(models)
            source_model = models[source]
            if source != target
                ct = cross_terms[target][source]
                eqs = storage[target][:equations]
                align_cross_terms_to_linearized_system!(ct, eqs, lsys, target_model, source_model, row_offset = row_offset, col_offset = col_offset)
                # Same number of rows as target, same number of columns as source
            end
            # Increment col and row offset
            col_offset += ndofs[source]
        end
        row_offset += ndofs[target]
    end
end


function align_cross_terms_to_linearized_system!(crossterms, equations, lsys, target::TervModel, source::TervModel; row_offset = 0, col_offset = 0)
    for ekey in keys(equations)
        eq = equations[ekey]
        ct = crossterms[ekey]
        if !isnothing(ct)
            align_to_jacobian!(ct, lsys.jac, target, source, row_offset = row_offset, col_offset = col_offset)
        end
        row_offset += number_of_equations(target, eq)
    end
    return row_offset
end

function get_sparse_arguments(storage, model::MultiModel, target::Symbol, source::Symbol)
    models = model.models
    target_model = models[target]
    source_model = models[source]
    source_layout = matrix_layout(source_model.context)
    F = float_type(source_model.context)

    if target == source
        # These are the main diagonal blocks each model knows how to produce themselves
        sarg = get_sparse_arguments(storage[target], target_model)
    else
        # Source differs from target. We need to get sparsity from cross model terms.
        I = []
        J = []
        ncols = number_of_degrees_of_freedom(source_model)
        # Loop over target equations and get the sparsity of the sources for each equation - with
        # derivative positions that correspond to that of the source
        equations = storage[target][:equations]
        cross_terms = storage[:cross_terms][target][source]
        row_offset = 0
        for (key, eq) in equations
            x = cross_terms[key]
            if !isnothing(x)
                col_offset = 0
                for u in get_primary_variable_ordered_units(source_model)
                    S = declare_sparsity(target_model, source_model, x, u, source_layout)
                    if !isnothing(S)
                        push!(I, S[1] .+ row_offset)
                        push!(J, S[2] .+ col_offset)
                    end
                    col_offset += degrees_of_freedom_per_unit(source_model, u)
                end
            end
            row_offset += number_of_equations(target_model, eq)
        end
        I = vcat(I...)
        J = vcat(J...)
        V = zeros(F, length(I))
        sarg = (I, J, V, row_offset, ncols)
    end
    return sarg
end

function get_sparse_arguments(storage, model::MultiModel, targets::Vector{Symbol}, sources::Vector{Symbol})
    I = []
    J = []
    V = []
    row_offset = 0
    col_offset = 0
    for target in targets
        col_offset = 0
        n = 0
        for source in sources
            i, j, v, n, m = get_sparse_arguments(storage, model, target, source)
            push!(I, i .+ row_offset)
            push!(J, j .+ col_offset)
            push!(V, v)
            col_offset += m
            @debug "$source → $target: $n rows and $m columns."
        end
        row_offset += n
    end
    I = vcat(I...)
    J = vcat(J...)
    V = vcat(V...)
    return (I, J, V, row_offset, col_offset)
end

function setup_linearized_system!(storage, model::MultiModel)
    F = float_type(model.context)

    groups = model.groups
    models = model.models
    ugroups = unique(groups)
    ng = length(ugroups)

    candidates = [i for i in keys(models)]
    if ng == 1
        # All Jacobians are grouped together and we assemble as a single linearized system
        I, J, V, n, m = get_sparse_arguments(storage, model, candidates, candidates)
        jac = sparse(I, J, V, n, m)
        lsys = LinearizedSystem(jac)
    else
        # We have multiple groups. Store as Matrix of sparse matrices
        @assert false "Needs implementation"
        jac = Matrix{Any}(ng, ng)
        # row_offset = 0
        # col_offset = 0
        for rowg in 1:ng
            t = candidates[groups .== rowg]
            for colg in 1:ng
                s = candidates[groups .== colg]
                I, J, V, n, m = get_sparse_arguments(storage, model, t, s)
            end
        end
    end
    storage[:LinearizedSystem] = lsys
end

function initialize_storage!(storage, model::MultiModel; kwarg...)
    for key in keys(model.models)
        initialize_storage!(storage[key], model.models[key]; kwarg...)
    end
end

function update_equations!(storage, model::MultiModel, arg...)
    # First update all equations
    submodels_storage_apply!(storage, model, update_equations!, arg...)
    # Then update the cross terms
    update_cross_terms!(storage, model::MultiModel, arg...)
    # Finally apply cross terms to the equations
    apply_cross_terms!(storage, model::MultiModel, arg...)
end

function update_cross_terms!(storage, model::MultiModel, arg...)
    models = model.models
    for target in keys(models)
        for source in keys(models)
            if source != target
                update_cross_terms_for_pair!(storage, model, source, target, arg...)
            end
        end
    end
end

function update_cross_terms_for_pair!(storage, model, source::Symbol, target::Symbol, arg...)
    cross_terms = storage[:cross_terms][target][source]

    storage_t, storage_s = get_submodel_storage(storage, target, source)
    model_t, model_s = get_submodels(model, target, source)

    eqs = storage_t[:equations]
    for ekey in keys(eqs)
        ct = cross_terms[ekey]
        update_cross_term!(ct, eqs[ekey], storage_t, storage_s, model_t, model_s, arg...)
    end
end

function update_cross_term!(ct::InjectiveCrossTerm, eq, target_storage, source_storage, target, source, dt)
    error("Cross term must be specialized for your equation and models.")
end

function update_cross_term!(::Nothing, arg...)
    # Do nothing.
end

function apply_cross_terms!(storage, model::MultiModel, arg...)
    models = model.models
    for target in keys(models)
        for source in keys(models)
            if source != target
                apply_cross_terms_for_pair!(storage, model, source, target, arg...)
            end
        end
    end
end

function apply_cross_terms_for_pair!(storage, model, source::Symbol, target::Symbol, arg...)
    cross_terms = storage[:cross_terms][target][source]

    storage_t, = get_submodel_storage(storage, target)
    model_t, model_s = get_submodels(model, target, source)

    eqs = storage_t[:equations]
    for ekey in keys(eqs)
        ct = cross_terms[ekey]
        apply_cross_term!(eqs[ekey], ct, model_t, model_s, arg...)
    end
end


function update_linearized_system!(storage, model::MultiModel; row_offset = 0)
    lsys = storage.LinearizedSystem
    models = model.models
    for key in keys(models)
        m = models[key]
        s = storage[key]
        eqs = s.equations
        update_linearized_system!(lsys, eqs, m; row_offset = row_offset)
        row_offset += number_of_degrees_of_freedom(m)
    end
    # Then, update cross terms
    for target in keys(models)
        for source in keys(models)
            if source != target
                update_linearized_system_crossterms!(lsys, storage, model, source, target)
            end
        end
    end
end

function update_linearized_system_crossterms!(lsys, storage, model::MultiModel, source, target)
    cross_terms = storage[:cross_terms][target][source]

    storage_t, = get_submodel_storage(storage, target)
    model_t, model_s = get_submodels(model, target, source)

    eqs = storage_t[:equations]
    for ekey in keys(eqs)
        ct = cross_terms[ekey]
        update_linearized_system_subset!(lsys.jac, model_t, model_s, ct::CrossTerm)
    end
end

function setup_state(model::MultiModel, initializers)
    state = Dict()
    for key in keys(model.models)
        m = model.models[key]
        init = initializers[key]
        state[key] = setup_state(m, init)
    end
    return state
end

function setup_parameters(model::MultiModel)
    p = Dict()
    for key in keys(model.models)
        m = model.models[key]
        p[key] = setup_parameters(m)
    end
    return p
end

function update_secondary_variables!(storage, model::MultiModel)
    submodels_storage_apply!(storage, model, update_secondary_variables!)
end

function check_convergence(storage, model::MultiModel; tol = 1e-3, extra_out = false, kwarg...)
    converged = true
    err = 0
    offset = 0
    r = storage.LinearizedSystem.r
    for key in keys(model.models)
        @debug "Checking convergence for submodel $key:"
        s = storage[key]
        m = model.models[key]
        eqs = s.equations

        n = number_of_degrees_of_freedom(m)
        ix = offset .+ (1:n)
        r_v = view(r, ix)
        conv, e, = check_convergence(r_v, eqs, s, m; extra_out = true, tol = tol, kwarg...)
        # Outer model has converged when all submodels are converged
        converged = converged && conv
        err = max(e, err)
        offset += n
    end
    if extra_out
        return (converged, err, tol)
    else
        return converged
    end
end

function update_primary_variables!(storage, model::MultiModel)
    dx = storage.LinearizedSystem.dx
    models = model.models

    offset = 0
    for key in keys(models)
        m = models[key]
        s = storage[key]
        ndof = number_of_degrees_of_freedom(m)
        dx_v = view(dx, (offset+1):(offset+ndof))
        update_primary_variables!(s.state, dx_v, m)
        offset += ndof
    end
end

function update_after_step!(storage, model::MultiModel)
    submodels_storage_apply!(storage, model, update_after_step!)
end

function apply_forces!(storage, model::MultiModel, dt, forces::Dict)
    for key in keys(model.models)
        apply_forces!(storage[key], model.models[key], dt, forces[key])
    end
end

function submodels_storage_apply!(storage, model, f!, arg...)
    for key in keys(model.models)
        f!(storage[key], model.models[key], arg...)
    end
end

function get_output_state(storage, model::MultiModel)
    out = Dict{Symbol, NamedTuple}()
    models = model.models
    for key in keys(models)
        out[key] = get_output_state(storage[key], models[key])
    end
    return out
end

function get_submodel_storage(storage, arg...)
    map((x) -> storage[x], arg)
end

function get_submodels(model, arg...)
    map((x) -> model.models[x], arg)
end
