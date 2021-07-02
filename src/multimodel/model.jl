function Base.show(io::IO, t::MIME"text/plain", m::MultiModel)
    submodels = m.models
    if get(io, :compact, false)
    else
    end
    println("MultiModel with $(length(submodels)) submodels:")
    for key in keys(submodels)
        m = submodels[key]
        s = m.system
        if hasproperty(m.domain, :grid)
            g = m.domain.grid
            println("$key: $(typeof(s)) ∈ $(typeof(g))")
        else
            println("$key: $(typeof(s)) ∈ $(typeof(m.domain))")
        end
    end
end


function number_of_models(model::MultiModel)
    return length(model.models)
end

function get_primary_variable_names(model::MultiModel)

end

function sort_secondary_variables!(model::MultiModel)
    for m in model.models
        sort_secondary_variables!(m)
    end
end

function setup_state!(state, model::MultiModel, init_values)
    error("Mutating version of setup_state not supported for multimodel.")
end

function setup_storage(model::MultiModel; state0 = setup_state(model), parameters = setup_parameters(model))
    storage = TervStorage()
    for key in submodels_symbols(model)
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
    debugstr = "Determining cross-terms\n"
    for target in keys(models)
        sources = Dict{Symbol, Any}()
        for source in keys(models)
            if target == source
                continue
            end
            tmpstr = "$source → $target:\n"

            target_model = models[target]
            source_model = models[source]
            d = Dict()
            found = false
            for (key, eq) in storage[target][:equations]
                ct = declare_cross_term(eq, target_model, source_model, target = target, source = source)
                if !isnothing(ct)
                    found = true
                    tmpstr *= String(key)*": Cross-term found.\n"
                end
                d[key] = ct
            end
            sources[source] = d
            if found
                debugstr *= tmpstr
            end
        end
        crossd[target] = sources
    end
    storage[:cross_terms] = crossd
    @debug debugstr
end

function declare_cross_term(eq::TervEquation, target_model, source_model; target = nothing, source = nothing)
    target_unit = associated_unit(eq)
    intersection = get_model_intersection(target_unit, target_model, source_model, target, source)
    if isnothing(intersection.target)
        # Declare nothing, so we can easily spot no overlap
        ct = nothing
    else
        ct = InjectiveCrossTerm(eq, target_model, source_model, intersection; target = target, source = source)
    end
    return ct
end

function align_equations_to_linearized_system!(storage, model::MultiModel; equation_offset = 0, variable_offset = 0)
    models = model.models
    ndofs = model.number_of_degrees_of_freedom
    lsys = storage[:LinearizedSystem]
    for key in keys(models)
        submodel = models[key]
        eqs = storage[key][:equations]
        nrow_end = align_equations_to_jacobian!(eqs, lsys.jac, submodel, equation_offset = equation_offset, variable_offset = variable_offset)
        nrow = nrow_end - equation_offset
        ndof = ndofs[key]
        @assert nrow == ndof "Submodels must have equal number of equations and degrees of freedom. Found $nrow equations and $ndof variables for submodel $key"
        equation_offset += ndof
        variable_offset += ndof # Assuming that each model by itself forms a well-posed, square Jacobian...
    end
end

function align_cross_terms_to_linearized_system!(storage, model::MultiModel; equation_offset = 0, variable_offset = 0)
    models = model.models
    ndofs = model.number_of_degrees_of_freedom

    lsys = storage[:LinearizedSystem]
    cross_terms = storage[:cross_terms]

    base_variable_offset = variable_offset
    # Iterate over targets (= rows)
    for target in keys(models)
        target_model = models[target]
        variable_offset = base_variable_offset
        # Iterate over sources (= columns)
        for source in keys(models)
            source_model = models[source]
            if source != target
                ct = cross_terms[target][source]
                eqs = storage[target][:equations]
                align_cross_terms_to_linearized_system!(ct, eqs, lsys, target_model, source_model, equation_offset = equation_offset, variable_offset = variable_offset)
                # Same number of rows as target, same number of columns as source
            end
            # Increment col and row offset
            variable_offset += ndofs[source]
        end
        equation_offset += ndofs[target]
    end
end


function align_cross_terms_to_linearized_system!(crossterms, equations, lsys, target::TervModel, source::TervModel; equation_offset = 0, variable_offset = 0)
    for ekey in keys(equations)
        eq = equations[ekey]
        ct = crossterms[ekey]

        if !isnothing(ct)
            align_to_jacobian!(ct, lsys.jac, target, source, equation_offset = equation_offset, variable_offset = variable_offset)
        end
        equation_offset += number_of_equations(target, eq)
    end
    return equation_offset
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
        equation_offset = 0
        for (key, eq) in equations
            x = cross_terms[key]
            if !isnothing(x)
                variable_offset = 0
                for u in get_primary_variable_ordered_units(source_model)
                    S = declare_sparsity(target_model, source_model, x, u, source_layout)
                    if !isnothing(S)
                        push!(I, S[1] .+ equation_offset)
                        push!(J, S[2] .+ variable_offset)
                    end
                    variable_offset += number_of_degrees_of_freedom(source_model, u)
                end
            end
            equation_offset += number_of_equations(target_model, eq)
        end
        I = vcat(I...)
        J = vcat(J...)
        V = zeros(F, length(I))
        sarg = (I, J, V, equation_offset, ncols)
    end
    return sarg
end

function get_sparse_arguments(storage, model::MultiModel, targets::Vector{Symbol}, sources::Vector{Symbol})
    I = []
    J = []
    V = []
    outstr = "Determining sparse pattern of $(length(targets))×$(length(sources)) models:\n"
    equation_offset = 0
    variable_offset = 0
    for target in targets
        variable_offset = 0
        n = 0
        for source in sources
            i, j, v, n, m = get_sparse_arguments(storage, model, target, source)
            if length(i) > 0
                push!(I, i .+ equation_offset)
                push!(J, j .+ variable_offset)
                push!(V, v)
                @assert maximum(i) <= n "I index exceeded $n for $source → $target (largest value: $(maximum(i))"
                @assert maximum(j) <= m "J index exceeded $m for $source → $target (largest value: $(maximum(j))"

                @assert minimum(i) >= 1 "I index was lower than 1 for $source → $target"
                @assert minimum(j) >= 1 "J index was lower than 1 for $source → $target"
            end
            outstr *= "$source → $target: $n rows and $m columns starting at $(equation_offset+1), $(variable_offset+1).\n"
            variable_offset += m
        end
        outstr *= "\n"
        equation_offset += n
    end
    @debug outstr
    I = vec(vcat(I...))
    J = vec(vcat(J...))
    V = vec(vcat(V...))
    return (I, J, V, equation_offset, variable_offset)
end

function setup_linearized_system!(storage, model::MultiModel)
    groups = model.groups
    models = model.models
    context = model.context

    candidates = [i for i in keys(models)]
    if isnothing(groups)
        # All Jacobians are grouped together and we assemble as a single linearized system
        if isnothing(context)
            context = models[1].context
        end
        layout = matrix_layout(context)
        sparse_arg = get_sparse_arguments(storage, model, candidates, candidates)
        lsys = LinearizedSystem(sparse_arg, context, layout)
    else
        ugroups = unique(groups)
        ng = length(ugroups)
    
        # We have multiple groups. Store as Matrix of sparse matrices
        lsys = Matrix{Any}(undef, ng, ng)
        use_groups_context = isnothing(context)
        for rowg in 1:ng
            t = candidates[groups .== rowg]
            if use_groups_context
                context = models[t[1]].context
            end
            layout = matrix_layout(context)
            for colg in 1:ng
                s = candidates[groups .== colg]
                sparse_arg = get_sparse_arguments(storage, model, t, s)
                lsys[rowg, colg] = LinearizedSystem(sparse_arg, context, layout, allocate_r = rowg == colg)
            end
        end
    end
    storage[:LinearizedSystem] = lsys
end

function initialize_storage!(storage, model::MultiModel; kwarg...)
    for key in submodels_symbols(model)
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
    @sync for target in keys(models)
        for source in keys(models)
            if source != target
                @async update_cross_terms_for_pair!(storage, model, source, target, arg...)
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
        update_cross_term!(ct, eqs[ekey], storage_t, storage_s, model_t, model_s, target, source, arg...)
    end
end

function apply_cross_terms!(storage, model::MultiModel, arg...)
    models = model.models
    @sync for target in keys(models)
        for source in keys(models)
            if source != target
                @async apply_cross_terms_for_pair!(storage, model, source, target, arg...)
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
        if !isnothing(ct)
            apply_cross_term!(eqs[ekey], ct, model_t, model_s, arg...)
        end
    end
end


function update_linearized_system!(storage, model::MultiModel; equation_offset = 0)
    lsys = storage.LinearizedSystem
    models = model.models
    offsets = get_submodel_degree_of_freedom_offsets(model)
    model_keys = submodels_symbols(model)
    @sync for (index, key) in enumerate(model_keys)
        m = models[key]
        s = storage[key]
        eqs = s.equations
        @async update_linearized_system!(lsys, eqs, m; equation_offset = offsets[index])
    end
    # Then, update cross terms
    @sync for target in model_keys
        for source in model_keys
            if source != target
                @async update_linearized_system_crossterms!(lsys, storage, model, source, target)
            end
        end
    end
end


function get_submodel_degree_of_freedom_offsets(model::MultiModel)
    n = cumsum(vcat([0], [i for i in values(model.number_of_degrees_of_freedom)]))
end

function submodels_symbols(model::MultiModel)
    return keys(model.models)
end

function update_linearized_system_crossterms!(lsys, storage, model::MultiModel, source, target)
    cross_terms = storage[:cross_terms][target][source]

    storage_t, = get_submodel_storage(storage, target)
    model_t, model_s = get_submodels(model, target, source)

    eqs = storage_t[:equations]
    for ekey in keys(eqs)
        ct = cross_terms[ekey]
        nz = get_nzval(lsys.jac)
        if !isnothing(ct)
            update_linearized_system_crossterm!(nz, model_t, model_s, ct::CrossTerm)
        end
    end
end

function setup_state(model::MultiModel, initializers)
    state = Dict()
    for key in submodels_symbols(model)
        m = model.models[key]
        init = initializers[key]
        state[key] = setup_state(m, init)
    end
    return state
end

function setup_parameters(model::MultiModel)
    p = Dict()
    for key in submodels_symbols(model)
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
    lsys = storage.LinearizedSystem
    errors = OrderedDict()
    for key in submodels_symbols(model)
        s = storage[key]
        m = model.models[key]
        eqs = s.equations

        conv, e, errors[key], = check_convergence(lsys, eqs, s, m; offset = offset, extra_out = true, tol = tol, kwarg...)
        # Outer model has converged when all submodels are converged
        converged = converged && conv
        err = max(e, err)
        offset += number_of_degrees_of_freedom(m)
    end
    if extra_out
        return (converged, err, errors)
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

function reset_to_previous_state!(storage, model::MultiModel)
    submodels_storage_apply!(storage, model, reset_to_previous_state!)
end

function update_after_step!(storage, model::MultiModel, dt, forces)
    for key in submodels_symbols(model)
        update_after_step!(storage[key], model.models[key], dt, forces[key])
    end
end

function update_before_step!(storage, model::MultiModel, dt, forces)
    for key in submodels_symbols(model)
        update_before_step!(storage[key], model.models[key], dt, forces[key])
    end
end

function apply_forces!(storage, model::MultiModel, dt, forces::Dict)
    for key in submodels_symbols(model)
        apply_forces!(storage[key], model.models[key], dt, forces[key])
    end
end

function submodels_storage_apply!(storage, model, f!, arg...)
    @sync for key in submodels_symbols(model)
        @async f!(storage[key], model.models[key], arg...)
    end
end

function get_output_state(storage, model::MultiModel)
    out = Dict{Symbol, NamedTuple}()
    models = model.models
    for key in keys(models)
        out[key] = get_output_state(storage[key], models[key])
    end
    out = convert_to_immutable_storage(out)
    return out
end

function get_submodel_storage(storage, arg...)
    map((x) -> storage[x], arg)
end

function get_submodels(model, arg...)
    map((x) -> model.models[x], arg)
end

function get_convergence_table(model::MultiModel, errors)
    get_convergence_table(submodels_symbols(model), errors)
end
