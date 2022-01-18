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

has_groups(model::MultiModel) = !isnothing(model.groups)

function number_of_groups(model::MultiModel)
    if has_groups(model)
        n = maximum(model.groups)
    else
        n = 1
    end
end


function get_primary_variable_names(model::MultiModel)

end

function sort_secondary_variables!(model::MultiModel)
    for m in model.models
        sort_secondary_variables!(m)
    end
end

function cross_term(storage, target)
    return storage[:cross_terms][target]
end

function cross_term_pair(storage, source, target)
    ct = cross_term(storage, target)
    if haskey(ct, source)
        x = ct[source]
    else
        x = nothing
    end
    return x
end

function target_cross_term_keys(storage, target)
    return keys(cross_term(storage, target))
end

@inline submodel_symbols(model::MultiModel) = keys(model.models)

function setup_state!(state, model::MultiModel, init_values)
    error("Mutating version of setup_state not supported for multimodel.")
end

function setup_storage(model::MultiModel; state0 = setup_state(model), parameters = setup_parameters(model))
    storage = JutulStorage()
    for key in submodels_symbols(model)
        m = model.models[key]
        storage[key] = setup_storage(m,  state0 = state0[key],
                                            parameters = parameters[key],
                                            setup_linearized_system = false,
                                            tag = key)
    end
    couplings = model.couplings;
    setup_cross_terms(storage, model)
    # overwriting potential couplings with explicit given
    setup_cross_terms!(storage, model, couplings)
    setup_linearized_system!(storage, model)
    align_equations_to_linearized_system!(storage, model)
    align_cross_terms_to_linearized_system!(storage, model)
    return storage
end

function setup_cross_terms!(storage, model::MultiModel, couplings)#::ModelCoupling)
    crossd = JutulStorage()
    models = model.models
    debugstr = "Determining cross-terms\n"
    for coupling in couplings
        sources = Dict{Symbol, Any}()
        target = coupling.target[:model]
        source = coupling.source[:model]
        def_target_eq = coupling.target[:equation]
        def_source_eq = coupling.target[:equation]
        intersection = coupling.intersection
        issym = coupling.issym
        crosstype = coupling.crosstype
        @assert !(target == source)
        tmpstr = "$source → $target:\n"
        target_model = models[target]
        source_model = models[source]

        target_eq = storage[target][:equations][def_target_eq]
        ct = setup_cross_term(target_eq,
                              target_model,
                              source_model,
                              target,
                              source,
                              intersection,
                              crosstype;transpose = false)

        @assert !isnothing(ct)
        if !haskey(storage[:cross_terms][target],source)
            setindex!(storage[:cross_terms][target], Dict(def_source_eq => ct), source)
        else
            if !haskey(storage[:cross_terms][target][source], def_source_eq)
                setindex!(storage[:cross_terms][target][source],ct, def_source_eq)
            else
                storage[:cross_terms][target][source][def_target_eq] = ct 
            end
        end
        
        if(issym)
            source_eq = storage[target][:equations][def_source_eq]
            cs = setup_cross_term(source_eq,
                              source_model,
                              target_model,
                              source,
                              target,
                              intersection,
                              crosstype;
                              transpose = true)
            if !haskey(storage[:cross_terms][source],target)
                setindex!(storage[:cross_terms][source], Dict(def_source_eq => cs), target)
            else
                if !haskey(storage[:cross_terms][source][target], def_source_eq)
                    setindex!(storage[:cross_terms][source][target],cs, def_source_eq)
                else
                    storage[:cross_terms][source][target][def_source_eq] = cs 
                end
            end                  
            #storage[:cross_terms][source][target][def_source_eq] = cs
        end
    end
    #storage[:cross_terms] = crossd
end

function setup_cross_terms(storage, model::MultiModel)
    crossd = JutulStorage()
    models = model.models
    debugstr = "Determining cross-terms\n"
    sms = submodel_symbols(model)
    for target in sms
        sources = Dict{Symbol, Any}()
        for source in sms
            if target == source
                continue
            end
            tmpstr = "$source → $target:\n"

            target_model = models[target]
            source_model = models[source]
            d = Dict{Symbol, Any}()
            found = false
            for (key, eq) in storage[target][:equations]
                ct = declare_cross_term(eq, target_model, source_model, target = target, source = source)
                if !isnothing(ct)
                    found = true
                    tmpstr *= String(key)*": Cross-term found.\n"
                    d[key] = ct
                end
            end
            if found
                sources[source] = d
                debugstr *= tmpstr
            end
        end
        crossd[target] = sources
    end
    storage[:cross_terms] = crossd
    @debug debugstr
end


function transpose_intersection(intersection)
    target, source, target_entity, source_entity = intersection
    (source, target, source_entity, target_entity)
end


function declare_cross_term(eq::JutulEquation, target_model, source_model; target = nothing, source = nothing)
    target_entity = associated_entity(eq)
    intersection = get_model_intersection(target_entity, target_model, source_model, target, source)
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
    model_keys = submodel_symbols(model)
    ndofs = model.number_of_degrees_of_freedom
    lsys = storage[:LinearizedSystem]
    if has_groups(model)
        ng = number_of_groups(model)
        groups = model.groups
        for g in 1:ng
            J = lsys[g, g].jac
            subs = groups .== g
            align_equations_subgroup!(storage, models, model_keys[subs], ndofs, J, equation_offset, variable_offset)
        end
    else
        J = lsys.jac
        align_equations_subgroup!(storage, models, model_keys, ndofs, J, equation_offset, variable_offset)
    end
end

function align_equations_subgroup!(storage, models, model_keys, ndofs, J, equation_offset, variable_offset)
    for key in model_keys
        submodel = models[key]
        eqs = storage[key][:equations]
        nrow_end = align_equations_to_jacobian!(eqs, J, submodel, equation_offset = equation_offset, variable_offset = variable_offset)
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
    model_keys = submodel_symbols(model)
    ndofs = model.number_of_degrees_of_freedom

    lsys = storage[:LinearizedSystem]
    if has_groups(model)
        ng = number_of_groups(model)
        groups = model.groups
        for target_g in 1:ng
            t_subs = groups .== target_g
            target_keys = model_keys[t_subs]
            for source_g in 1:ng
                s_subs = groups .== source_g
                source_keys = model_keys[s_subs]
                ls = lsys[target_g, source_g]
                align_crossterms_subgroup!(storage, models, target_keys, source_keys, ndofs, ls, equation_offset, variable_offset)
            end
        end
    else
        align_crossterms_subgroup!(storage, models, model_keys, model_keys, ndofs, lsys, equation_offset, variable_offset)
    end
end

function align_crossterms_subgroup!(storage, models, target_keys, source_keys, ndofs, lsys, equation_offset, variable_offset)
    base_variable_offset = variable_offset
    # Iterate over targets (= rows)
    for target in target_keys
        target_model = models[target]
        variable_offset = base_variable_offset
        # Iterate over sources (= columns)
        for source in source_keys
            source_model = models[source]
            if source != target
                ct = cross_term_pair(storage, source, target)
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

function align_cross_terms_to_linearized_system!(crossterms, equations, lsys, target::JutulModel, source::JutulModel; equation_offset = 0, variable_offset = 0)
    for ekey in keys(equations)
        eq = equations[ekey]
        if !isnothing(crossterms) && haskey(crossterms, ekey)
            ct = crossterms[ekey]
            if !isnothing(ct)
                align_to_jacobian!(ct, lsys, target, source, equation_offset = equation_offset, variable_offset = variable_offset)
            end
        end
        equation_offset += number_of_equations(target, eq)
    end
    return equation_offset
end

function get_sparse_arguments(storage, model::MultiModel, target::Symbol, source::Symbol, context)
    models = model.models
    target_model = models[target]
    source_model = models[source]
    layout = matrix_layout(context)

    if target == source
        # These are the main diagonal blocks each model knows how to produce themselves
        sarg = get_sparse_arguments(storage[target], target_model)
    else
        # Source differs from target. We need to get sparsity from cross model terms.
        T = index_type(context)
        I = Vector{Vector{T}}()
        J = Vector{Vector{T}}()
        ncols = number_of_degrees_of_freedom(source_model)
        # Loop over target equations and get the sparsity of the sources for each equation - with
        # derivative positions that correspond to that of the source
        equations = storage[target][:equations]
        cross_terms = cross_term_pair(storage, source, target)
        equation_offset = 0
        for (key, eq) in equations
            if !isnothing(cross_terms) && haskey(cross_terms, key)
                x = cross_terms[key]
                if !isnothing(x)
                    variable_offset = 0
                    for u in get_primary_variable_ordered_entities(source_model)
                        S = declare_sparsity(target_model, source_model, x, u, layout)
                        if !isnothing(S)
                            push!(I, S.I .+ equation_offset)
                            push!(J, S.J .+ variable_offset)
                        end
                        variable_offset += number_of_degrees_of_freedom(source_model, u)
                    end
                end
            end
            equation_offset += number_of_equations(target_model, eq)
        end
        I = vcat(I...)
        J = vcat(J...)
        sarg = SparsePattern(I, J, equation_offset, ncols, layout)
    end
    return sarg
end

function get_sparse_arguments(storage, model::MultiModel, targets::Vector{Symbol}, sources::Vector{Symbol}, context)
    I = []
    J = []
    outstr = "Determining sparse pattern of $(length(targets))×$(length(sources)) models:\n"
    equation_offset = 0
    variable_offset = 0
    function treat_block_size(bz, bz_new)
        if !isnothing(bz)
            @assert bz == bz_new
        end
        return bz_new
    end
    function finalize_block_size(bz)
        if isnothing(bz)
            bz = 1
        end
        return bz
    end
    bz_n = nothing
    bz_m = nothing
    for target in targets
        variable_offset = 0
        n = 0
        for source in sources
            sarg = get_sparse_arguments(storage, model, target, source, context)
            i, j, n, m = ijnm(sarg)
            bz_n = treat_block_size(bz_n, sarg.block_n)
            bz_m = treat_block_size(bz_m, sarg.block_m)
            if length(i) > 0
                push!(I, i .+ equation_offset)
                push!(J, j .+ variable_offset)
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
    bz_n = finalize_block_size(bz_n)
    bz_m = finalize_block_size(bz_m)
    return SparsePattern(I, J, equation_offset, variable_offset, matrix_layout(context), bz_n, bz_m)
end

function setup_linearized_system!(storage, model::MultiModel)
    models = model.models
    context = model.context

    candidates = [i for i in submodel_symbols(model)]
    if has_groups(model)
        ndof = values(model.number_of_degrees_of_freedom)
        n = sum(ndof)
        groups = model.groups
        ng = number_of_groups(model)

        # We have multiple groups. Store as Matrix of linearized systems
        F = float_type(context)
        r = zeros(F, n)
        dx = zeros(F, n)

        subsystems = Matrix{LinearizedType}(undef, ng, ng)
        has_context = !isnothing(context)

        block_sizes = zeros(ng)
        # Groups system with respect to themselves
        base_pos = 0
        for dpos in 1:ng
            local_models = groups .== dpos
            local_size = sum(ndof[local_models])
            t = candidates[local_models]
            ctx = models[t[1]].context
            layout = matrix_layout(ctx)
            sparse_arg = get_sparse_arguments(storage, model, t, t, ctx)

            block_sizes[dpos] = sparse_arg.block_n
            global_subs = (base_pos+1):(base_pos+local_size)
            r_i = view(r, global_subs)
            dx_i = view(dx, global_subs)
            subsystems[dpos, dpos] = LinearizedSystem(sparse_arg, ctx, layout, dx = dx_i, r = r_i)
            base_pos += local_size
        end
        # Off diagonal groups (cross-group connections)
        base_pos = 0
        for rowg in 1:ng
            local_models = groups .== rowg
            local_size = sum(ndof[local_models])
            t = candidates[local_models]
            row_context = models[first(t)].context
            row_layout = matrix_layout(row_context)
            for colg in 1:ng
                if rowg == colg
                    continue
                end
                s = candidates[groups .== colg]
                col_context = models[first(s)].context
                col_layout = matrix_layout(col_context)

                if has_context
                    ctx = context
                else
                    ctx = row_context
                end
                layout = matrix_layout(ctx)
                sparse_arg = get_sparse_arguments(storage, model, t, s, ctx)
                bz_pair = (block_sizes[rowg], block_sizes[colg])
                subsystems[rowg, colg] = LinearizedBlock(sparse_arg, ctx, layout, row_layout, col_layout, bz_pair)
            end
            base_pos += local_size
        end
        lsys = MultiLinearizedSystem(subsystems, context, matrix_layout(context), r = r, dx = dx, reduction = model.reduction)
    else
        # All Jacobians are grouped together and we assemble as a single linearized system
        if isnothing(context)
            context = models[1].context
        end
        layout = matrix_layout(context)
        sparse_arg = get_sparse_arguments(storage, model, candidates, candidates, context)
        lsys = LinearizedSystem(sparse_arg, context, layout)
    end
    storage[:LinearizedSystem] = lsys
end

function initialize_storage!(storage, model::MultiModel; kwarg...)
    for key in submodels_symbols(model)
        initialize_storage!(storage[key], model.models[key]; kwarg...)
    end
end

function update_equations!(storage, model::MultiModel, dt)
    @timeit "model equations" submodels_storage_apply!(storage, model, update_equations!, dt)
end

function update_equations_and_apply_forces!(storage::MultiModel, model, dt, forces; time = NaN)
    # First update all equations
    @timeit "equations" update_equations!(storage, model, dt)
    # Then update the cross terms
    @timeit "crossterm update" update_cross_terms!(storage, model::MultiModel, dt)
    # Apply forces
    @timeit "forces" apply_forces!(storage, model, dt, forces; time = time)
    # Apply forces to cross-terms
    @timeit "crossterm forces" apply_forces_to_cross_terms!(storage, model, dt, forces; time = time)
    # Finally apply cross terms to the equations
    @timeit "crossterm apply" apply_cross_terms!(storage, model::MultiModel, dt)
end

function update_cross_terms!(storage, model::MultiModel, dt; targets = submodel_symbols(model), sources = targets)
    for target in targets
        for source in intersect(target_cross_term_keys(storage, target), sources)
            @timeit "$source→$target" begin
                cross_terms = cross_term_pair(storage, source, target)
                update_cross_terms_for_pair!(cross_terms, storage, model, source, target, dt)
            end
        end
    end
end

update_cross_terms_for_pair!(cross_terms::Nothing, storage, model, source::Symbol, target::Symbol, dt) = nothing

function update_cross_terms_for_pair!(cross_terms, storage, model, source::Symbol, target::Symbol, dt)
    storage_t, storage_s = get_submodel_storage(storage, target, source)
    model_t, model_s = get_submodels(model, target, source)

    eqs = storage_t.equations
    for (ekey, ct) in pairs(cross_terms)
        if !isnothing(ct)
            # @info "$source -> $target: $ekey"
            update_cross_term!(ct, eqs[ekey], storage_t, storage_s, model_t, model_s, target, source, dt)
        end
    end
end

function apply_cross_terms!(storage, model::MultiModel, dt; targets = submodel_symbols(model), sources = targets)
    for target in targets
        for source in intersect(target_cross_term_keys(storage, target), sources)
            @timeit "$source→$target" begin
                cross_terms = cross_term_pair(storage, source, target)
                apply_cross_terms_for_pair!(cross_terms, storage, model, source, target, dt)
            end
        end
    end
end

apply_cross_terms_for_pair!(cross_terms::Nothing, storage, model, source::Symbol, target::Symbol, dt) = nothing

function apply_cross_terms_for_pair!(cross_terms, storage, model, source::Symbol, target::Symbol, dt)
    storage_t, = get_submodel_storage(storage, target)
    model_t, model_s = get_submodels(model, target, source)

    eqs = storage_t[:equations]
    for (ekey, ct) in pairs(cross_terms)
        if !isnothing(ct)
            apply_cross_term!(eqs[ekey], ct, model_t, model_s, dt)
        end
    end
end


function update_linearized_system!(storage, model::MultiModel; equation_offset = 0, targets = submodel_symbols(model), sources = targets)
    @assert equation_offset == 0 "The multimodel version assumes offset == 0, was $offset"
    # Update diagonal blocks (model with respect to itself)
    @timeit "models" update_diagonal_blocks!(storage, model, targets)
    # Then, update cross terms (models' impact on other models)
    @timeit "cross-model" update_offdiagonal_blocks!(storage, model, targets, sources)
end

function update_diagonal_blocks!(storage, model::MultiModel, targets)
    lsys = storage.LinearizedSystem
    model_keys = submodel_symbols(model)
    if has_groups(model)
        ng = number_of_groups(model)
        groups = model.groups
        for g in 1:ng
            lsys_g = lsys[g, g]
            subs = groups .== g
            group_targets = model_keys[subs]
            group_keys = intersect(group_targets, targets)
            offsets = get_submodel_degree_of_freedom_offsets(model, g)
            update_main_linearized_system_subgroup!(storage, model, group_keys, offsets, lsys_g)
        end
    else
        offsets = get_submodel_degree_of_freedom_offsets(model)
        update_main_linearized_system_subgroup!(storage, model, targets, offsets, lsys)
    end
end

function update_main_linearized_system_subgroup!(storage, model, model_keys, offsets, lsys)
    for (index, key) in enumerate(model_keys)
        m = model.models[key]
        s = storage[key]
        eqs = s.equations
        update_linearized_system!(lsys, eqs, m; equation_offset = offsets[index])
    end
end

function update_offdiagonal_blocks!(storage, model, targets, sources)
    linearized_system = storage.LinearizedSystem
    for target in targets
        for source in intersect(target_cross_term_keys(storage, target), sources)
            cross_terms = cross_term_pair(storage, source, target)
            lsys = get_linearized_system_model_pair(storage, model, source, target, linearized_system)
            update_linearized_system_crossterms!(lsys, cross_terms, storage, model, source, target)
        end
    end
end


function update_linearized_system_crossterms!(lsys, cross_terms, storage, model::MultiModel, source, target)
    # storage_t, = get_submodel_storage(storage, target)
    model_t, model_s = get_submodels(model, target, source)

    for ekey in keys(cross_terms)
        ct = cross_terms[ekey]
        nz = nonzeros(lsys.jac)
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

function check_convergence(storage, model::MultiModel; tol = nothing, extra_out = false, kwarg...)
    converged = true
    err = 0
    offset = 0
    lsys = storage.LinearizedSystem
    errors = OrderedDict()
    for (i, key) in enumerate(submodels_symbols(model))
        if has_groups(model) && i > 1
            if model.groups[i] != model.groups[i-1]
                offset = 0
            end
        end
        s = storage[key]
        m = model.models[key]
        eqs = s.equations
        ls = get_linearized_system_submodel(storage, model, key, lsys)
        conv, e, errors[key], = check_convergence(ls, eqs, s, m; offset = offset, extra_out = true, tol = tol, kwarg...)
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

function update_primary_variables!(storage, model::MultiModel; kwarg...)
    lsys = storage.LinearizedSystem
    dx = lsys.dx_buffer
    models = model.models

    offset = 0
    model_keys = submodel_symbols(model)
    for (i, key) in enumerate(model_keys)
        m = models[key]
        s = storage[key]
        ndof = number_of_degrees_of_freedom(m)
        dx_v = view(dx, (offset+1):(offset+ndof))
        if isa(matrix_layout(m.context), BlockMajorLayout)
            bz = block_size(lsys[i, i])
            dx_v = reshape(dx_v, bz, :)
        end
        update_primary_variables!(s.state, dx_v, m; kwarg...)
        offset += ndof
    end
end

function reset_to_previous_state!(storage, model::MultiModel)
    submodels_storage_apply!(storage, model, reset_to_previous_state!)
end

function update_after_step!(storage, model::MultiModel, dt, forces; targets = submodels_symbols(model))
    for key in targets
        update_after_step!(storage[key], model.models[key], dt, forces[key])
    end
end

function update_before_step!(storage, model::MultiModel, dt, forces; targets = submodels_symbols(model))
    for key in targets
        update_before_step!(storage[key], model.models[key], dt, forces[key])
    end
end

function apply_forces!(storage, model::MultiModel, dt, forces; time = NaN, targets = submodels_symbols(model))
    for key in targets
        apply_forces!(storage[key], model.models[key], dt, forces[key]; time = time)
    end
end

function apply_forces_to_cross_terms!(storage, model::MultiModel, dt, forces; time = NaN, targets = submodels_symbols(model), sources = targets)
    for target in targets
        # Target: Model where force has impact
        force = forces[target]
        if isnothing(force)
            continue
        end
        for source in intersect(target_cross_term_keys(storage, target), sources)
            for to_target = [true, false]
                apply_force_to_cross_terms!(storage, model, source, target, force, dt, time; to_target = to_target)
            end
        end
    end
end

function apply_force_to_cross_terms!(storage, model, source, target, force, dt, time; to_target = true)
    storage_t, storage_s = get_submodel_storage(storage, target, source)
    model_t, model_s = get_submodels(model, target, source)
    eqs = storage_t.equations
    # Target matches where the force is assigned.
    if to_target
        # Equation comes from target model and we are looking at the cross term for that model.
        cross_terms = cross_term_pair(storage, source, target)
        fn = apply_force_to_cross_term_target!
    else
        # Equation comes from target, but we are looking at the cross term for the source model
        cross_terms = cross_term_pair(storage, source, target)
        fn = apply_force_to_cross_term_source!
    end
    for (ekey, ct) in pairs(cross_terms)
        if !isnothing(ct)
            fn(ct, eqs[ekey], storage_t, storage_s, model_t, model_s, source, target, force, dt, time)
        end
    end
end

apply_force_to_cross_term_target!(ct, equation, storage_t, storage_s, model_t, model_s, source, target, force, dt, time) = nothing
apply_force_to_cross_term_source!(ct, equation, storage_t, storage_s, model_t, model_s, source, target, force, dt, time) = nothing

function apply_boundary_conditions!(storage, model::MultiModel; targets = submodels_symbols(model))
    for key in targets
        apply_boundary_conditions!(storage[key], storage[key].parameters, model.models[key])
    end
end


function submodels_storage_apply!(storage, model, f!, arg...)
    for key in submodels_symbols(model)
        f!(storage[key], model.models[key], arg...)
    end
end

function get_output_state(storage, model::MultiModel)
    out = Dict{Symbol, Any}()
    models = model.models
    for key in submodel_symbols(model)
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

function get_convergence_table(model::MultiModel, errors)
    get_convergence_table(submodels_symbols(model), errors)
end
