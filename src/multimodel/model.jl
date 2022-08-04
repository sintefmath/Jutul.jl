function Base.show(io::IO, t::MIME"text/plain", model::MultiModel)
    submodels = model.models
    cross_terms = model.cross_terms

    if get(io, :compact, false)
    else
    end
    println(io, "MultiModel with $(length(submodels)) models and $(length(cross_terms)) cross-terms.")
    println(io , "\n  models:")
    for (i, key) in enumerate(keys(submodels))
        m = submodels[key]
        s = m.system
        if hasproperty(m.domain, :grid)
            g = m.domain.grid
        else
            g = typeof(m.domain)
        end
        println(io, "    $i) $key\n       $(s) ∈ $g")

    end
    if length(cross_terms) > 0
        println(io , "\n  cross_terms:")
        for (i, ct_s) in enumerate(cross_terms)
            (; cross_term, target, source) = ct_s
            t = typeof(cross_term)
            if has_symmetry(cross_term)
                arrow = "<->"
            else
                arrow = " ->"
            end
            println(io, "    $i) $source $arrow $target")
            println(io, "       $t")
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

function replace_variables!(model::MultiModel; kwarg...)
    for m in model.models
        replace_variables!(m, throw = false; kwarg...)
    end
    return model
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
    setup_cross_terms_storage!(storage, model)
    setup_linearized_system!(storage, model)
    align_equations_to_linearized_system!(storage, model)
    align_cross_terms_to_linearized_system!(storage, model)
    return storage
end

function transpose_intersection(intersection)
    target, source, target_entity, source_entity = intersection
    (source, target, source_entity, target_entity)
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
        eqs_s = storage[key][:equations]
        eqs = submodel.equations
        nrow_end = align_equations_to_jacobian!(eqs_s, eqs, J, submodel, equation_offset = equation_offset, variable_offset = variable_offset)
        nrow = nrow_end - equation_offset
        ndof = ndofs[key]
        @assert nrow == ndof "Submodels must have equal number of equations and degrees of freedom. Found $nrow equations and $ndof variables for submodel $key"
        equation_offset += ndof
        variable_offset += ndof # Assuming that each model by itself forms a well-posed, square Jacobian...
    end
end

function local_group_offset(keys, target_key, ndofs)
    offset = 0
    for k in keys
        if k == target_key
            return offset
        end
        offset += ndofs[k]
    end
    error("Should not happen")
end

function get_equation_offset(model::SimulationModel, eq_label::Symbol)
    offset = 0
    for k in keys(model.equations)
        if k == eq_label
            return offset
        else
            offset += number_of_equations(model, model.equations[k])
        end
    end
    error("Did not find equation")
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
        equations = target_model.equations
        cross_terms, cross_term_storage = cross_term_pair(model, storage, source, target, true)

        for (ctp, s) in zip(cross_terms, cross_term_storage)
            ct = ctp.cross_term
            transp = ctp.source == target
            eq_label = ctp.equation
            if transp
                # The filter found a cross term with symmetry, that has "target" as the source. We then need to add it here,
                # reversing most of the inputs
                @assert ctp.source == target
                @assert has_symmetry(ctp)
                ct_storage = s.target
                entities = s.source_entities
            else
                ct_storage = s.source
                entities = s.target_entities
            end
            add_sparse_local!(I, J, ct, eq_label, ct_storage, target_model, source_model, entities, layout)
        end
        I = vec(vcat(I...))
        J = vec(vcat(J...))
        nrows = number_of_rows(target_model, layout)
        sarg = SparsePattern(I, J, nrows, ncols, layout)
    end
    return sarg
end

function number_of_rows(model, layout::Union{EquationMajorLayout, UnitMajorLayout})
    n = 0
    for eq in values(model.equations)
        n += number_of_equations(model, eq)
    end
    return n
end

function number_of_rows(model, layout::BlockMajorLayout)
    n = 0
    for eq in values(model.equations)
        n += number_of_entities(model, eq)
    end
    return n
end

function add_sparse_local!(I, J, x, eq_label, s, target_model, source_model, ind, layout::EquationMajorLayout)
    eq = target_model.equations[eq_label]
    target_e = associated_entity(eq)
    entities = get_primary_variable_ordered_entities(source_model)
    equation_offset = get_equation_offset(target_model, eq_label)
    variable_offset = 0
    for (i, source_e) in enumerate(entities)
        S = declare_sparsity(target_model, source_model, eq, x, s, ind, target_e, source_e, layout)
        if !isnothing(S)
            rows = S.I
            cols = S.J
            push!(I, rows .+ equation_offset)
            push!(J, cols .+ variable_offset)
        end
        variable_offset += number_of_degrees_of_freedom(source_model, source_e)
    end
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

function update_equations_and_apply_forces!(storage, model::MultiModel, dt, forces; time = NaN)
    # First update all equations
    @timeit "equations" update_equations!(storage, model, dt)
    # Then update the cross terms
    @timeit "crossterm update" update_cross_terms!(storage, model, dt)
    # Apply forces
    @timeit "forces" apply_forces!(storage, model, dt, forces; time = time)
    # Apply forces to cross-terms
    @timeit "crossterm forces" apply_forces_to_cross_terms!(storage, model, dt, forces; time = time)
end

function update_cross_terms!(storage, model::MultiModel, dt; targets = submodel_symbols(model), sources = targets)
    models = model.models
    for (ctp, ct_s) in zip(model.cross_terms, storage.cross_terms)
        target = ctp.target
        source = ctp.source
        if target in targets && source in sources
            ct = ctp.cross_term
            model_t = models[target]
            eq = model_t.equations[ctp.equation]
            update_cross_term!(ct_s, ct, eq, storage[target], storage[source], model_t, models[source], dt)
        end
    end
end

function update_cross_term!(ct_s, ct::CrossTerm, eq, storage_t, storage_s, model_t, model_s, dt)
    state_t = storage_t.state
    state0_t = storage_t.state0

    state_s = storage_s.state
    state0_s = storage_s.state0

    param_t = storage_s.parameters
    param_s = storage_s.parameters

    c = first(ct_s.target)
    for i in 1:number_of_entities(c)
        prepare_cross_term_in_entity!(i, state_t, state0_t, state_s, state0_s, model_t, model_s, param_t, param_s, ct, eq, dt)
    end

    for (k, cache) in pairs(ct_s.target)
        update_cross_term_for_entity!(cache, ct, eq, state_t, state0_t, state_s, state0_s, model_t, model_s, param_t, param_s, dt, true)
    end

    for (k, cache) in pairs(ct_s.source)
        update_cross_term_for_entity!(cache, ct, eq, state_t, state0_t, state_s, state0_s, model_t, model_s, param_s, param_t, dt, false)
    end
end

function update_cross_term_for_entity!(cache, ct, eq, state_t, state0_t, state_s, state0_s, model_t, model_s, param_t, param_s, dt, is_target)
    v = cache.entries
    vars = cache.variables
    Tv = eltype(v)
    for i in 1:number_of_entities(cache)
        ldisc = local_discretization(ct, i)
        for j in vrange(cache, i)
            v_i = @views v[:, j]
            var = vars[j]
            if is_target
                f_t = (x) -> local_ad(x, var, Tv)
                f_s = (x) -> as_value(x)
            else
                f_t = (x) -> as_value(x)
                f_s = (x) -> local_ad(x, var, Tv)
            end
            update_cross_term_in_entity!(v_i, i, f_t(state_t), f_t(state0_t), f_s(state_s), f_s(state0_s), model_t, model_s, param_t, param_s, ct, eq, dt, ldisc)
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

function setup_state(model::MultiModel, initializers)
    state = Dict()
    for key in submodels_symbols(model)
        m = model.models[key]
        init = initializers[key]
        state[key] = setup_state(m, init)
    end
    return state
end

function setup_state(model::MultiModel; kwarg...)
    init = Dict{Symbol, Any}()
    # Set up empty initializers first
    for k in submodels_symbols(model)
        init[k] = nothing
    end
    # Then unpack kwarg as separate initializers
    for (k, v) in kwarg
        @assert haskey(model.models, k) "$k not found in models" keys(model.models)
        init[k] = v
    end
    return setup_state(model, init)
end

function setup_parameters(model::MultiModel)
    p = Dict()
    for key in submodels_symbols(model)
        m = model.models[key]
        p[key] = setup_parameters(m)
    end
    return p
end

function setup_forces(model::MultiModel; kwarg...)
    models = model.models
    forces = Dict{Symbol, Any}()
    for k in submodels_symbols(model)
        forces[k] = setup_forces(models[k])
    end
    for (k, v) in kwarg
        @assert haskey(models, k) "$k not found in models" keys(model.models)
        forces[k] = v
    end
    return forces
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
        eqs = m.equations
        eqs_s = s.equations
        ls = get_linearized_system_submodel(storage, model, key, lsys)
        conv, e, errors[key], = check_convergence(ls, eqs, eqs_s, s, m; offset = offset, extra_out = true, tol = tol, kwarg...)
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

function reset_previous_state!(storage, model::MultiModel, state0)
    for key in submodels_symbols(model)
        reset_previous_state!(storage[key], model.models[key], state0[key])
    end
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
    # @info forces
    # @warn "Not reimplemented yet"
    return
    for target in targets
        # Target: Model where force has impact
        force = forces[target]
        if isnothing(force)
            continue
        end
        ct_t = cross_term(storage, target)
        for source in intersect(target_cross_term_keys(storage, target), sources)
            for f in force
                for to_target = [true, false]
                    apply_force_to_cross_terms!(storage, ct_t, model, source, target, f, dt, time; to_target = to_target)
                end
            end
        end
    end
end

function apply_force_to_cross_terms!(storage, ct_t, model, source, target, force, dt, time; to_target = true)
    storage_t, storage_s = get_submodel_storage(storage, target, source)
    model_t, model_s = get_submodels(model, target, source)
    # Target matches where the force is assigned.
    if to_target
        # Equation comes from target model and we are looking at the cross term for that model.
        cross_terms = cross_term_pair(storage, source, target)
        fn = apply_force_to_cross_term_target!
    else
        # Equation comes from target, but we are looking at the cross term for the source model
        cross_terms = cross_terms_if_present(ct_t, source)
        fn = apply_force_to_cross_term_source!
    end
    for (ekey, ct) in pairs(cross_terms)
        if !isnothing(ct)
            fn(ct, storage_t, storage_s, model_t, model_s, source, target, force, dt, time)
        end
    end
end

apply_force_to_cross_term_target!(ct, storage_t, storage_s, model_t, model_s, source, target, force, dt, time) = nothing
apply_force_to_cross_term_source!(ct, storage_t, storage_s, model_t, model_s, source, target, force, dt, time) = nothing

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

get_submodel_storage(storage, k) = (storage[k]::JutulStorage, )

function get_submodels(model, arg...)
    map((x) -> model.models[x], arg)
end

function get_convergence_table(model::MultiModel, errors)
    get_convergence_table(submodels_symbols(model), errors)
end
