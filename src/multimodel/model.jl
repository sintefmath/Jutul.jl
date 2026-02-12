function Base.show(io::IO, t::MIME"text/plain", model::MultiModel)
    submodels = model.models
    cross_terms = model.cross_terms

    if get(io, :compact, false)
    else
    end
    ndof = number_of_degrees_of_freedom(model)
    neq = number_of_equations(model)
    nprm = number_of_parameters(model)
    println(io, "MultiModel with $(length(submodels)) models and $(length(cross_terms)) cross-terms. $neq equations, $ndof degrees of freedom and $nprm parameters.")
    println(io , "\n  models:")
    for (i, key) in enumerate(keys(submodels))
        m = submodels[key]
        s = m.system
        ndofi = number_of_degrees_of_freedom(m)
        neqi = number_of_equations(m)
        g = physical_representation(m.domain)
        println(io, "    $i) $key ($(neqi)x$ndofi)\n       $(s)\n       ∈ $g")
    end
    if length(cross_terms) > 0
        println(io , "\n  cross_terms:")
        for (i, ct_s) in enumerate(cross_terms)
            (; cross_term, target, source, target_equation, source_equation) = ct_s
            equation = target_equation
            print_t = Base.typename(typeof(cross_term)).wrapper
            if has_symmetry(cross_term)
                arrow = "<->"
                eqstr = "Eqs: $target_equation <-> $source_equation"
            else
                arrow = " ->"
                eqstr = "Eq: $target_equation"
            end
            println(io, "    $i) $source $arrow $target ($eqstr)")
            println(io, "       $print_t")
        end
    end
    if multi_model_is_specialized(model)
        opt = "runtime"
    else
        opt = "compilation"
    end
    println(io, "\nModel storage will be optimized for $opt performance.")
end


function number_of_models(model::MultiModel)
    return length(model.models)
end

function number_of_values(model::MultiModel, type = :primary)
    return sum(m -> number_of_values(m, type), values(model.models))
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

@inline function submodels_symbols(model::MultiModel)
    return keys(model.models)
end

function setup_state!(state, model::MultiModel, init_values)
    error("Mutating version of setup_state not supported for multimodel.")
end

function setup_storage!(storage, model::MultiModel;
        state0 = setup_state(model),
        parameters = setup_parameters(model),
        setup_linearized_system = true,
        state0_ad = false,
        state_ad = true,
        kwarg...
    )
    state0_ref = JutulStorage()
    state_ref = JutulStorage()
    use_internal_ad = state0_ad || state_ad
    @tic "model" for key in submodels_symbols(model)
        m = model[key]
        @tic "$key" begin
            storage[key] = setup_storage(m; state0 = state0[key],
                                        parameters = parameters[key],
                                        setup_linearized_system = false,
                                        setup_equations = false,
                                        state0_ad = state0_ad,
                                        state_ad = state_ad,
                                        tag = submodel_ad_tag(model, key),
                                        kwarg...)
        end
        # Add outer references to state that matches the nested structure
        state_ref[key] = storage[key][:state]
        state0_ref[key] = storage[key][:state0]
    end
    storage[:state] = state_ref
    storage[:state0] = state0_ref
    @tic "cross terms" setup_cross_terms_storage!(storage, model, ad = use_internal_ad)
    @tic "equations" for key in submodels_symbols(model)
        m = model[key]
        @tic "$key" begin
            ct_i = extra_cross_term_sparsity(model, storage, key, true)
            storage[key][:equations] = setup_storage_equations(storage[key], m,
                ad = use_internal_ad,
                extra_sparsity = ct_i,
                tag = submodel_ad_tag(model, key)
            )
        end
    end
    if setup_linearized_system
        @tic "linear system" begin
            @tic "setup" setup_linearized_system!(storage, model)
            @tic "alignment" align_equations_to_linearized_system!(storage, model)
            @tic "alignment cross terms" align_cross_terms_to_linearized_system!(storage, model)
            @tic "views" setup_equations_and_primary_variable_views!(storage, model)
        end
    end
    setup_multimodel_maps!(storage, model)
    return storage
end

mutable struct MutableWrapper
    maps
end

function setup_multimodel_maps!(storage, model)
    groups = model.groups
    if isnothing(groups)
        offset_map = get_submodel_offsets(model, nothing)
    else
        offset_map = map(g -> get_submodel_offsets(model, g), unique(groups))
    end
    storage[:multi_model_maps] = (offset_map = offset_map, );
    storage[:eq_maps] = MutableWrapper(nothing)
end

function setup_equations_and_primary_variable_views!(storage, model::MultiModel, lsys)
    mkeys = submodels_symbols(model)
    groups = model.groups
    no_groups = isnothing(groups) || lsys isa NamedTuple
    if no_groups
        groups = ones(Int, length(mkeys))
    end
    ug = unique(groups)

    for group in ug
        if no_groups
            # Hack for helper sim
            lsys_g = lsys
        else
            lsys_g = lsys[group, group]
        end
        dx = lsys_g.dx_buffer
        if !ismissing(dx)
            dx = vec(dx)
        end
        r = lsys_g.r_buffer
        if !ismissing(r)
            r = vec(r)
        end
        eq_offset = 0
        var_offset = 0
        for (i, g) in enumerate(groups)
            if g != group
                continue
            end
            k = mkeys[i]
            submodel = model[k]
            neqs = number_of_equations(submodel)
            ndof = number_of_degrees_of_freedom(submodel)
            if ismissing(r)
                r_i = r
            else
                r_i = view(r, (eq_offset+1):(eq_offset+neqs))
            end
            if ismissing(dx)
                dx_i = dx
            else
                dx_i = view(dx, (var_offset+1):(var_offset+ndof))
            end
            storage[k][:views] = setup_equations_and_primary_variable_views(storage[k], submodel, r_i, dx_i)
            eq_offset += neqs
            var_offset += ndof
        end
    end
end

function specialize_simulator_storage(storage::JutulStorage, model::MultiModel, specialize)
    specialize_outer = multi_model_is_specialized(model)
    specialize = specialize || specialize_outer
    sym = submodels_symbols(model)
    for (k, v) in data(storage)
        if k in sym
            storage[k] = specialize_simulator_storage(v, model[k], specialize)
        elseif v isa JutulStorage
            storage[k] = specialize_simulator_storage(v, nothing, specialize)
        else
            storage[k] = convert_to_immutable_storage(v)
        end
    end
    ct = storage[:cross_terms]
    for i in eachindex(ct)
        ct[i] = specialize_simulator_storage(ct[i], nothing, specialize)
    end
    if specialize_outer
        storage = convert_to_immutable_storage(storage)
    end
    return storage
end

function transpose_intersection(intersection)
    target, source, target_entity, source_entity = intersection
    (source, target, source_entity, target_entity)
end

function align_equations_to_linearized_system!(storage, model::MultiModel; equation_offset = 0, variable_offset = 0)
    models = model.models
    model_keys = submodels_symbols(model)
    neqs = sub_number_of_equations(model)
    ndofs = sub_number_of_degrees_of_freedom(model)
    bz = sub_block_sizes(model)

    dims = (neqs, ndofs, bz)
    lsys = storage[:LinearizedSystem]
    if has_groups(model)
        ng = number_of_groups(model)
        groups = model.groups
        for g in 1:ng
            J = lsys[g, g].jac
            subs = groups .== g
            align_equations_subgroup!(storage, models, model_keys[subs], dims, J, equation_offset, variable_offset)
        end
    else
        J = lsys.jac
        align_equations_subgroup!(storage, models, model_keys, dims, J, equation_offset, variable_offset)
    end
end

function align_equations_subgroup!(storage, models, model_keys, dims, J, equation_offset, variable_offset)
    neqs, nvars, bz = dims
    for key in model_keys
        submodel = models[key]
        eqs_s = storage[key][:equations]
        eqs = submodel.equations
        align_equations_to_jacobian!(eqs_s, eqs, J, submodel, equation_offset = equation_offset, variable_offset = variable_offset)
        equation_offset += neqs[key]÷bz[key]
        variable_offset += nvars[key]÷bz[key]
    end
end

function local_group_offset(keys, target_key, ndofs)
    offset = 0
    for k in keys
        if k == target_key
            return offset
        end
        ndof = ndofs[k]
        offset += ndof
    end
    error("Should not happen")
end

function get_equation_offset(model::SimulationModel, eq_label::Pair, arg...)
    return get_equation_offset(model, last(eq_label), arg...)
end

function get_equation_offset(model::SimulationModel, eq_label::Symbol)
    offset = 0
    layout = matrix_layout(model.context)
    for k in keys(model.equations)
        if k == eq_label
            return offset
        else
            offset += number_of_equations(model, model.equations[k])
        end
    end
    error("Did not find equation $eq_label in $(keys(model.equations))")
end

function get_sparse_arguments(storage, model::MultiModel, target::Symbol, source::Symbol, row_context, col_context)
    models = model.models
    target_model = models[target]
    source_model = models[source]

    if target == source
        # These are the main diagonal blocks each model knows how to produce themselves
        sarg = get_sparse_arguments(storage[target], target_model)
    else
        row_layout = matrix_layout(row_context)
        col_layout = matrix_layout(col_context)

        row_layout = scalarize_layout(row_layout, col_layout)
        col_layout = scalarize_layout(col_layout, row_layout)
        has_blocks = col_layout == BlockMajorLayout()
        bz = 1
        if has_blocks
            @assert row_layout == col_layout
            # Assume that block layout uses a single entity, grab the only one with primaries
            prim_e = get_primary_variable_ordered_entities(target_model)
            some_entity = only(prim_e)
            bz = degrees_of_freedom_per_entity(target_model, some_entity)
        end

        # Source differs from target. We need to get sparsity from cross model terms.
        T = index_type(row_context)
        I = Vector{Vector{T}}()
        J = Vector{Vector{T}}()
        ncols = Int(number_of_degrees_of_freedom(source_model)/bz)
        # Loop over target equations and get the sparsity of the sources for each equation - with
        # derivative positions that correspond to that of the source
        cross_terms, cross_term_storage = cross_term_pair(model, storage, source, target, true)
        # TODO: Fix here?
        for (ctp, s) in zip(cross_terms, cross_term_storage)
            ct = ctp.cross_term
            transp = ctp.source == target
            if transp
                # The filter found a cross term with symmetry, that has "target"
                # as the source. We then need to add it here, reversing most of
                # the inputs
                @assert ctp.source == target
                @assert has_symmetry(ctp)
                eq_label = ctp.source_equation
                ct_storage = s.target
                entities = s.source_entities
            else
                eq_label = ctp.target_equation
                ct_storage = s.source
                entities = s.target_entities
            end
            add_sparse_local!(I, J, ct, eq_label, ct_storage, target_model, source_model, entities, row_layout, col_layout)
        end
        I = vec(vcat(I...))
        J = vec(vcat(J...))
        nrows = number_of_rows(target_model, row_layout)
        sarg = SparsePattern(I, J, nrows, ncols, row_layout, col_layout, bz, bz)
    end
    return sarg
end

function number_of_rows(model, layout::Union{EquationMajorLayout, EntityMajorLayout})
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

function add_sparse_local!(I, J, x, eq_label, s, target_model, source_model, ind, row_layout::ScalarLayout, col_layout::ScalarLayout; base_equation_offset = 0)
    eq = ct_equation(target_model, eq_label)
    target_e = associated_entity(eq)
    entities = get_primary_variable_ordered_entities(source_model)
    equation_offset = get_equation_offset(target_model, eq_label)
    variable_offset = 0
    row_is_eqn_major = row_layout isa EquationMajorLayout
    if row_is_eqn_major
        Bz = 1
        eq_inner_offset = 0
    else
        Ne = count_active_entities(target_model.domain, target_e)
        Bz = model_block_size(target_model)
        eq_inner_offset = (equation_offset - base_equation_offset) ÷ Ne
        equation_offset = base_equation_offset
    end

    for (i, source_e) in enumerate(entities)
        S = declare_sparsity(
            target_model, source_model,
            eq, x, s, ind,
            target_e, source_e,
            row_layout, col_layout;
            equation_offset = eq_inner_offset,
            block_size = Bz
        )
        if !isnothing(S) && length(S.I) > 0
            rows = S.I .+ equation_offset
            cols = S.J .+ variable_offset
            push!(I, rows)
            push!(J, cols)
        end
        variable_offset += number_of_degrees_of_freedom(source_model, source_e)
    end
end

function add_sparse_local!(I, J, x, eq_label, s, target_model, source_model, ind, row_layout::BlockMajorLayout, col_layout::BlockMajorLayout; base_equation_offset = 0)
    eq = ct_equation(target_model, eq_label)
    target_e = associated_entity(eq)
    entities = get_primary_variable_ordered_entities(source_model)
    bz = degrees_of_freedom_per_entity(target_model, only(entities))
    equation_offset = get_equation_offset(target_model, eq_label)
    variable_offset = 0
    for (i, source_e) in enumerate(entities)
        S = declare_sparsity(target_model, source_model, eq, x, s, ind, target_e, source_e, row_layout, col_layout)
        if !isnothing(S)
            rows = S.I
            cols = S.J
            push!(I, rows .+ equation_offset)
            push!(J, cols .+ variable_offset)
        end
        variable_offset += count_active_entities(source_model.domain, source_e)
    end
end

function get_sparse_arguments(storage, model::MultiModel, targets::Vector{Symbol}, sources::Vector{Symbol}, row_context, col_context)
    I = Int[]
    J = Int[]
    outstr = "Determining sparse pattern of $(length(targets))×$(length(sources)) models:\n"
    equation_offset = 0
    variable_offset = 0
    function treat_block_size(bz, bz_new)
        if !isnothing(bz)
            @assert bz == bz_new "Block sizes must match $bz != $bz_new"
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
            sarg = get_sparse_arguments(storage, model, target, source, row_context, col_context)
            i, j, n, m = ijnm(sarg)
            bz_n = treat_block_size(bz_n, sarg.block_n)
            bz_m = treat_block_size(bz_m, sarg.block_m)
            if length(i) > 0
                @assert maximum(i) <= n "I index exceeded $n for $source → $target (largest value: $(maximum(i))"
                @assert maximum(j) <= m "J index exceeded $m for $source → $target (largest value: $(maximum(j))"

                @assert minimum(i) >= 1 "I index was lower than 1 for $source → $target"
                @assert minimum(j) >= 1 "J index was lower than 1 for $source → $target"

                for ii in i
                    push!(I, ii + equation_offset)
                end
                for jj in j
                    push!(J, jj + variable_offset)
                end
            end
            outstr *= "$source → $target: $n rows and $m columns starting at $(equation_offset+1), $(variable_offset+1).\n"
            variable_offset += m
        end
        outstr *= "\n"
        equation_offset += n
    end
    @debug outstr
    bz_n = finalize_block_size(bz_n)
    bz_m = finalize_block_size(bz_m)
    return SparsePattern(I, J, equation_offset, variable_offset, matrix_layout(row_context), matrix_layout(col_context), bz_n, bz_m)
end

function setup_linearized_system!(storage, model::MultiModel)
    models = model.models
    context = model.context

    candidates = [i for i in submodels_symbols(model)]
    if has_groups(model)
        ndof = values(map(number_of_degrees_of_freedom, models))
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
            sparse_arg = get_sparse_arguments(storage, model, t, t, ctx, ctx)
            if represented_as_adjoint(layout)
                sparse_arg = sparse_arg'
            end

            block_sizes[dpos] = sparse_arg.block_n
            global_subs = (base_pos+1):(base_pos+local_size)
            r_i = view(r, global_subs)
            dx_i = view(dx, global_subs)
            subsystems[dpos, dpos] = LinearizedSystem(sparse_arg, ctx, layout, dx = dx_i, r = r_i)
            base_pos += local_size
        end
        # Off diagonal groups (cross-group connections)
        is_trans = has_context && represented_as_adjoint(matrix_layout(context))
        for rowg in 1:ng
            local_models = groups .== rowg
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
                sparse_pattern = get_sparse_arguments(storage, model, t, s, row_context, col_context)
                if represented_as_adjoint(row_layout)
                    sparse_pattern = sparse_pattern'
                end
                bz_pair = (block_sizes[rowg], block_sizes[colg])
                this_block = LinearizedBlock(sparse_pattern, bz_pair, row_layout, col_layout)
                if is_trans
                    subsystems[colg, rowg] = this_block
                else
                    subsystems[rowg, colg] = this_block
                end
            end
        end
        lsys = MultiLinearizedSystem(subsystems, context, matrix_layout(context), r = r, dx = dx, reduction = model.reduction)
    else
        # All Jacobians are grouped together and we assemble as a single linearized system
        if isnothing(context)
            context = models[1].context
        end
        layout = matrix_layout(context)
        sparse_pattern = get_sparse_arguments(storage, model, candidates, candidates, context, context)
        if represented_as_adjoint(matrix_layout(model.context))
            sparse_pattern = sparse_pattern'
        end
        lsys = LinearizedSystem(sparse_pattern, context, layout)
    end
    storage[:LinearizedSystem] = lsys
end

function initialize_storage!(storage, model::MultiModel; kwarg...)
    for key in submodels_symbols(model)
        initialize_storage!(storage[key], model.models[key]; kwarg...)
    end
end

function update_equations!(storage, model::MultiModel, dt; targets = submodels_symbols(model))
    @tic "model equations" for k in targets
        update_equations!(storage[k], model[k], dt)
    end
end

function update_equations_and_apply_forces!(storage, model::MultiModel, dt, forces; time = NaN, kwarg...)
    # First update all equations
    @tic "equations" update_equations!(storage, model, dt; kwarg...)
    # Then update the cross terms
    @tic "crossterm update" update_cross_terms!(storage, model, dt; kwarg...)
    # Apply forces
    @tic "forces" apply_forces!(storage, model, dt, forces; time = time, kwarg...)
    # Boundary conditions
    @tic "boundary conditions" apply_boundary_conditions!(storage, model; kwarg...)
    # Apply forces to cross-terms
    @tic "crossterm forces" apply_forces_to_cross_terms!(storage, model, dt, forces; time = time, kwarg...)
end

function update_cross_terms!(storage, model::MultiModel, dt; targets = submodels_symbols(model), sources = submodels_symbols(model))
    models = model.models
    for (ctp, ct_s) in zip(model.cross_terms, storage.cross_terms)
        target = ctp.target::Symbol
        source = ctp.source::Symbol
        ct = ctp.cross_term
        is_match = target in targets && source in sources
        is_match = is_match || (has_symmetry(ct) && (target in sources && source in targets))
        if is_match
            model_t = models[target]
            eq = ct_equation(model_t, ctp.target_equation)
            ct_bare_type = Base.typename(typeof(ct)).name
            @tic "$ct_bare_type" update_cross_term!(ct_s, ct, eq, storage[target], storage[source], model_t, models[source], dt)
        end
    end
end

function update_cross_term!(ct_s, ct::CrossTerm, eq, storage_t, storage_s, model_t, model_s, dt)
    state_t = storage_t.state
    state0_t = storage_t.state0

    state_s = storage_s.state
    state0_s = storage_s.state0
    if ct_s[:helper_mode]
        update_cross_term_helper_impl!(state_t, state0_t, state_s, state0_s, ct_s.target, ct_s.source, ct_s, ct::CrossTerm, eq, storage_t, storage_s, model_t, model_s, dt)
    else
        update_cross_term_impl!(state_t, state0_t, state_s, state0_s, ct_s.target, ct_s.source, ct_s, ct::CrossTerm, eq, storage_t, storage_s, model_t, model_s, dt)
    end
end

function update_cross_term_impl!(state_t, state0_t, state_s, state0_s, ct_s_target, ct_s_source, ct_s, ct::CrossTerm, eq, storage_t, storage_s, model_t, model_s, dt)
    for i in 1:ct_s.N
        prepare_cross_term_in_entity!(i, state_t, state0_t, state_s, state0_s, model_t, model_s, ct, eq, dt)
    end
    state_s_v = as_value(state_s)
    state0_s_v = as_value(state0_s)
    for cache in values(ct_s_target)
        update_cross_term_inner_target!(cache, ct, eq, state_s_v, state0_s_v, state_t, state0_t, model_t, model_s, dt)
    end

    state_t_v = as_value(state_t)
    state0_t_v = as_value(state0_t)
    for cache in values(ct_s_source)
        update_cross_term_inner_source!(cache, ct, eq, state_s, state0_s, state_t_v, state0_t_v, model_t, model_s, dt)
    end
end

function update_cross_term_helper_impl!(state_t, state0_t, state_s, state0_s, ct_s_target, ct_s_source, ct_s, ct::CrossTerm, eq, storage_t, storage_s, model_t, model_s, dt)
    for i in 1:ct_s.N
        prepare_cross_term_in_entity!(i, state_t, state0_t, state_s, state0_s, model_t, model_s, ct, eq, dt)
    end
    # Target and source are aliased. We just update one of them.
    @assert ct_s_target === ct_s_source
    update_cross_term_for_entity!(ct_s_source, ct, eq, state_t, state0_t, state_s, state0_s, model_t, model_s, dt)
end


function update_cross_term_inner_source!(cache, ct, eq, state_s, state0_s, state_t_v, state0_t_v, model_t, model_s, dt)
    nothing
end

function update_cross_term_inner_source!(cache::GenericAutoDiffCache{<:Any, <:Any, ∂x, <:Any, <:Any, <:Any, <:Any, <:Any}, ct, eq, state_s, state0_s, state_t, state0_t, model_t, model_s, dt) where ∂x
    state_s_local = local_ad(state_s, 1, ∂x)
    state0_s_local = local_ad(state0_s, 1, ∂x)
    update_cross_term_for_entity!(cache, ct, eq, state_t, state0_t, state_s_local, state0_s_local, model_t, model_s, dt)
end

function update_cross_term_inner_target!(cache, ct, eq, state_s, state0_s, state_t_v, state0_t_v, model_t, model_s, dt)
    nothing
end

function update_cross_term_inner_target!(cache::GenericAutoDiffCache{<:Any, <:Any, ∂x, <:Any, <:Any, <:Any, <:Any, <:Any}, ct, eq, state_s, state0_s, state_t, state0_t, model_t, model_s, dt) where ∂x
    state_t_local = local_ad(state_t, 1, ∂x)
    state0_t_local = local_ad(state0_t, 1, ∂x)
    update_cross_term_for_entity!(cache, ct, eq, state_t_local, state0_t_local, state_s, state0_s, model_t, model_s, dt)
end

function update_cross_term_for_entity!(cache, ct, eq, state_t, state0_t, state_s, state0_s, model_t, model_s, dt)
    v = cache.entries
    vars = cache.variables
    for i in 1:number_of_entities(cache)
        ldisc = local_discretization(ct, i)
        @inbounds for j in vrange(cache, i)
            v_i = @views v[:, j]
            var = vars[j]
            state_t = new_entity_index(state_t, var)
            state0_t = new_entity_index(state0_t, var)
            state_s = new_entity_index(state_s, var)
            state0_s = new_entity_index(state0_s, var)
            update_cross_term_in_entity!(v_i, i, state_t, state0_t, state_s, state0_s, model_t, model_s, ct, eq, dt, ldisc)
        end
    end
end

function update_cross_term_for_entity!(cache::AbstractArray, ct, eq, state_t, state0_t, state_s, state0_s, model_t, model_s, dt)
    for i in axes(cache, 2)
        ldisc = local_discretization(ct, i)
        v_i = @views cache[:, i]
        update_cross_term_in_entity!(v_i, i, state_t, state0_t, state_s, state0_s, model_t, model_s, ct, eq, dt, ldisc)
    end
end

function update_linearized_system!(storage, model::MultiModel, executor = default_executor();
        equation_offset = 0,
        targets = submodels_symbols(model),
        sources = submodels_symbols(model),
        kwarg...)
    @assert equation_offset == 0 "The multimodel version assumes equation_offset == 0, was $equation_offset"
    # Update diagonal blocks (model with respect to itself)
    @tic "models" update_diagonal_blocks!(storage, model, targets; kwarg...)
    # Then, update cross terms (models' impact on other models)
    @tic "cross-model" update_offdiagonal_blocks!(storage, model, targets, sources; kwarg...)
    if haskey(storage, :LinearizedSystem)
        post_update_linearized_system!(storage.LinearizedSystem, executor, storage, model)
    end
end

function update_diagonal_blocks!(storage, model::MultiModel, targets; lsys = storage.LinearizedSystem, kwarg...)
    model_keys = submodels_symbols(model)
    if has_groups(model)
        ng = number_of_groups(model)
        groups = model.groups
        for g in 1:ng
            lsys_g = lsys[g, g]
            subs = groups .== g
            group_targets = model_keys[subs]
            group_keys = intersect(group_targets, targets)
            offsets = get_submodel_offsets(storage, g)
            update_main_linearized_system_subgroup!(storage, model, group_keys, offsets, lsys_g; kwarg...)
        end
    else
        offsets = get_submodel_offsets(storage)
        update_main_linearized_system_subgroup!(storage, model, targets, offsets, lsys; kwarg...)
    end
end

function setup_state(model::MultiModel, initializers; kwarg...)
    state = Dict()
    for key in submodels_symbols(model)
        m = model.models[key]
        init = initializers[key]
        state[key] = setup_state(m, init; kwarg...)
    end
    return state
end

function setup_state(model::MultiModel; kwarg...)
    return internal_multimodel_setup_state(setup_state, model; kwarg...)
end

function setup_parameters(model::MultiModel, arg...; kwarg...)
    data_domains = Dict{Symbol, DataDomain}()
    for k in submodels_symbols(model)
        data_domains[k] = model[k].data_domain
    end
    return setup_parameters(data_domains, model, arg...; kwarg...)
end

function setup_parameters(data_domains::AbstractDict, model::MultiModel; kwarg...)
    F = (model, init) -> setup_parameters(data_domains, model, init)
    return internal_multimodel_setup_state(F, model; kwarg...)
end

function setup_parameters(data_domains::AbstractDict, model::MultiModel, init)
    p = Dict{Symbol, Any}()
    for key in submodels_symbols(model)
        m = model.models[key]
        d = data_domains[key]
        if haskey(init, key) && !isnothing(init[key])
            prm = setup_parameters(d, m, init[key])
        else
            prm = setup_parameters(d, m)
        end
        p[key] = prm
    end
    return p
end


function internal_multimodel_setup_state(F, model::MultiModel; kwarg...)
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
    return F(model, init)
end

export setup_state_and_parameters

function setup_state_and_parameters(model::MultiModel, init)
    state_init = Dict{Symbol, Any}()
    prm_init = Dict{Symbol, Any}()
    for (k, v) in init
        if haskey(model.models, k)
            state_init[k], prm_init[k] = setup_state_and_parameters(model[k], v)
        end
    end
    state = setup_state(model, state_init)
    parameters = setup_parameters(model, prm_init)
    return (state, parameters)
end

function set_default_tolerances!(tol_cfg, model::MultiModel; kwarg...)
    for (k, model) in pairs(model.models)
        cfg_k = Dict{Symbol, Any}()
        set_default_tolerances!(cfg_k, model; kwarg...)
        tol_cfg[k] = cfg_k
    end
end

function setup_forces(model::MultiModel; kwarg...)
    models = model.models
    forces = Dict{Symbol, Any}()
    for k in submodels_symbols(model)
        forces[k] = setup_forces(models[k])
    end
    for (k, v) in pairs(kwarg)
        @assert haskey(models, k) "$k not found in models" keys(model.models)
        forces[k] = v
    end
    return forces
end

function update_secondary_variables!(storage, model::MultiModel, is_state0::Bool; targets = submodels_symbols(model))
    for key in targets
        update_secondary_variables!(storage[key], model.models[key], is_state0)
    end
end

function update_secondary_variables!(storage, model::MultiModel; targets = submodels_symbols(model))
    for key in targets
        update_secondary_variables!(storage[key], model.models[key])
    end
end

function update_secondary_variables_state!(state, model::MultiModel; targets = submodels_symbols(model))
    for key in targets
        update_secondary_variables_state!(state[key], model[key])
    end
end

function evaluate_all_secondary_variables(m::MultiModel, state, parameters = setup_parameters(m))
    out = JUTUL_OUTPUT_TYPE()
    for key in submodels_symbols(m)
        out[key] = evaluate_all_secondary_variables(m.models[key], state[key], parameters[key])
    end
    return out
end

function check_convergence(storage, model::MultiModel, cfg;
        tol = nothing,
        extra_out = false,
        update_report = missing,
        targets = submodels_symbols(model),
        kwarg...
    )
    converged = true
    err = 0
    tol_cfg = cfg[:tolerances]
    errors = OrderedDict()
    for key in targets
        if ismissing(update_report)
            inc = missing
        else
            inc = update_report[key]
        end
        s = storage[key]
        m = model.models[key]
        eqs = m.equations
        eqs_s = s.equations
        eqs_view = s.views.equations
        conv, e, errors[key], = check_convergence(eqs_view, eqs, eqs_s, s, m, tol_cfg[key];
            extra_out = true,
            update_report = inc,
            tol = tol,
            kwarg...
        )
        # Outer model has converged when all submodels are converged
        converged = converged && conv
        err = max(e, err)
    end

    if converged
        if haskey(tol_cfg, :global_convergence_check_function) && !isnothing(tol_cfg[:global_convergence_check_function])
            converged = tol_cfg[:global_convergence_check_function](model, storage)
        end
    end

    if extra_out
        return (converged, err, errors)
    else
        return converged
    end
end

function update_primary_variables!(storage, model::MultiModel; targets = submodels_symbols(model), kwarg...)
    models = model.models
    report = Dict{Symbol, AbstractDict}()
    for key in targets
        m = models[key]
        s = storage[key]
        dx_v = s.views.primary_variables
        pdef = s.variable_definitions.primary_variables
        report[key] = update_primary_variables!(s.state, dx_v, m, pdef; state = s.state, kwarg...)
    end
    return report
end

function update_extra_state_fields!(storage, model::MultiModel, dt, time)
    for key in submodels_symbols(model)
        update_extra_state_fields!(storage[key], model.models[key], dt, time)
    end
    return storage
end

function reset_state_to_previous_state!(storage, model::MultiModel)
    submodels_storage_apply!(storage, model, reset_state_to_previous_state!)
end

function reset_previous_state!(storage, model::MultiModel, state0)
    for key in submodels_symbols(model)
        reset_previous_state!(storage[key], model.models[key], state0[key])
    end
end

function update_after_step!(storage, model::MultiModel, dt, forces; targets = submodels_symbols(model), kwarg...)
    report = JUTUL_OUTPUT_TYPE()
    for key in targets
        report[key] = update_after_step!(storage[key], model.models[key], dt, forces[key]; kwarg...)
    end
    return report
end

function update_before_step!(storage, model::MultiModel, dt, forces; targets = submodels_symbols(model), kwarg...)
    for key in targets
        m = model.models[key]
        update_before_step_multimodel!(storage, model, m, dt, forces, key; kwarg...)
        f = forces[key]
        s = storage[key]
        update_before_step!(s, m, dt, f; kwarg...)
    end
end

function update_before_step_multimodel!(storage, model, submodel, dt, forces, label; kwarg...)

end

function apply_forces!(storage, model::MultiModel, dt, forces; time = NaN, targets = submodels_symbols(model))
    for key in targets
        subforce = forces[key]
        if !isnothing(subforce)
            apply_forces!(storage[key], model.models[key], dt, subforce; time = time)
        end
    end
end

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
    out = JUTUL_OUTPUT_TYPE()
    models = model.models
    for key in submodels_symbols(model)
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

function number_of_degrees_of_freedom(model::MultiModel)
    return sum(number_of_degrees_of_freedom, model.models)
end

function number_of_parameters(model::MultiModel)
    return sum(number_of_parameters, model.models)
end

function number_of_equations(model::MultiModel)
    return sum(number_of_equations, model.models)
end

function reset_variables!(storage, model::MultiModel, state; kwarg...)
    for (k, m) in pairs(model.models)
        reset_variables!(storage[k], m, state[k]; kwarg...)
    end
end

function sort_variables!(model::MultiModel, t = :primary)
    for (k, m) in pairs(model.models)
        sort_variables!(m, t)
    end
    return model
end

function ensure_model_consistency!(model::MultiModel)
    for (k, m) in pairs(model.models)
        ensure_model_consistency!(m)
    end
    return model
end

function check_output_variables(model::MultiModel; label::Symbol = :Model)
    for (k, v) in pairs(model.models)
        check_output_variables(v, label = k)
    end
end
