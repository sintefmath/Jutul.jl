export MultiModel

struct MultiModel <: TervModel
    models::NamedTuple
    groups::Vector
    context::TervContext
    function MultiModel(models; groups = collect(1:length(models)), context = DefaultContext())
        nm = length(models)
        @assert maximum(groups) <= nm
        @assert minimum(groups) > 0
        @assert length(groups) == nm
        for m in models
            @assert context == m.context
        end
        new(models, groups, context)
    end
end

abstract type CrossTerm end

"""
A cross model term where the dependency is injective and the term is additive:
(each addition to a unit in the target only depends one unit from the source, 
and is added into that position upon application)
"""
struct InjectiveCrossTerm <: CrossTerm
    impact             # 2 by N - first row is target, second is source
    units              # tuple - first tuple is target, second is source
    crossterm_target   # The cross-term, with AD values taken relative to the target
    crossterm_source   # Same cross-term, AD values taken relative to the source
    cross_jac_pos      # Positions that map crossterm_source C into the off-diagonal block ∂C/∂S where S are impacted primary variables of source
    equations_per_unit # Number of equations per impact
    npartials_target   # Number of partials per equation (in target)
    npartials_source   # (in source)
    function InjectiveCrossTerm(target_eq, target_model, source_model)
        context = target_model.context
        target_unit = domain_unit(target_eq)
        target_impact, source_impact, source_unit = get_domain_intersection(target_unit, target_model, source_model)
        noverlap = length(target_impact)
        @assert noverlap == length(source_impact) "Injective source must have one to one mapping between impact and source."
        I = index_type(context)
        # Infer Unit from target_eq
        equations_per_unit = number_of_equations_per_unit(target_eq)

        npartials_target = number_of_partials_per_unit(target_model, target_unit)
        npartials_source = number_of_partials_per_unit(source_model, source_unit)
        
        alloc = (n) -> allocate_array_ad(equations_per_unit, noverlap, context = context, npartials = n)

        c_term_target = alloc(npartials_target)
        c_term_source = alloc(npartials_source)

        jac_pos = zeros(I, equations_per_unit*npartials_source, noverlap)
        # Units and overlap - target, then source
        units = (target = target_unit, source = source_unit)
        overlap = (target = target_impact, source = source_impact)
        new(overlap, units, c_term_target, c_term_source, jac_pos, equations_per_unit, npartials_target, npartials_target)
    end
end

function align_to_jacobian!(ct::InjectiveCrossTerm, jac, target::TervModel, source::TervModel; row_offset = 0, col_offset = 0)
    jpos = ct.cross_jac_pos
    nunits = size(ct.cross_jac_pos, 2)

    nunits_source = count_units(source.domain, ct.units.source)

    impact_target = ct.impact[1]
    impact_source = ct.impact[2]
    
    @assert length(impact_target) == nunits
    @assert length(impact_source) == nunits

    ne = ct.equations_per_unit
    nder = ct.npartials_source
    for overlap_no = 1:nunits
        for eqNo = 1:ne
            for derNo = 1:nder
                p = (eqNo-1)*nder + derNo
                row = (eqNo-1)*nunits + impact_target[overlap_no] + row_offset
                col = (derNo-1)*nunits_source + impact_source[overlap_no] + col_offset
                jpos[p, overlap_no] = find_sparse_position(jac, row, col)
            end
        end
    end
end

function declare_sparsity(target_model, source_model, x::InjectiveCrossTerm)
    n_partials = x.npartials_source
    n_eqs = x.equations_per_unit
    nunits_source = count_units(source_model.domain, x.units[2])

    target_impact = x.impact.target
    source_impact = x.impact.source

    n_impact = length(target_impact)

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
    return (I, J, n, m)
end


function get_domain_intersection(u::TervUnit, target_model::TervModel, source_model::TervModel)
    return get_domain_intersection(u, target_model.domain, source_model.domain)
end

"""
For a given unit in domain target_d, find any indices into that unit that is connected to
any units in source_d. The interface is limited to a single unit-unit impact.
The return value is a tuple of indices and the corresponding unit
"""
function get_domain_intersection(u::TervUnit, target_d::TervDomain, source_d::TervDomain)
    (nothing, nothing, Cells())
end

function number_of_models(model::MultiModel)
    return length(model.models)
end

function get_primary_variable_names(model::MultiModel)

end

function setup_state!(state, model::MultiModel, init_values)
    error("Mutating version of setup_state not supported for multimodel.")
end

function setup_simulation_storage(model::MultiModel; state0 = setup_state(model), parameters = setup_parameters(model))
    storage = Dict()
    for key in keys(model.models)
        m = model.models[key]
        storage[key] = setup_simulation_storage(m,  state0 = state0[key], 
                                                    parameters = parameters[key], 
                                                    setup_linearized_system = false)
    end
    allocate_cross_terms(storage, model)
    allocate_linearized_system!(storage, model)
    align_equations_to_linearized_system!(storage, model)
    align_cross_terms_to_linearized_system!(storage, model)
    return storage
end

function allocate_cross_terms(storage, model::MultiModel)
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
                ct = InjectiveCrossTerm(eq, target_model, source_model)
                if length(ct.impact) == 0
                    # Just insert nothing, so we can easily spot no overlap
                    ct = nothing
                end
                d[key] = ct
            end
            sources[source] = d
        end
        crossd[target] = sources
    end
    storage[:cross_terms] = crossd
end

function align_equations_to_linearized_system!(storage, model::MultiModel; row_offset = 0, col_offset = 0)
    models = model.models
    lsys = storage[:LinearizedSystem]
    for key in keys(models)
        submodel = models[key]
        eqs = storage[key][:equations]
        nrow_end = align_equations_to_jacobian!(eqs, lsys.jac, submodel, row_offset = row_offset, col_offset = col_offset)
        nrow = nrow_end - row_offset
        ndof = number_of_degrees_of_freedom(submodel)
        @assert nrow == ndof "Submodels must have equal number of equations and degrees of freedom. Found $nrow equations and $ndof variables for submodel $key"
        row_offset += ndof
        col_offset += ndof # Assuming that each model by itself forms a well-posed, square Jacobian...
    end
end

function align_cross_terms_to_linearized_system!(storage, model::MultiModel; row_offset = 0, col_offset = 0)
    models = model.models
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
            col_offset += number_of_degrees_of_freedom(source_model)
        end
        row_offset += number_of_degrees_of_freedom(target_model)
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

#function align_cross_terms_to_linearized_system!(crossterms, lsys, model; row_offset = 0, col_offset = 0)
#    models = model.models

    #for target in keys(models)
    #    for source in keys(models)
    #    end
    #end
    #for key in keys(equations)
        # eq = equations[key]
        # align_to_linearized_system!(eq, lsys, model, row_offset = row_offset, col_offset = col_offset)
        # row_offset += number_of_equations(model, eq)
    #end
    #row_offset
#end


function get_sparse_arguments(storage, model::MultiModel, target::Symbol, source::Symbol)
    models = model.models
    target_model = models[target]
    source_model = models[source]
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
                i, j, = declare_sparsity(target_model, source_model, x)
                push!(I, i .+ row_offset)
                push!(J, j)
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
        end
        row_offset += n
    end
    I = vcat(I...)
    J = vcat(J...)
    V = vcat(V...)
    return (I, J, V, row_offset, col_offset)
end

function allocate_linearized_system!(storage, model::MultiModel)
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

function initialize_storage!(storage, model::MultiModel)
    submodels_storage_apply!(storage, model, initialize_storage!)
end

function update_equations!(storage, model::MultiModel, arg...)
    submodels_storage_apply!(storage, model, update_equations!, arg...)
end

function update_linearized_system!(storage, model::MultiModel; row_offset = 0)
    lsys = storage.LinearizedSystem
    for key in keys(model.models)
        m = model.models[key]
        s = storage[key]
        eqs = s.equations
        update_linearized_system!(lsys, eqs, m; row_offset = row_offset)
        row_offset += number_of_degrees_of_freedom(m)
    end
end

function setup_state(model::MultiModel, subs...)
    @assert length(subs) == number_of_models(model)
    state = Dict()
    for (i, key) in enumerate(keys(model.models))
        m = model.models[key]
        state[key] = setup_state(m, subs[i])
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

function convert_state_ad(model::MultiModel, state)
    stateAD = deepcopy(state)
    for key in keys(model.models)
        stateAD[key] = convert_state_ad(model.models[key], state[key])
    end
    return stateAD
end

function update_properties!(storage, model::MultiModel)
    submodels_storage_apply!(storage, model, update_properties!)
end

function update_equations!(storage, model::MultiModel, dt)
    submodels_storage_apply!(storage, model, update_equations!, dt)
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

function update_state!(storage, model::MultiModel)
    dx = storage.LinearizedSystem.dx
    models = model.models

    offset = 0
    for key in keys(models)
        m = models[key]
        s = storage[key]
        ndof = number_of_degrees_of_freedom(m)
        dx_v = view(dx, (offset+1):(offset+ndof))
        update_state!(s.state, dx_v, m)
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
