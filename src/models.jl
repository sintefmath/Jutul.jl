export update_state_dependents!, check_convergence

"""
    get_primary_variables(model::SimulationModel)

Get the primary variable definitions (as `OrderedDict`) for a given `model`.

Primary variables are sometimes referred to as solution variables or
primary unknowns. The set of primary variables completely determines the state
of the system together with the `parameters`.
"""
function get_primary_variables(model::SimulationModel)
    return model.primary_variables
end

"""
    get_secondary_variables(model::SimulationModel)

Get the secondary variable definitions (as `OrderedDict`) for a given `model`.

Secondary variables are variables that can be computed from the primary variables
together with the parameters.
"""
function get_secondary_variables(model::SimulationModel)
    return model.secondary_variables
end


"""
    get_variables(model::SimulationModel)

Get all variable definitions (as `OrderedDict`) for a given `model`.

This is the union of [`get_secondary_variables`](@ref) and [`get_primary_variables`](@ref).
"""
function get_variables(model::SimulationModel)
    return merge(get_primary_variables(model), get_secondary_variables(model))
end

"""
    get_variables_and_parameters(model::SimulationModel)

Get all variable definitions (as `OrderedDict`) for a given `model`.

This is the union of [`get_secondary_variables`](@ref),  [`get_primary_variables`](@ref) and [`get_parameters`](@ref).
"""
function get_variables_and_parameters(model::SimulationModel)
    return merge(get_primary_variables(model), get_secondary_variables(model), get_parameters(model))
end

"""
    get_parameters(model::SimulationModel)

Get the parameter definitions (as `OrderedDict`) for a given `model`.

Parameters are defined as static values in a forward simulation that combine
with the primary variables to compute secondary variables and model equations.
"""
function get_parameters(model::SimulationModel)
    return model.parameters
end

function get_variables_by_type(model, type)
    if type == :primary
        return get_primary_variables(model)
    elseif type == :secondary
        return get_secondary_variables(model)
    elseif type == :parameters
        return get_parameters(model)
    elseif type == :all
        return merge(map(x -> get_variables_by_type(model, x), (:primary, :secondary, :parameters))...)
    else
        error("type $type was not :primary, :secondary, :parameters or :all.")
    end
end

export get_variable
"""
    get_variable(model::SimulationModel, name::Symbol)

Get implementation of variable or parameter with name `name` for the model.
"""
function get_variable(model::SimulationModel, name::Symbol; throw = true)
    pvar = model.primary_variables
    svar = model.secondary_variables
    prm = model.parameters
    if haskey(pvar, name)
        var = pvar[name]
    elseif haskey(svar, name)
        var = svar[name]
    elseif haskey(prm, name)
        var = prm[name]
    else
        if throw
            error("Variable $name not found in primary/secondary variables or parameters.")
        else
            var = nothing
        end
    end
    return var
end


export set_primary_variables!, set_secondary_variables!, set_parameters!, replace_variables!
"""
    set_primary_variables!(model, varname = vardef)
    set_primary_variables!(model, varname1 = vardef1, varname2 = vardef2)

Set a primary variable with name `varname` to the definition `vardef` (adding if it does not exist)
"""
function set_primary_variables!(model; kwarg...)
    pvar = get_primary_variables(model)
    set_variable_internal!(pvar, model; kwarg...)
end

"""
    set_secondary_variables!(model, varname = vardef)
    set_secondary_variables!(model, varname1 = vardef1, varname2 = vardef2)


Set a secondary variable with name `varname` to the definition `vardef` (adding if it does not exist)
"""
function set_secondary_variables!(model; kwarg...)
    pvar = get_secondary_variables(model)
    set_variable_internal!(pvar, model; kwarg...)
end

"""
    set_parameters!(model, parname = pardef)

Set a parameter with name `varname` to the definition `vardef` (adding if it does not exist)
"""
function set_parameters!(model; kwarg...)
    pvar = get_parameters(model)
    set_variable_internal!(pvar, model; kwarg...)
end

function set_variable_internal!(vars, model; kwarg...)
    for (k, v) in kwarg
        delete_variable!(model, k)
        v::JutulVariables
        vars[k] = v
    end
end

"""
    replace_variables!(model, throw = true, varname = vardef, varname2 = vardef2)

Replace one or more variables that already exists in the model (primary,
secondary or parameters) with a new definition.

# Arguments
- `model`: instance where variables is to be replaced
- `varname=vardef::JutulVariables`: replace variable with `varname` by `vardef`
- `throw=true`: throw an error if the named variable definition is not found in
  primary or secondary, otherwise silently return the `model`.
"""
function replace_variables!(model; throw = true, kwarg...)
    pvar = get_primary_variables(model)
    svar = get_secondary_variables(model)
    prm = get_parameters(model)
    for (k, v) in kwarg
        done = false
        for vars in [pvar, svar, prm]
            # v::JutulVariables
            if haskey(vars, k)
                oldvar = vars[k]
                if oldvar isa JutulVariables
                    vars[k] = v
                elseif oldvar isa Pair
                    @assert model.system isa CompositeSystem
                    if v isa Pair
                        vars[k] = v
                    else
                        vars[k] = Pair(first(oldvar), v)
                    end
                end
                done = true
                break
            end
        end
        if !done && throw
            error("Unable to replace variable $k, misspelled?")
        end
    end
    return model
end

function delete_variable!(model, var)
    delete!(model.primary_variables, var)
    delete!(model.secondary_variables, var)
    delete!(model.parameters, var)
end

"""
    get_primary_variable_ordered_entities(model::SimulationModel)

Get only the entities where primary variables are present, sorted by their order in the primary variables.
"""
function get_primary_variable_ordered_entities(model::SimulationModel)
    out = JutulEntity[]
    current_entity = nothing
    found = Dict{JutulEntity, Bool}()
    for p in values(model.primary_variables)
        u = associated_entity(p)
        if u != current_entity
            # Note: We assume that primary variables for the same entity follows
            # each other. This is currently asserted in the model constructor.
            push!(out, u)
            current_entity = u
            !haskey(found, u) || throw(ArgumentError("Primary variables are not sorted by entity ($u repeats)."))
            found[u] = true
        end
    end
    return out
end

"""
    number_of_partials_per_entity(model::SimulationModel, entity::JutulEntity)

Get the number of local partial derivatives per entity in a `model` for a given [`JutulEntity`](@ref).
This is the sum of [`degrees_of_freedom_per_entity`](@ref) for all primary variables defined on `entity`.
"""
function number_of_partials_per_entity(model::SimulationModel, entity::JutulEntity)
    n = 0
    for pvar in values(get_primary_variables(model))
        if associated_entity(pvar) == entity
            n += degrees_of_freedom_per_entity(model, pvar)
        end
    end
    return n
end

export setup_state, setup_state!
"""
    setup_state(model::JutulModel, name1 = value1, name2 = value2)

Set up a state for a given model with values for the primary variables defined in the model.
Normally all primary variables must be initialized in this way.

# Arguments
- `name=value`: The name of the primary variable together with the value(s) used to initialize the primary variable.
A scalar (or short vector of the right size for [`VectorVariables`](@ref)) will be repeated over the entire domain,
while a vector (or matrix for [`VectorVariables`](@ref)) with length (number of columns for [`VectorVariables`](@ref))
equal to the entity count (for example, number of cells for a cell variable) will be used directly.

Note: You likely want to overload [`setup_state!`]@ref for a custom model instead of `setup_state`
"""
function setup_state(model::JutulModel, arg...; kwarg...)
    state = Dict{Symbol, Any}()
    setup_state!(state, model, arg...; kwarg...)
    return state
end

function setup_state(model::JutulModel; T = float_type(model.context), kwarg...)
    init = Dict{Symbol, Any}()
    for (k, v) in kwarg
        init[k] = v
    end
    return setup_state(model, init; T = T)
end

"""
    setup_state!(state, model::JutulModel, init_values::AbstractDict = Dict())

Initialize primary variables and other state fields, given initial values as a Dict
"""
function setup_state!(state, model::JutulModel, init_values::Union{JutulStorage, AbstractDict} = Dict(); T = float_type(model.context))
    for (psym, pvar) in get_primary_variables(model)
        initialize_variable_value!(state, model, pvar, psym, init_values, need_value = true, T = T)
    end
    for (psym, svar) in get_secondary_variables(model)
        initialize_variable_value!(state, model, svar, psym, init_values, need_value = false, T = T)
    end
    initialize_extra_state_fields!(state, model, T = T)
end

"""
    initialize_extra_state_fields!(state, model::JutulModel)


Add model-dependent changing variables that need to be in state, but are never AD variables themselves (for example status flags).
"""
function initialize_extra_state_fields!(state, model::JutulModel; kwarg...)
    initialize_extra_state_fields!(state, model.domain, model; kwarg...)
    initialize_extra_state_fields!(state, model.system, model; kwarg...)
    initialize_extra_state_fields!(state, model.formulation, model; kwarg...)
    return state
end

function initialize_extra_state_fields!(state, ::Any, model; kwarg...)
    # Do nothing
end

function setup_parameters!(prm, data_domain, model, initializer::AbstractDict = Dict(); kwarg...)
    for (psym, pvar) in get_parameters(model)
        initialize_parameter_value!(prm, data_domain, model, pvar, psym, initializer; kwarg...)
    end
    return prm
end

"""
    setup_parameters(model::JutulModel; name = value)

Set up a parameter storage for a given model with values for the parameter defined in the model.

# Arguments
- `name=value`: The name of the parameter together with the value(s) of the parameter.
A scalar (or short vector of the right size for [`VectorVariables`](@ref)) will be repeated over the entire domain,
while a vector (or matrix for [`VectorVariables`](@ref)) with length (number of columns for [`VectorVariables`](@ref))
equal to the entity count (for example, number of cells for a cell variable) will be used directly.
"""
function setup_parameters(d::DataDomain, model::JutulModel; T = Float64, perform_copy = true, kwarg...)
    init = Dict{Symbol, Any}()
    for (k, v) in kwarg
        init[k] = v
    end
    return setup_parameters(d, model, init, T = T, perform_copy = perform_copy)
end

function setup_parameters(model::JutulModel, arg...; kwarg...)
    data_domain = DataDomain(physical_representation(model))
    return setup_parameters(data_domain, model, arg...; kwarg...)
end

function setup_parameters(model::SimulationModel, arg...; kwarg...)
    data_domain = model.data_domain
    return setup_parameters(data_domain, model, arg...; kwarg...)
end

function setup_parameters(data_domain::DataDomain, model::JutulModel, init::AbstractDict; kwarg...)
    prm = Dict{Symbol, Any}()
    return setup_parameters!(prm, data_domain, model, init; kwarg...)
end

"""
    state, prm = setup_state_and_parameters(model, init)

Simultaneously set up state and parameters from a single `init` file (typically
a `Dict` containing values that might either be initial values or parameters)
"""
function setup_state_and_parameters(data_domain::DataDomain, model::JutulModel, init::AbstractDict)
    init = copy(init)
    prm = Dict{Symbol, Any}()
    for (k, v) in init
        if k in keys(model.parameters)
            prm[k] = v
            delete!(init, k)
        end
    end
    state = setup_state(model, init)
    parameters = setup_parameters(data_domain, model, prm)
    return (state, parameters)
end

function setup_state_and_parameters(model::JutulModel; kwarg...)
    data_domain = DataDomain(physical_representation(model))
    return setup_state_and_parameters(data_domain, model; kwarg...)
end

function setup_state_and_parameters(model::SimulationModel; kwarg...)
    data_domain = model.data_domain
    return setup_state_and_parameters(data_domain, model; kwarg...)
end

function setup_state_and_parameters(model::SimulationModel, init::AbstractDict)
    data_domain = model.data_domain
    return setup_state_and_parameters(data_domain, model, init)
end

function setup_state_and_parameters(model; kwarg...)
    init = Dict{Symbol, Any}()
    for (k, v) in kwarg
        init[k] = v
    end
    return setup_state_and_parameters(model, init)
end

function setup_state_and_parameters(d::DataDomain, model::SimulationModel; kwarg...)
    init = Dict{Symbol, Any}()
    for (k, v) in kwarg
        init[k] = v
    end
    return setup_state_and_parameters(d::DataDomain, model::SimulationModel, init)
end

"""
    setup_storage(model::JutulModel; kwarg...)

Allocate storage for the model. You should overload setup_storage! if you have a custom
definition.
"""
function setup_storage(model::JutulModel; kwarg...)
    d = JutulStorage()
    setup_storage!(d, model; kwarg...)
    return d
end

"""
    initialize_storage!(storage, model::JutulModel; initialize_state0 = true)

Initialize the already allocated storage at the beginning of a simulation.
Use this to e.g. set up extra stuff in state0 needed for initializing the simulation loop.
"""
function initialize_storage!(storage, model::JutulModel; initialize_state0 = true)
    if initialize_state0
        # Convert state and parameters to immutable for evaluation
        state0_eval = convert_to_immutable_storage(storage[:state0])
        # Evaluate everything (with doubles) to make sure that possible 
        @tic "secondary variables (state0)" update_secondary_variables_state!(state0_eval, model)
        storage[:state0] = state0_eval
    end
    synchronize(model.context)
end

"""
    setup_storage!(storage, model::JutulModel; setup_linearized_system = true,
                                                    setup_equations = true,
                                                    state0 = setup_state(model),
                                                    parameters = setup_parameters(model),
                                                    tag = nothing,
                                                    state0_ad = false,
                                                    state_ad = true,
                                                    kwarg...)

Allocate storage for a given model. The storage consists of all dynamic quantities used in
the simulation. The default implementation allocates properties, equations and linearized system.
"""
function setup_storage!(storage, model::JutulModel; setup_linearized_system = true,
                                                    setup_equations = true,
                                                    state0 = setup_state(model),
                                                    parameters = missing,
                                                    tag = nothing,
                                                    state0_ad = false,
                                                    state_ad = true,
                                                    T = float_type(model.context),
                                                    kwarg...)
    if ismissing(parameters)
        parameters = setup_parameters(model, T = T)
    else
        if T == Float64
            parameters = deepcopy(parameters)
        else
            parameters = setup_parameters(model, parameters, T = T)
        end
    end
    use_internal_ad = state0_ad || state_ad
    @tic "state" if !isnothing(state0)
        if state_ad
            state = convert_state_ad(model, state0, tag)
        else
            state = setup_state(model, deepcopy(state0), T = T)
        end
        if state0_ad
            state0 = convert_state_ad(model, state0, tag)
        else
            state0 = setup_state(model, deepcopy(state0), T = T)
        end
        for (k, v) in pairs(model.parameters)
            if haskey(parameters, k)
                state0[k] = parameters[k]
                state[k] = parameters[k]
            end
        end
        # Both states now contain all parameters, ready to store.
        storage[:state0] = state0
        storage[:state] = state
        storage[:parameters] = parameters
        storage[:primary_variables] = reference_variables(storage, model, :primary)
    end
    @tic "model" setup_storage_model(storage, model)
    @tic "equations" if setup_equations
        storage[:equations] = setup_storage_equations(storage, model; ad = use_internal_ad, tag = tag, kwarg...) 
    end
    @tic "linear system" if setup_linearized_system
        @tic "setup" storage[:LinearizedSystem] = setup_linearized_system!(storage, model)
        # We have the equations and the linearized system.
        # Give the equations a chance to figure out their place in the Jacobians.
        @tic "alignment" align_equations_to_linearized_system!(storage, model)
        @tic "views" setup_equations_and_primary_variable_views!(storage, model)
    end
end

function setup_storage_model(storage, model)
    # Reference the variable definitions used for the model.
    # These are immutable, unlike the model definitions.
    primary = get_primary_variables(model)
    secondary = get_secondary_variables(model)
    parameters = get_parameters(model)
    extra_keys = Symbol[]
    for k in keys(storage.state)
        if k in keys(primary)
            continue
        end
        if k in keys(secondary)
            continue
        end
        if k in keys(parameters)
            continue
        end
        push!(extra_keys, k)
    end
    mutable_references = model.optimization_level > 0
    vars = JutulStorage(always_mutable = mutable_references)
    if !mutable_references
        primary = NamedTuple(pairs(primary))
        secondary = NamedTuple(pairs(secondary))
        parameters = NamedTuple(pairs(parameters))
    end
    vars[:primary_variables] = primary
    vars[:secondary_variables] = secondary
    vars[:parameters] = parameters
    vars[:extra_variable_fields] = extra_keys
    storage[:variable_definitions] = vars
    # Allow for dispatch specific to model's constitutive parts.
    setup_storage_domain!(storage, model, model.domain)
    setup_storage_system!(storage, model, model.system)
    setup_storage_formulation!(storage,  model, model.formulation)
end

function setup_storage_domain!(storage, model, domain)
    # Do nothing
end

function setup_storage_system!(storage, model, system)
    # Do nothing
end

function setup_storage_formulation!(storage,  model, formulation)
    # Do nothing
end

function reference_variables(storage, model::JutulModel, vartype = :primary; kwarg...)
    primaries = OrderedDict()
    state = storage[:state]
    for key in keys(get_variables_by_type(model, vartype))
        primaries[key] = state[key]
    end
    return primaries
end

function setup_storage_equations(storage, model::JutulModel; kwarg...)
    # We use ordered dict since equation alignment with primary variables matter.
    eqs = OrderedDict()
    setup_storage_equations!(eqs, storage, model; kwarg...)
    return eqs
end

function setup_storage_equations!(eqs, storage, model::JutulModel; extra_sparsity = nothing, tag = nothing, kwarg...)
    outstr = "Setting up $(length(model.equations)) groups of governing equations...\n"
    if !isnothing(tag)
        outstr = "$tag: "*outstr
    end
    counter = 1
    num_equations_total = 0
    for (sym, eq) in model.equations
        num = number_of_equations_per_entity(model, eq)
        ne = number_of_entities(model, eq)
        n = num*ne
        # If we were provided with extra sparsity for this equation, pass that on.
        if !isnothing(extra_sparsity) && haskey(extra_sparsity, sym)
            extra = extra_sparsity[sym]
        else
            extra = nothing
        end
        eqs[sym] = setup_equation_storage(model, eq, storage; tag = tag, extra_sparsity = extra, kwarg...)
        counter += 1
        num_equations_total += n
    end
    outstr *= "$num_equations_total equations total distributed over $counter groups.\n"
    @debug outstr
end


"""
    get_sparse_arguments(storage, model)

Get the [`SparsePattern`]@ref for the Jacobian matrix of a given simulator storage and corresponding model.
"""
function get_sparse_arguments(storage, model)
    layout = matrix_layout(model.context)
    return get_sparse_arguments(storage, model, layout, layout)
end

function get_sparse_arguments(storage, model, row_layout::ScalarLayout, col_layout::ScalarLayout)
    ndof = number_of_degrees_of_freedom(model)
    eq_storage = storage[:equations]
    I = Vector{Int}[]
    J = Vector{Int}[]
    numrows = 0
    primary_entities = get_primary_variable_ordered_entities(model)
    for eqname in keys(model.equations)
        numcols = 0
        eq = model.equations[eqname]
        eq_s = eq_storage[eqname]
        for u in primary_entities
            S = declare_sparsity(model, eq, eq_s, u, row_layout, col_layout)
            if !isnothing(S)
                push!(I, S.I .+ numrows) # Row indices, offset by the size of preceeding equations
                push!(J, S.J .+ numcols) # Column indices, offset by the partials in entities we have passed
            end
            numcols += number_of_degrees_of_freedom(model, u)
        end
        @assert numcols == ndof "Mismatch in number of columns ($numcols) and number degrees of freedom ($ndof) for equation $eqname"
        # Number of equations correspond to number of rows
        numrows += number_of_equations(model, eq)
    end
    I = vcat(I...)
    J = vcat(J...)
    return SparsePattern(I, J, numrows, ndof, row_layout, col_layout)
end

function get_sparse_arguments(storage, model, row_layout::T, col_layout::T) where T<:BlockMajorLayout
    eq_storage = storage[:equations]
    primary_entities = get_primary_variable_ordered_entities(model)
    entity = only(primary_entities)
    block_size = degrees_of_freedom_per_entity(model, entity)
    ndof = number_of_degrees_of_freedom(model) ÷ block_size
    S = missing
    for eqname in keys(model.equations)
        eq = model.equations[eqname]
        eq_s = eq_storage[eqname]
        eqs_per_entity = number_of_equations_per_entity(model, eq)
        dof_per_entity = degrees_of_freedom_per_entity(model, entity)

        S_i = declare_sparsity(model, eq, eq_s, entity, row_layout, col_layout)
        if ismissing(S)
            S = S_i
        else
            @assert S.I == S_i.I
            @assert S.J == S_i.J
            @assert S.n/S.block_n == S_i.n/S_i.block_n
            @assert S.m/S.block_m == S_i.m/S_i.block_m
        end
    end
    @assert !ismissing(S)

    it = index_type(model.context)

    I = Vector{it}()
    J = Vector{it}()
    nv = length(S.I)
    sizehint!(I, nv)
    sizehint!(J, nv)
    for i in S.I
        push!(I, convert(it, i))
    end
    for j in S.J
        push!(J, convert(it, j))
    end
    return SparsePattern(I, J, ndof, ndof, row_layout, col_layout, block_size)
end

function setup_linearized_system!(storage, model::JutulModel)
    # Linearized system is going to have dimensions of
    # total number of equations x total number of primary variables
    if !haskey(storage, :equations)
        error("Unable to allocate linearized system - no equations found.")
    end
    # layout = matrix_layout(model.context)
    sparse_pattern = get_sparse_arguments(storage, model)
    if represented_as_adjoint(matrix_layout(model.context))
        sparse_pattern = sparse_pattern'
    end
    lsys = setup_linearized_system(sparse_pattern, model)
    storage[:LinearizedSystem] = lsys
    # storage[:LinearizedSystem] = transfer(model.context, lsys)
    return lsys
end

function setup_linearized_system(sparse_arg, model)
    context = model.context
    return LinearizedSystem(sparse_arg, context, matrix_layout(context))
end

function align_equations_to_linearized_system!(storage, model::JutulModel; kwarg...)
    eq_storage = storage[:equations]
    eqs = model.equations
    jac = storage[:LinearizedSystem].jac
    align_equations_to_jacobian!(eq_storage, eqs, jac, model; kwarg...)
end

function align_equations_to_jacobian!(eq_storage, equations, jac, model; equation_offset = 0, variable_offset = 0)
    for key in keys(equations)
        eq_s = eq_storage[key]
        eq = equations[key]
        align_to_jacobian!(eq_s, eq, jac, model, equation_offset = equation_offset, variable_offset = variable_offset)
        equation_offset += number_of_equations(model, eq)
    end
    equation_offset
end

function allocate_array(context::JutulContext, value, n...)
    tmp = transfer(context, value)
    return repeat(tmp, n...)
end

"""
    update_state_dependents!(storage, model, dt, forces; time = NaN, update_secondary = true)

Perform updates of everything that depends on the state: A full linearization for the current primary variables.

This includes properties, governing equations and the linearized system itself.
"""
function update_state_dependents!(storage, model::JutulModel, dt, forces; time = NaN, update_secondary = true, kwarg...)
    t_s = @elapsed if update_secondary
        @tic "secondary variables" update_secondary_variables!(storage, model; kwarg...)
        @tic "extra state fields" update_extra_state_fields!(storage, model, dt, time)
    end
    t_eq = @elapsed update_equations_and_apply_forces!(storage, model, dt, forces; time = time, kwarg...)
    return (secondary = t_s, equations = t_eq)
end

"""
    update_equations_and_apply_forces!(storage, model, dt, forces; time = NaN)

Update the model equations and apply boundary conditions and forces. Does not fill linearized system.
"""
function update_equations_and_apply_forces!(storage, model, dt, forces; time = NaN, kwarg...)
    @tic "equations" update_equations!(storage, model, dt; kwarg...)
    @tic "forces" apply_forces!(storage, model, dt, forces; time = time, kwarg...)
    @tic "boundary conditions" apply_boundary_conditions!(storage, model; kwarg...)
end

function apply_boundary_conditions!(storage, model::JutulModel)
    parameters = storage.parameters
    apply_boundary_conditions!(storage, parameters, model)
end

apply_boundary_conditions!(storage, parameters, model) = nothing

"""
    update_equations!(storage, model, dt = nothing)

Update the governing equations using the current set of primary variables, parameters and secondary variables. Does not fill linearized system.
"""
function update_equations!(storage, model, dt = nothing)
    update_equations!(storage, storage.equations, model.equations, model, dt)
end

function update_equations!(storage, equations_storage, equations, model, dt)
    for (key, eq) in pairs(equations)
        @tic "$key" update_equation!(equations_storage[key], eq, storage, model, dt)
    end
end

"""
    update_linearized_system!(storage, model::JutulModel; <keyword arguments>)

Update the linearized system with the current set of equations.
"""
function update_linearized_system!(storage, model::JutulModel, executor = default_executor(); lsys = storage.LinearizedSystem, kwarg...)
    eqs = model.equations
    eqs_storage = storage.equations
    eqs_views = storage.views.equations
    update_linearized_system!(lsys, eqs, eqs_storage, eqs_views, model; kwarg...)
    post_update_linearized_system!(lsys, executor, storage, model)
end

function post_update_linearized_system!(lsys, executor, storage, model)
    # Do nothing.
end

function update_linearized_system!(lsys, equations, eqs_storage, eqs_views, model::JutulModel; equation_offset = 0, r = lsys.r_buffer, nzval = lsys.jac_buffer)
    for key in keys(equations)
        @tic "$key" begin
            eq = equations[key]
            eqs_s = eqs_storage[key]
            r_view = eqs_views[key]
            update_linearized_system_equation!(nzval, r_view, model, eq, eqs_s)
        end
    end
end

"""
    local_residual_view(r_buf, model, eq, equation_offset)

Get a matrix view of the residual so that, independent of ordering,
the column index corresponds to the entity index for the given equation `eq`
starting at `equation_offset` in the global residual buffer `r_buf`.
"""
function local_residual_view(r_buf, model, eq, equation_offset)
    N = number_of_equations(model, eq)
    n = number_of_equations_per_entity(model, eq)
    m = N ÷ n
    return as_cell_major_matrix(r_buf, n, m, model, equation_offset)
end

function local_residual_view(r_buf, model, n::Int, m::Int)
    return as_cell_major_matrix(r_buf, n, m, model, 0)
end

"""
    set_default_tolerances(model)

Set default tolerances for the nonlinear convergence check of the governing equations.
"""
function set_default_tolerances(model; kwarg...)
    tol_cfg = Dict{Symbol, Any}()
    set_default_tolerances!(tol_cfg, model; kwarg...)
    return tol_cfg
end

function set_default_tolerances!(tol_cfg, model::SimulationModel; tol = 1e-3)
    tol_cfg[:default] = tol
end

function check_convergence(storage, model, config; kwarg...)
    eqs_views = storage.views.equations
    eqs = model.equations
    eqs_s = storage.equations
    if config isa JutulConfig
        cfg_tol = config[:tolerances]
    else
        cfg_tol = config
    end
    check_convergence(eqs_views, eqs, eqs_s, storage, model, cfg_tol; kwarg...)
end

function check_convergence(eqs_views, eqs, eqs_s, storage, model, tol_cfg;
        iteration = nothing,
        extra_out = false,
        tol = nothing,
        tol_factor = 1.0,
        kwarg...
    )
    converged = true
    e = 0
    if isnothing(tol)
        tol = tol_cfg[:default]
    end
    output = []

    for key in keys(eqs)
        eq = eqs[key]
        eq_s = eqs_s[key]
        r_v = eqs_views[key]
        @tic "$key" all_crits = convergence_criterion(model, storage, eq, eq_s, r_v; kwarg...)
        e_keys = keys(all_crits)
        tols = Dict()
        for e_k in e_keys
            if haskey(tol_cfg, key)
                v = tol_cfg[key]
                if isa(v, AbstractFloat)
                    t_e = v
                elseif haskey(v, e_k)
                    t_e = v[e_k]
                else
                    t_e = tol
                end
            else
                t_e = tol
            end
            errors = all_crits[e_k].errors
            if minimum(errors) < -10*eps(Float64)
                @warn "Negative residuals detected for $key: $e_k. Programming error?" errors
            end
            max_error = maximum(errors)
            e = max(e, max_error/t_e)
            t_actual = t_e*tol_factor
            converged = converged && max_error < t_actual
            tols[e_k] = t_actual
        end
        if extra_out
            push!(output, (name = key, criterions = all_crits, tolerances = tols))
        end
    end
    if extra_out
        return (converged, e, output)
    else
        return converged
    end
end

"""
Apply a set of forces to all equations. Equations that don't support a given force
will just ignore them, thanks to the power of multiple dispatch.
"""
function apply_forces!(storage, model, dt, forces; time = NaN)
    equations = model.equations
    equations_storage = storage.equations
    for key in keys(equations)
        eq = equations[key]
        eq_s = equations_storage[key]
        diag_part = get_diagonal_entries(eq, eq_s)
        for fkey in keys(forces)
            force = forces[fkey]
            apply_forces_to_equation!(diag_part, storage, model, eq, eq_s, force, time)
        end
    end
end

function apply_forces!(storage, model, dt, ::Nothing; time = NaN)

end

export setup_forces
"""
    setup_forces(model::JutulModel; force_name = force_value)

Set up forces for a given model. Keyword arguments varies depending on what the
model supports.
"""
function setup_forces(model::JutulModel)
    return NamedTuple()
end

function solve_and_update!(storage, model::JutulModel, dt = nothing; linear_solver = nothing, recorder = nothing, executor = default_executor(), kwarg...)
    lsys = storage.LinearizedSystem
    context = model.context
    t_solve = @elapsed begin
        @tic "linear solve" (ok, n, history) = linear_solve!(lsys, linear_solver, context, model, storage, dt, recorder, executor)
    end
    t_update = @elapsed @tic "primary variables" update = update_primary_variables!(storage, model; kwarg...)
    return (t_solve, t_update, n, history, update)
end

function update_primary_variables!(storage, model::JutulModel; kwarg...)
    dx = storage.views.primary_variables
    primary_defs = storage.variable_definitions.primary_variables
    primary = storage.primary_variables
    update_primary_variables!(primary, dx, model, primary_defs; state = storage.state, kwarg...)
end

function update_extra_state_fields!(storage, model, dt, time)
    return storage
end

function update_primary_variables!(primary_storage, dx, model::JutulModel, primary = get_primary_variables(model); relaxation = 1.0, check = false, state = missing)
    report = Dict{Symbol, Any}()
    for (pkey, p) in pairs(primary)
        dxi = dx[pkey]
        if check
            ok = check_increment(dxi, p, pkey)
            if !ok
                error("Primary variables recieved invalid updates.")
            end
        end
        @tic "$pkey" update_primary_variable!(primary_storage, p, pkey, model, dxi, relaxation)
        report[pkey] = increment_norm(dxi, state, model, primary_storage[pkey], p)
    end
    return report
end

function increment_norm(dX, state, model, X, pvar)
    T = eltype(dX)
    scale = @something variable_scale(pvar) one(T)
    max_v = sum_v = zero(T)
    for dx in dX
        dx_abs = abs(dx)
        max_v = max(max_v, dx_abs)
        sum_v += dx_abs
    end
    return (sum = scale*sum_v, max = scale*max_v)
end

"""

"""
function update_before_step!(storage, model, dt, forces; kwarg...)
    update_before_step!(storage, model.domain, model, dt, forces; kwarg...)
    update_before_step!(storage, model.system, model, dt, forces; kwarg...)
    update_before_step!(storage, model.formulation, model, dt, forces; kwarg...)
end

function update_before_step!(storage, ::Any, model, dt, forces; time = NaN, recorder = ProgressRecorder(), update_explicit = true)
    state = storage.state
    for (k, prm) in pairs(storage.variable_definitions.parameters)
        update_parameter_before_step!(state[k], prm, storage, model, dt, forces)
    end
end

function update_after_step!(storage, model, dt, forces; kwarg...)
    state = storage.state
    state0 = storage.state0
    report = OrderedDict{Symbol, Any}()
    defs = storage.variable_definitions
    pvar = defs.primary_variables
    for k in keys(pvar)
        report[k] = variable_change_report(state[k], state0[k], pvar[k])
    end
    svar = defs.secondary_variables
    for k in keys(svar)
        report[k] = variable_change_report(state[k], state0[k], svar[k])
    end
    update_after_step!(storage, model.domain, model, dt, forces; kwarg...)
    update_after_step!(storage, model.system, model, dt, forces; kwarg...)
    update_after_step!(storage, model.formulation, model, dt, forces; kwarg...)

    # Synchronize previous state with new state
    for key in keys(pvar)
        update_values!(state0[key], state[key])
    end
    for key in keys(svar)
        update_values!(state0[key], state[key])
    end
    for key in defs.extra_variable_fields
        update_values!(state0[key], state[key])
    end
    return report
end

"""
    update_parameter_before_step!(prm_val, prm, storage, model, dt, forces)

Update parameters before time-step. Used for hysteretic parameters.
"""
function update_parameter_before_step!(prm_val, prm, storage, model, dt, forces)
    # Do nothing by default.
    return prm_val
end

function variable_change_report(X::AbstractArray, X0::AbstractArray{T}, pvar) where T<:Real
    max_dv = max_v = sum_dv = sum_v = zero(T)
    @inbounds @simd for i in eachindex(X)
        x = value(X[i])::T
        dx = x - value(X0[i])

        dx_abs = abs(dx)
        max_dv = max(max_dv, dx_abs)
        sum_dv += dx_abs

        x_abs = abs(x)
        max_v = max(max_v, x_abs)
        sum_v += x_abs
    end
    return (dx = (sum = sum_dv, max = max_dv), x = (sum = sum_v, max = max_v), n = length(X))
end

function variable_change_report(X, X0, pvar)
    return nothing
end

function update_after_step!(storage, ::Any, model, dt, forces; time = NaN)
    # Do nothing
end

function get_output_state(storage, model)
    # As this point (after a converged step) state0 should be state without AD.
    s0 = storage.state0
    D = JUTUL_OUTPUT_TYPE()
    for k in model.output_variables
        if haskey(s0, k)
            D[k] = copy(s0[k])
        end
    end
    return D
end

function replace_values!(old, updated)
    for f in keys(old)
        if haskey(updated, f)
            update_values!(old[f], updated[f])
        end
    end
end

function reset_state_to_previous_state!(storage, model)
    # Replace primary variable values with those from previous state
    replace_values!(storage.primary_variables, storage.state0)
    # Update secondary variables to be in sync with current primary values
    update_secondary_variables!(storage, model)
end

function reset_previous_state!(storage, model, state0)
    replace_values!(storage.state0, state0)
end

function reset_variables!(storage, model, new_vars; type = :state)
    replace_values!(storage[type], new_vars)
end

function setup_equations_and_primary_variable_views!(storage, model)
    setup_equations_and_primary_variable_views!(storage, model, storage.LinearizedSystem)
end

function setup_equations_and_primary_variable_views!(storage, model, lsys)
    storage[:views] = setup_equations_and_primary_variable_views(storage, model, lsys.r_buffer, lsys.dx_buffer)
end

function setup_equations_and_primary_variable_views(storage, model, r, dx)
    if ismissing(dx)
        pvar = missing
    else
        pvar = setup_primary_variable_views(storage, model, dx)
    end
    if ismissing(r)
        equations = missing
    else
        equations = setup_equations_views(storage, model, r)
    end
    return (equations = equations, primary_variables = pvar)
end

function setup_primary_variable_views(storage, model, dx)
    out = JutulStorage()
    layout = matrix_layout(model.context)
    primary = get_primary_variables(model)
    # Depending on the variable ordering, this can require a bit of array
    # reshaping/indexing tricks.
    if is_cell_major(layout)
        offset = 0 # Offset into global r array
        for u in get_primary_variable_ordered_entities(model)
            np = number_of_partials_per_entity(model, u)
            nu = count_active_entities(model.domain, u)
            Dx = get_matrix_view(dx, np, nu, false, offset)
            local_offset = 0
            for (pkey, p) in primary
                # This is a bit inefficient
                if u != associated_entity(p)
                    continue
                end
                ni = degrees_of_freedom_per_entity(model, p)
                dxi = view(Dx, (local_offset+1):(local_offset+ni), :)
                local_offset += ni
                out[pkey] = dxi
            end
            offset += nu*np
        end
    else
        offset = 0
        for (pkey, p) in primary
            n = number_of_degrees_of_freedom(model, p)
            m = degrees_of_freedom_per_entity(model, p)
            rng = (offset+1):(n+offset)
            if m == 0
                @assert n == 0
                dxi = similar(dx, 0)
            else
                dxi = reshape(view(dx, rng), n÷m, m)'
            end
            out[pkey] = dxi
            offset += n
        end
    end
    return convert_to_immutable_storage(out)
end

function setup_equations_views(storage, model, r)
    out = JutulStorage()
    bz = model_block_size(model)
    if bz == 1
        equation_offset = 0
        for (key, eq) in pairs(model.equations)
            neq = number_of_equations(model, eq)
            r_view = local_residual_view(r, model, eq, equation_offset)
            equation_offset += neq
            out[key] = r_view
        end
    else
        ndof = number_of_degrees_of_freedom(model)
        block_offset = 0
        R = local_residual_view(r, model, bz, ndof ÷ bz)
        for (key, eq) in pairs(model.equations)
            n_block_local = number_of_equations_per_entity(model, eq)
            r_view_eq = view(R, (block_offset+1):(block_offset+n_block_local), :)
            block_offset += n_block_local
            out[key] = r_view_eq
        end
    end
    return convert_to_immutable_storage(out)
end

function ensure_model_consistency!(model::JutulModel)
    return model
end

function check_output_variables(model::JutulModel; label::Symbol = :Model)
    tmp = JutulStorage()
    initialize_extra_state_fields!(tmp, model)
    vars = get_variables_by_type(model, :all)
    possible = [keys(tmp)..., keys(vars)...]
    missing_vars = Symbol[]
    for k in model.output_variables
        if k in possible
            continue
        end
        push!(missing_vars, k)
    end
    n = length(missing_vars)
    if n > 0
        possible = sort!(possible)
        fmt(x) = "\n    "*join(x, ",\n    ")
        jutul_message("$label", "$n requested output variables were not found in model. Possible spelling errors?\nRequested variables that were missing: $(fmt(missing_vars))\nVariables in model: $(fmt(possible)).", color = :yellow)
    end
end
