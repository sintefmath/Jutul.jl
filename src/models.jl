abstract type TervModel end

# Concrete models follow
struct SimulationModel{O<:TervDomain, 
                       S<:TervSystem,
                       F<:TervFormulation,
                       C<:TervContext,
                       D<:TervDiscretization} <: TervModel
    domain::O
    system::S
    context::C
    formulation::F
    discretization::D
    primary_variables
end


function SimulationModel(G, system;
    formulation = FullyImplicit(), 
    context = DefaultContext(),
    discretization = DefaultDiscretization())
    grid = transfer(context, G)
    primary = select_primary_variables(system, formulation, discretization)
    return SimulationModel(grid, system, context, formulation, discretization, primary)
end


function get_primary_variable_names(model::SimulationModel)
    return map((x) -> get_name(x), get_primary_variables(model))
end

function get_primary_variables(model::SimulationModel)
    return model.primary_variables
end

function setup_state(model::TervModel, arg...)
    state = Dict{String, Any}()
    setup_state!(state, model, arg...)
    return state
end

function setup_state!(state, model::TervModel, init_values)
    pvars = get_primary_variables(model)
    for pvar in get_primary_variables(model)
        initialize_primary_variable_value(state, model, pvar, init_values)
    end
    add_extra_state_fields!(state, model)
end

function allocate_storage(model::TervModel)
    d = Dict()
    allocate_storage!(d, model)
    return d
end

function initialize_storage!(d, model::TervModel)
    # Do nothimg
end

function allocate_storage!(d, model::TervModel)
    # Do nothing for Any.
end

"Convert a state containing regular numbers to a state with AD (Dual) status"
function convert_state_ad(model, state)
    context = model.context
    stateAD = deepcopy(state)
    vars = String.(keys(state))

    primary = get_primary_variables(model)
    # Loop over primary variables and set them to AD, with ones at the correct diagonal
    counts = map((x) -> degrees_of_freedom_per_unit(x), primary)
    n_partials = sum(counts)
    @debug "Found $n_partials primary variables."
    offset = 0
    for (i, pvar) in enumerate(primary)
        stateAD = initialize_primary_variable_ad(stateAD, model, pvar, offset, n_partials)
        offset += counts[i]
    end
    primary_names = get_primary_variable_names(model)
    secondary = setdiff(vars, primary_names)
    # Loop over secondary variables and initialize as AD with zero partials
    for s in secondary
        stateAD[s] = allocate_array_ad(stateAD[s], context = context, npartials = n_partials)
    end
    return stateAD
end

function allocate_array(context::TervContext, value, n...)
    tmp = context_convert(context, value)
    return repeat(tmp, n...)
end
# Equations logic follows
function allocate_equations!(d, model, lsys, npartials)
    d["Equations"] = Dict()
end

function update_equations!(model, storage)
    # Do nothing
end

function update_linearized_system!(model::TervModel, storage)
    equations = storage["Equations"]
    lsys = storage["LinearizedSystem"]
    for (key, eq) in equations
        update_linearized_system!(lsys, model, eq)
    end
end


function setup_parameters(model)
    return Dict{String, Any}()
end