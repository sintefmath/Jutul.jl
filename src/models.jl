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
    for pvar in get_primary_variables(model)
        initialize_primary_variable_value(state, model, pvar, init_values)
    end
    add_extra_state_fields!(state, model)
end

"""
Add variables that are not primary (e.g. total masses) but need to be in state.
"""
function add_extra_state_fields!(state, model::TervModel)
    # Do nothing
end

function allocate_storage(model::TervModel)
    d = Dict()
    allocate_storage!(d, model)
    return d
end

"""
Initialize the already allocated storage at the beginning of a simulation.
Use this to e.g. set up extra stuff in state0 needed for initializing the simulation loop.
"""
function initialize_storage!(d, model::TervModel)
    # Do nothing
end

"""
Allocate storage for a given model. The storage consists of all dynamic quantities used in
the simulation. The default implementation allocates properties, equations and linearized system.
"""
function allocate_storage!(d, model::TervModel)
    allocate_properties!(d, model) 
    eqs = allocate_equations!(d, model)
    lsys = allocate_linearized_system!(d, model)
    # We have the equations and the linearized system.
    # Give the equations a chance to figure out their place in the Jacobians.
    align_equations_to_linearized_system!(eqs, lsys, model)
end

function allocate_linearized_system!(d, model::TervModel)
    # Linearized system is going to have dimensions of
    # total number of equations x total number of primary variables
    ndof = 0
    for pvar in get_primary_variables(model)
        ndof += number_of_degrees_of_freedom(model, pvar)
    end

    eqs = d["Equations"]
    I = []
    J = []
    nrows = 0
    for (k, eq) in eqs
        i, j = declare_sparsity(model, eq)
        push!(I, i .+ nrows) # Row indices, offset by the size of preceeding equations
        push!(J, j)          # Column indices
        nrows += number_of_equations(model, eq)
    end
    I = vcat(I...)
    J = vcat(J...)
    vt = float_type(model.context)
    V = zeros(vt, length(I))

    jac = sparse(I, J, V, nrows, ndof)
    lsys = LinearizedSystem(jac)
    d["LinearizedSystem"] = transfer(model.context, lsys)
    return lsys
end

function align_equations_to_linearized_system!(equations, lsys, model)
    for key in keys(equations)
        align_to_linearized_system!(equations[key], lsys, model)
    end
end

"""
Convert a state containing variables as arrays of doubles
to a state where those arrays contain the same value as Dual types.
The dual type is currently taken from ForwardDiff.
"""
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
function allocate_equations!(d, model)
    d["Equations"] = Dict()
end

function update_equations!(model, storage)
    # Do nothing
end

function update_linearized_system!(model::TervModel, storage)
    equations = storage.Equations
    lsys = storage.LinearizedSystem
    for key in keys(equations)
        update_linearized_system!(lsys, model, equations[key])
    end
end

function setup_parameters(model)
    return Dict{String, Any}()
end
