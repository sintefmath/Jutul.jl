export update_state_dependents!

#function get_primary_variable_names(model::SimulationModel)
#    return map((x) -> get_name(x), get_primary_variables(model))
#end

function get_primary_variable_symbols(model::SimulationModel)
    return map((x) -> get_name(x), get_primary_variables(model))
end

function get_primary_variables(model::SimulationModel)
    return model.primary_variables
end

function number_of_partials_per_unit(model::SimulationModel, unit::TervUnit)
    n = 0
    for p in get_primary_variables(model)
        if associated_unit(p) == unit
            n += 1
        end
    end
    return n
end

"""
Set up a state. You likely want to overload setup_state! instead of this one.
"""
function setup_state(model::TervModel, arg...)
    state = Dict{Symbol, Any}()
    setup_state!(state, model, arg...)
    return state
end

"""
Initialize primary variables and other state fields, given initial values as a Dict
"""
function setup_state!(state, model::TervModel, init_values::Dict)
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

"""
Main function for storage that allocates and initializes storage for a simulation
"""
function setup_simulation_storage(model::TervModel; state0 = setup_state(model), parameters = setup_parameters(model), kwargs...)
    storage = allocate_storage(model; kwargs...)
    storage[:parameters] = parameters
    storage[:state0] = state0
    storage[:state] = convert_state_ad(model, state0)
    return storage
end

"""
Allocate storage for the model. You should overload allocate_storage! if you have a custom
definition.
"""
function allocate_storage(model::TervModel; kwarg...)
    d = Dict{Symbol, Any}()
    allocate_storage!(d, model; kwarg...)
    return d
end

"""
Initialize the already allocated storage at the beginning of a simulation.
Use this to e.g. set up extra stuff in state0 needed for initializing the simulation loop.
"""
function initialize_storage!(storage, model::TervModel)
    # Do nothing
end

"""
Allocate storage for a given model. The storage consists of all dynamic quantities used in
the simulation. The default implementation allocates properties, equations and linearized system.
"""
function allocate_storage!(storage, model::TervModel; setup_linearized_system = true)
    storage[:properties] = allocate_properties(storage, model) 
    storage[:equations] = allocate_equations(storage, model)
    if setup_linearized_system
        storage[:LinearizedSystem] = allocate_linearized_system!(storage, model)
        # We have the equations and the linearized system.
        # Give the equations a chance to figure out their place in the Jacobians.
        align_equations_to_linearized_system!(storage, model)
    end
end

function allocate_properties(storage, model::TervModel)
    props = Dict()
    allocate_properties!(props, storage, model)
    return props
end

function allocate_properties!(props, storage, model::TervModel)
    # Default: No properties
end

function allocate_equations(storage, model::TervModel)
    eqs = Dict()
    allocate_equations!(eqs, storage, model)
    return eqs
end

function allocate_equations!(eqs, storage, model::TervModel)
    # Default: No equations.
end

function get_sparse_arguments(storage, model)
    ndof = number_of_degrees_of_freedom(model)
    eqs = storage[:equations]
    I = []
    J = []
    nrows = 0
    for (k, eq) in eqs
        S = declare_sparsity(model, eq)
        i = S[1]
        j = S[2]
        push!(I, i .+ nrows) # Row indices, offset by the size of preceeding equations
        push!(J, j)          # Column indices
        nrows += S[3]
    end
    I = vcat(I...)
    J = vcat(J...)
    vt = float_type(model.context)
    V = zeros(vt, length(I))
    return (I, J, V, nrows, ndof)
end

function allocate_linearized_system!(storage, model::TervModel)
    # Linearized system is going to have dimensions of
    # total number of equations x total number of primary variables
    if !haskey(storage, :equations)
        error("Unable to allocate linearized system - no equations found.")
    end
    I, J, V, nrows, ncols = get_sparse_arguments(storage, model)
    jac = sparse(I, J, V, nrows, ncols)
    lsys = LinearizedSystem(jac)
    storage[:LinearizedSystem] = transfer(model.context, lsys)
    return lsys
end

function align_equations_to_linearized_system!(storage, model::TervModel; kwarg...)
    align_equations_to_linearized_system!(storage[:equations], storage[:LinearizedSystem], model; kwarg...)
end

function align_equations_to_linearized_system!(equations, lsys, model; row_offset = 0, col_offset = 0)
    for key in keys(equations)
        eq = equations[key]
        align_to_linearized_system!(eq, lsys, model, row_offset = row_offset, col_offset = col_offset)
        row_offset += number_of_equations(model, eq)
    end
end

function allocate_array(context::TervContext, value, n...)
    tmp = context_convert(context, value)
    return repeat(tmp, n...)
end

# Equations logic follows
function allocate_equations!(d, model)
    d[:Equations] = Dict()
end

"""
Perform updates of everything that depends on the state.

This includes properties, governing equations and the linearized system
"""
function update_state_dependents!(storage, model::TervModel, dt, forces)
    t_asm = @elapsed begin 
        update_properties!(storage, model)
        update_equations!(storage, model, dt)
        apply_forces!(storage, model, dt, forces)
    end
    @debug "Assembled equations in $t_asm seconds."
end

function update_properties!(storage, model)
    # No default properties
end

function update_equations!(storage, model, dt = nothing)
    equations = storage.equations
    for key in keys(equations)
        update_equation!(storage, model, equations[key], dt)
    end
end

function update_linearized_system!(storage, model::TervModel)
    equations = storage.equations
    lsys = storage.LinearizedSystem
    for key in keys(equations)
        update_linearized_system_subset!(lsys, model, equations[key])
    end
end

"""
Apply a set of forces to all equations. Equations that don't support a given force
will just ignore them, thanks to the power of multiple dispatch.
"""
function apply_forces!(storage, model::TervModel, dt, forces::NamedTuple)
    equations = storage.equations
    for key in keys(equations)
        eq = equations[key]
        for fkey in keys(forces)
            force = forces[fkey]
            apply_forces_to_equation!(storage, model, eq, force)
        end
    end
end

function apply_forces!(storage, model, dt, ::Nothing)

end

function setup_parameters(model)
    return Dict{Symbol, Any}()
end

function build_forces(model::TervModel)
    return NamedTuple()
end
