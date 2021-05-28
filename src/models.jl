export update_state_dependents!, check_convergence

function get_primary_variables(model::SimulationModel)
    return model.primary_variables
end

function get_secondary_variables(model::SimulationModel)
    return model.secondary_variables
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
        initialize_variable_value!(state, model, pvar, init_values)
    end
    add_extra_state_fields!(state, model)
end

"""
Add variables that need to be in state, but are never AD variables (e.g. phase status flag)
"""
function add_extra_state_fields!(state, model::TervModel)
    # Do nothing
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
function allocate_storage!(storage, model::TervModel; setup_linearized_system = true,
                                                      state0 = setup_state(model),
                                                      parameters = setup_parameters(model),
                                                      tag = nothing,
                                                      kwarg...)
    if !isnothing(state0)
        storage[:parameters] = parameters
        storage[:state0] = state0
        storage[:primary_state] = convert_state_ad(model, state0, tag)
    end
    storage[:state] = allocate_state(storage, model; tag = tag, kwarg...) 
    storage[:equations] = allocate_equations(storage, model; tag = tag, kwarg...) 
    if setup_linearized_system
        storage[:LinearizedSystem] = allocate_linearized_system!(storage, model)
        # We have the equations and the linearized system.
        # Give the equations a chance to figure out their place in the Jacobians.
        align_equations_to_linearized_system!(storage, model)
    end
end

function allocate_state(storage, model::TervModel; kwarg...)
    props = Dict()
    allocate_secondary_variables!(props, storage, model; kwarg...)
    if haskey(storage, :primary_state)
        # Add references to primary variables. The state functions
        # should not be able to tell the difference if another state function
        # is a primary variable or a dependent variable.
        for (key, val) in storage[:primary_state]
            props[key] = val
        end
    end
    return props
end

function allocate_equations(storage, model::TervModel; kwarg...)
    # We use ordered dict since equation alignment with primary variables matter.
    eqs = OrderedDict()
    allocate_equations!(eqs, storage, model; kwarg...)
    return eqs
end

function allocate_equations!(eqs, storage, model::TervModel; tag = nothing)
    # Default: No equations.
end

function get_sparse_arguments(storage, model, layout = matrix_layout(model.context))
    ndof = number_of_degrees_of_freedom(model)
    eqs = storage[:equations]
    I = []
    J = []
    nrows = 0
    for (k, eq) in eqs
        for u in get_units(model.domain)
            S = declare_sparsity(model, eq, u, layout)
            if !isnothing(S)
                i = S[1]
                j = S[2]
                push!(I, i .+ nrows) # Row indices, offset by the size of preceeding equations
                push!(J, j)          # Column indices
                nrows += S[3]
            end
            # TODO: Fix offset for multiple units with primary variables
        end
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
    eqs = storage[:equations]
    jac = storage[:LinearizedSystem].jac
    align_equations_to_jacobian!(eqs, jac, model; kwarg...)
end

function align_equations_to_jacobian!(equations, jac, model; row_offset = 0, col_offset = 0)
    for key in keys(equations)
        eq = equations[key]
        align_to_jacobian!(eq, jac, model, row_offset = row_offset, col_offset = col_offset)
        row_offset += number_of_equations(model, eq)
    end
    row_offset
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
        update_secondary_variables!(storage, model)
        update_equations!(storage, model, dt)
        apply_forces!(storage, model, dt, forces)
    end
    @debug "Assembled equations in $t_asm seconds."
end

function update_equations!(storage, model, dt = nothing)
    equations = storage.equations
    for key in keys(equations)
        update_equation!(equations[key], storage, model, dt)
    end
end

function update_linearized_system!(storage, model::TervModel; kwarg...)
    equations = storage.equations
    lsys = storage.LinearizedSystem
    update_linearized_system!(lsys, equations, model; kwarg...)
end

function update_linearized_system!(lsys, equations, model::TervModel; row_offset = 0)
    for key in keys(equations)
        eq = equations[key]
        n = number_of_equations(model, eq)
        update_linearized_system_subset!(lsys, model, eq, r_subset = (row_offset+1):(row_offset+n))
        row_offset += n
    end
end

function check_convergence(storage, model; kwarg...)
    lsys = storage.LinearizedSystem
    eqs = storage.equations
    check_convergence(lsys.r, eqs, storage, model; kwarg...)
end

function check_convergence(r, eqs, storage, model; iteration = nothing, extra_out = false, tol = 1e-3, kwarg...)
    converged = true
    e = 0
    offset = 0
    for key in keys(eqs)
        eq = eqs[key]
        n = number_of_equations(model, eq)
        r_v = view(r, (1:n) .+ offset)
        errors, tscale = convergence_criterion(model, storage, eq, r_v; kwarg...)
        for (index, e) in enumerate(errors)
            s = @sprintf("It %d: |%s_%d| = %e\n", iteration, String(key), index, e)
            @debug s
        end
        converged = converged && all(errors .< tol*tscale)
        e = maximum([e, maximum(errors)])
        offset += n
    end
    if extra_out
        return (converged, e, tol)
    else
        return converged
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

function solve_update!(storage, model::TervModel; linsolve = nothing)
    lsys = storage.LinearizedSystem
    t_solve = @elapsed solve!(lsys, linsolve)
    t_update = @elapsed update_primary_variables!(storage, model)
    return (t_solve, t_update)
end

function update_primary_variables!(storage, model::TervModel)
    dx = storage.LinearizedSystem.dx
    update_primary_variables!(storage.primary, dx, model)
end

function update_primary_variables!(primary_storage, dx, model::TervModel)
    offset = 0
    primary = get_primary_variables(model)
    for p in primary
        n = number_of_degrees_of_freedom(model, p)
        rng = (1:n) .+ offset
        update_primary_variable!(primary_storage, p, model, view(dx, rng))
        offset += n
    end
end

function update_after_step!(storage, model::TervModel)
    state = storage.state
    state0 = storage.state0
    for key in keys(state0)
        @. state0[key] = value(state[key])
    end
end

function get_output_state(storage, model::TervModel)
    # As this point (after a converged step) state0 should be state without AD.
    return deepcopy(storage.state0)
end
