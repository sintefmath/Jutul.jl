# Manual sparsity version
function setup_equation_storage(model, e::ScalarTestEquation{ManualTestDisc}, storage; kwarg...)
    Ω = model.domain
    nc = number_of_cells(Ω)
    @assert nc == 1 # We use nc for clarity of the interface - but it should always be one!
    ne = 1 # Single, scalar equation
    npartials = number_of_equations_per_entity(e)
    e = CompactAutoDiffCache(ne, nc, npartials, context = model.context; kwarg...)
    return e
end

function declare_pattern(model, e::ScalarTestEquation{ManualTestDisc}, eq_storage::CompactAutoDiffCache, unit)
    @assert unit == Cells()
    return ([1], [1])
end

function update_equation!(eq_s::CompactAutoDiffCache, eq::ScalarTestEquation{ManualTestDisc}, storage, model, dt)
    X = storage.state.XVar
    X0 = storage.state0.XVar
    equation = get_entries(eq_s)
    @. equation = (X - X0)/dt
end

function apply_forces_to_equation!(storage, model, eq::ScalarTestEquation{ManualTestDisc}, eq_s, force::ScalarTestForce, time)
    equation = get_entries(eq_s)
    @. equation -= force.value
end

