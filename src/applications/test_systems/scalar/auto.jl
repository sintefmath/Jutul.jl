function update_equation_in_entity!(v, i, state, state0, eq::ScalarTestEquation{AutoTestDisc}, model, dt, ldisc = local_discretization(eq, i))
    X = state.XVar
    X0 = state0.XVar
    return @. v = (X - X0) / dt
end
