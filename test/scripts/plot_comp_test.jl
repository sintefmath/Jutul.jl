name = "simple_compositional_fake_wells";
name = "compositional_three_phases";

function solve(grid, setup; linear_solver = nothing, kwarg...)
    state0, model, prm, f, t = get_test_setup(grid, case_name = setup; kwarg...)
    push!(model.output_variables, :Saturations)
    sim = Simulator(model, state0 = state0, parameters = prm)
    if linear_solver == :auto
        linear_solver = reservoir_linsolve(model)
    end
    cfg = simulator_config(sim, info_level = 0, linear_solver = linear_solver)
    states, reports = simulate(sim, t, forces = f, config = cfg)
    return states, reports, sim
end
##
grid = CartesianMesh((10,1), (10000.0, 1.0))
dt = repeat([0.001], 10000)
# dt = [0.001]
states, reports, sim = solve(grid, name, timesteps = dt);
##
using Plots
ix = length(states)
#ix = 1
plot(states[ix][:OverallMoleFractions]', title = "Mole fractions")
##
plot(states[ix][:Saturations]', title = "Saturations")
##
plot(states[ix][:Pressure]./1e5, title = "Pressure")