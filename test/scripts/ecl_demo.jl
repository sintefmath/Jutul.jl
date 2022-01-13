using Jutul
using Statistics, DataStructures, LinearAlgebra, Krylov, IterativeSolvers
# ENV["JULIA_DEBUG"] = Jutul
ENV["JULIA_DEBUG"] = nothing

casename = "sleipner_ecl_bo"
# casename = "olympus_simple"
models, parameters, initializer, timesteps, forces = setup_case_from_mrst(casename, block_backend = true);
##

# Reservoir as first group
groups = repeat([2], length(models))
groups[1] = 1

outer_context = DefaultContext()

red = :schur_apply
mmodel = MultiModel(convert_to_immutable_storage(models), groups = groups, context = outer_context, reduction = red)
# Set up joint state and simulate
state0 = setup_state(mmodel, initializer)

dt = timesteps[1:10]
dt = timesteps[1:1].*0.01
# dt = timesteps
# Set up linear solver and preconditioner
lsolve = reservoir_linsolve(mmodel.models[:Reservoir], :cpr)
m = 20
il = 1
dl = 0
# Simulate
sim = Simulator(mmodel, state0 = state0, parameters = deepcopy(parameters))
cfg = simulator_config(sim, info_level = il, debug_level = dl,
                            max_nonlinear_iterations = m,
                            output_states = true,
                            linear_solver = lsolve)
t_base = TimestepSelector(initial_absolute = 1*24*3600.0)
t_its = IterationTimestepSelector()

cfg[:timestep_selectors] = [t_base, t_its]
states, reports = simulate(sim, dt, forces = forces, config = cfg);
error()
## Plotting
res_states = map((x) -> x[:Reservoir], states)
g = MRSTWrapMesh(mrst_data["G"])

fig, ax = plot_interactive(g, res_states, colormap = :roma)
w_raw = mrst_data["W"]
for w in w_raw
    if w["sign"] > 0
        c = :midnightblue
    else
        c = :firebrick
    end
    plot_well!(ax, g, w, color = c, textscale = 0*5e-2)
end
display(fig)