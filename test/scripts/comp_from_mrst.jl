using Jutul
using Statistics, DataStructures, LinearAlgebra, Krylov, IterativeSolvers, MultiComponentFlash
ENV["JULIA_DEBUG"] = nothing

casename = "1d_validation"
casename = "1d_validation_water"
# casename = "1d_validation_water_mini"

# casename = "fractures_compositional"
function bench(casename)
    block = true
    # block = false
    sw = true
    sw = false
    mb = 250
    models, parameters, initializer, timesteps, forces, mrst_data = setup_case_from_mrst(casename, block_backend = block, simple_well = sw, minbatch = mb);
    ##
    push!(models[:Reservoir].output_variables, :Saturations)
    # Reservoir as first group
    outer_context = DefaultContext()
    if block
        groups = repeat([2], length(models))
        groups[1] = 1

        red = :schur_apply
    else
        red = nothing
        groups = nothing
    end
    # sys = models[:Reservoir].system
    # models[:Reservoir].secondary_variables[:FlashResults] = Terv.FlashResults(sys, method = SSIFlash())
    mmodel = MultiModel(convert_to_immutable_storage(models), groups = groups, context = outer_context, reduction = red)
    # Set up joint state and simulate
    state0 = setup_state(mmodel, initializer)
    dt = timesteps

    # dt = timesteps[1:10]
    # dt = timesteps[1:1].*0.001
    # dt = timesteps[[1]]
    # Set up linear solver and preconditioner
    lsolve = reservoir_linsolve(mmodel, :cpr, verbose = false)
    m = 20
    # m = 2
    # m = 5
    il = 0
    # il = 5
    dl = 0
    # max_cuts = 0
    max_cuts = 6
    # Simulate
    extra_timing = true
    # extra_timing = false
    sim = Simulator(mmodel, state0 = state0, parameters = deepcopy(parameters))
    cfg = simulator_config(sim, info_level = il, debug_level = dl,
                                max_nonlinear_iterations = m,
                                output_states = true,
                                extra_timing = extra_timing,
                                max_timestep_cuts = max_cuts,
                                linear_solver = lsolve)
    t_base = TimestepSelector(initial_absolute = 1*24*3600.0)
    t_its = IterationTimestepSelector()

    cfg[:timestep_selectors] = [t_base, t_its]
    # error()
    res = simulate(sim, dt, forces = forces, config = cfg);
    return (res[1], res[2], sim, mrst_data)
end
states, reports, sim, mrst_data = bench(casename);
error("All done.")
##
using GLMakie
k = :OverallMoleFractions
k = :Saturations
getdata = x -> x[:Reservoir][k]
z0 = getdata(states[180])

plotvals = Observable(z0)
f = Figure()
ax = Axis(f[1, 1])
series!(plotvals, color = :Set1)
GLMakie.ylims!(0, 1)
f
##
for i = 1:length(states)
    plotvals[] = getdata(states[i])
    sleep(0.01)
end
##
using GLMakie
res_states = map((x) -> x[:Reservoir], states)
g = MRSTWrapMesh(mrst_data["G"])

fig, ax = plot_interactive(g, res_states, colormap = :turbo)
w_raw = mrst_data["W"]
for w in w_raw
    if w["sign"] > 0
        c = :midnightblue
    else
        c = :firebrick
    end
    # plot_well!(ax, g, w, color = c, textscale = 1*5e-2)
end
display(fig) 
##