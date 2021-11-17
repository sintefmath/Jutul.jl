#=
Compare the simulation of electrolyte in 1D in Julia and Matlab
=#

##

function test_simple_elyte_1d()
    name = "model1d_"
    fn = string(dirname(pathof(Terv)), "/../data/models/", name, ".mat")
    exported = MAT.matread(fn)
    ex_model = exported["model"]
    boundary = ex_model["bcstruct"]["dirichlet"]

    b_phi = boundary["phi"][:, 1]
    b_c = boundary["conc"][:, 1]
    b_faces = Int64.(boundary["faces"])

    T_all = ex_model["operators"]["T_all"]
    N_all = Int64.(ex_model["G"]["faces"]["neighbors"])
    isboundary = (N_all[b_faces, 1].==0) .| (N_all[b_faces, 2].==0)
    @assert all(isboundary)
    bc_cells = N_all[b_faces,1] + N_all[b_faces,2]
    b_T_hf   = T_all[b_faces] 

    domain = exported_model_to_domain(ex_model, bc=bc_cells, b_T_hf=b_T_hf)

    timesteps = exported["schedule"]["step"]["val"][:, 1]
    
    G = ex_model["G"]
    sys = SimpleElyte()
    model = SimulationModel(domain, sys, context = DefaultContext())
    parameters = setup_parameters(model)
    parameters[:tolerances][:default] = 1e-8
    t1, t2 = exported["model"]["sp"]["t"]
    z1, z2 = exported["model"]["sp"]["z"]
    tDivz_eff = (t1/z1 + t2/z2)
    parameters[:t] = tDivz_eff
    parameters[:z] = 1

    S = model.secondary_variables
    S[:BoundaryPhi] = BoundaryPotential{Phi}()
    S[:BoundaryC] = BoundaryPotential{C}()

    init_states = exported["state0"]
    init = Dict(
        :Phi            => init_states["phi"][:, 1],
        :C              => init_states["cs"][1][:, 1],
        :T              => init_states["T"][:, 1],
        :BoundaryPhi    => b_phi,
        :BoundaryC      => b_c,
        )

    state0 = setup_state(model, init)
  
    sim = Simulator(model, state0=state0, parameters=parameters)
    cfg = simulator_config(sim)
    cfg[:linear_solver] = nothing
    cfg[:debug_level] = 2
    cfg[:info_level] = 2
    states, report = simulate(sim, timesteps, config = cfg)

    rs = exported["states"][:, 1]
    return states, G, model, rs, exported
end

states, G, model, rs, exported = test_simple_elyte_1d();


# Transelate bw states from matlab and julia
j2m = Dict{Symbol, String}(
    :C                  => "cs",
    :T                  => "T",
    :Phi                => "phi", 
    :Conductivity       => "conductivity",
    :Diffusivity        => "D",
    :TotalCurrent       => "j",
    :ChargeCarrierFlux  => "LiFlux" # This is not correct - CC is more than Li
)
ref_states = get_ref_states(j2m, rs);
println(states[end][:Phi] - ref_states[end][:Phi])

##

using Plots
state0 = states[1]

x = G["cells"]["centroids"][:, 1]
xf = G["faces"]["centroids"][end]
xfi= G["faces"]["centroids"][2:10]

function print_diff(s, sref, n, k)
    Δ = abs.(1 .-sref[n][k] ./ s[n][k])
    println("k = $k, n = $n")
    println("rel.diff = $(maximum(Δ))")
end

function print_diff_j(s, sref, n, k)
    Δ = abs.(1 .- (sref[n][k]) ./ s[n][k][2:2:end])
    println("k = $k, n = $n")
    println("rel.diff = $(maximum(Δ))")
end


plot1 = Plots.plot([], []; title = "Phi", size=(1000, 800))
plot2 = Plots.plot([], []; title = "Current", size=(1000, 800))

p = plot!(plot1, plot2, layout = (1, 2), legend = false)
k1 = :Phi; k2 = :TotalCurrent

for (n, state) in enumerate(states)
    plot!(plot1, x, states[n][k1], color="red", m=:cross)
    scatter!(plot1, x, ref_states[n][k1], color="blue", m=:circle)
    print_diff(states, ref_states, n, k1)

    plot!(plot2, xfi, states[n][k2][2:2:end], color="red")
    scatter!(plot2, xfi, ref_states[n][k2], color="blue", m=:circle)
    print_diff_j(states, ref_states, n, k2)

    display(plot!(plot1, plot2, layout = (1, 2), legend = false))
end
closeall()

##

plot1 = Plots.plot(x, state0[:C]; title = "C", size=(1000, 800))
plot2 = Plots.plot([], [], title = "CC Flux", size=(1000, 800))

p = plot(plot1, plot2, layout = (1, 2), legend = false)
k1 = :C; k2 = :ChargeCarrierFlux

for (n, state) in enumerate(states)
    plot!(plot1, x, states[n][k1], color="red", m=:cross)
    scatter!(plot1, x, ref_states[n][k1], color="blue", m=:circle)
    print_diff(states, ref_states, n, k1)

    plot!(plot2, xfi, states[n][k2][2:2:end], color="red")
    scatter!(plot2, xfi, ref_states[n][k2] ,color="blue", m=:circle)
    print_diff_j(states, ref_states, n, k2)

    display(plot!(plot1, plot2, layout = (1, 2), legend = false))
end

closeall()

##

plot1 = Plots.plot([], []; title = "κ", size=(1000, 800))

p = plot!(plot1, layout = (1,), legend = false)
k = :Conductivity
for (n, state) in enumerate(states[1:end])
    plot!(plot1, x, states[n][k], color="red", m=:cross)
    scatter!(plot1, x, ref_states[n][k], color="blue", m=:circle)
    print_diff(states, ref_states, n, k)

    display(plot!(plot1, layout = (1,), legend = false))
end
closeall()

##
plot1 = Plots.plot([], []; title = "D", size=(1000, 800))

p = plot!(plot1, layout = (1,), legend = false)
k = :Diffusivity
for (n, state) in enumerate(states[1:end])
    plot!(plot1, x, states[n][k], color="red", m=:cross)
    scatter!(plot1, x, ref_states[n][k], color="blue", m=:circle)
    print_diff(states, ref_states, n, k)

    display(plot!(plot1, layout = (1,), legend = false))
end
closeall()
