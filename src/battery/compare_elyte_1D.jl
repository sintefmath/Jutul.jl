#=
Compare the simulation of electrolyte in 1D in Julia and Matlab
=#
using Terv
using MAT

##

function test_simple_elyte_1d()
    name = "test_electrolyte1D"
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
    parameters[:tolerances][:default] = 1e-12
    parameters[:t] = ex_model["sp"]["t"][1]
    parameters[:z] = ex_model["sp"]["z"][1]

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
    states = simulate(sim, timesteps, config = cfg)

    rs = exported["states"][:, 1]
    return states, G, model, rs
end

states, G, model, rs = test_simple_elyte_1d();

##
# Transelate bw states from matlab and julia
j2m = Dict{Symbol, String}(
    :C                  => "cs",
    :T                  => "T",
    :Phi                => "phi", 
    :Conductivity       => "conductivity",
    :Diffusivity        => "D",
    :TotalCurrent       => "j",
    :ChargeCarrierFlux  => "LiFlux" 
)
ref_states = get_ref_states(j2m, rs);

##

using Plots
state0 = states[1]

x = G["cells"]["centroids"][:, 1]
xf = G["faces"]["centroids"][end]
xfi= G["faces"]["centroids"][2:10]

function print_diff(s, sref, n, k)
    Δ = abs.(1 .- s[n][k] ./ sref[n][k])
    println("k = $k, n = $n")
    println("rel.diff = $(maximum(Δ))")
end

function print_diff_j(s, sref, n, k)
    Δ = abs.(1 .- s[n][k][2:2:end] ./ (sref[n][k]*1.011))
    println("k = $k, n = $n")
    println("rel.diff = $(maximum(Δ))")
end


plot1 = Plots.plot([], []; title = "Phi", size=(1000, 800))
plot2 = Plots.plot([], []; title = "Current", size=(1000, 800))

p = plot!(plot1, plot2, layout = (1, 2), legend = false)
k1 = :Phi; k2 = :TotalCurrent

for (n, state) in enumerate(states[1:end])
    plot!(plot1, x, states[n][k1], color="red", m=:cross)
    scatter!(plot1, x, ref_states[n][k1], color="blue", m=:circle)
    print_diff(states, ref_states, n, k1)

    plot!(plot2, xfi, states[n][:TotalCurrent][2:2:end], color="red")
    # !OBS fudge factor somhow makes states almost equal
    scatter!(plot2, xfi, ref_states[n][:TotalCurrent] .* 1.011,color="blue", m=:circle)
    # print_diff_j(states, ref_states, n, k2)

    display(plot!(plot1, plot2, layout = (1, 2), legend = false))
end
closeall()

##

plot1 = Plots.plot(x, state0[:C]; title = "C", size=(1000, 800))
plot2 = Plots.plot([], [], title = "CC Flux", size=(1000, 800))

p = plot(plot1, plot2, layout = (1, 2), legend = false)
k1 = :C; k2 = :ChargeCarrierFlux

for (n, state) in enumerate(states[1:end])
    plot!(plot1, x, states[n][k1], color="red", m=:cross)
    scatter!(plot1, x, ref_states[n][k1], color="blue", m=:circle)
    print_diff(states, ref_states, n, k1)

    plot!(plot2, xfi, states[n][:TotalCurrent][2:2:end], color="red")
    # !OBS fudge factor somhow makes states almost equal
    scatter!(plot2, xfi, ref_states[n][:TotalCurrent] .* 1.011,color="blue", m=:circle)
    # print_diff_j(states, ref_states, n, k2)

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
