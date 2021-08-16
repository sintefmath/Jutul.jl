#=
Electro-Chemical component
A component with electric potential, concentration and temperature
The different potentials are independent (diagonal onsager matrix),
and conductivity, diffusivity is constant.
=#
@time using Terv
using MAT
using Plots
ENV["JULIA_DEBUG"] = Terv;


##
function make_system(exported, sys, bcfaces, srccells)

    T_all = exported["operators"]["T_all"]
    N_all = Int64.(exported["G"]["faces"]["neighbors"])
    isboundary = (N_all[bcfaces,1].==0) .| (N_all[bcfaces,2].==0)
    @assert all(isboundary)
    bccells = N_all[bcfaces,1] + N_all[bcfaces,2]
    T_hf   = T_all[bcfaces]
    bcvaluesrc = ones(size(srccells))
    bcvaluephi = ones(size(bccells)).*0.0

    domain = exported_model_to_domain(exported, bc = bccells, b_T_hf = T_hf)  
    G = exported["G"]    
    model = SimulationModel(domain, sys, context = DefaultContext())
    parameters = setup_parameters(model)
    parameters[:boundary_currents] = (:BCCharge, :BCMass)

    # State is dict with pressure in each cell
    phi0 = 1.0
    C0 = 1.
    T0 = 1.
    D = 1.
    σ = exported["EffectiveElectricalConductivity"][1]

    S = model.secondary_variables
    S[:BoundaryPhi] = BoundaryPotential{Phi}()
    S[:BoundaryC] = BoundaryPotential{C}()
 
    S[:BCCharge] = BoundaryCurrent{Charge}(srccells)
    S[:BCMass] = BoundaryCurrent{Mass}(srccells)
 
    phi0 = 0.
    init = Dict(
        :Phi                    => phi0,
        :C                      => C0,
        :T                      => T0,
        :Conductivity           => σ,
        :Diffusivity            => D,
        :BoundaryPhi            => bcvaluephi, 
        :BoundaryC              => bcvaluephi, 
        :BCCharge               => -bcvaluesrc.*9.4575,
        :BCMass                 => bcvaluesrc,
        )

    state0 = setup_state(model, init)
    return model, G, state0, parameters, init
end


function test_ac()
    name="model1d"
    fn = string(dirname(pathof(Terv)), "/../data/models/", name, ".mat")
    exported_all = MAT.matread(fn)
    exported_cc = exported_all["model"]["NegativeElectrode"]["CurrentCollector"];

    sys_cc = CurrentCollector()
    bcfaces=[1]
    srccells = []
    (model_cc, G_cc, state0_cc, parm_cc,init_cc) = 
        make_system(exported_cc,sys_cc, bcfaces, srccells)
    sys_nam = Grafite()
    exported_nam = exported_all["model"]["NegativeElectrode"]["ElectrodeActiveComponent"];

    bcfaces=[]
    srccells = [10]
    (model_nam, G_nam, state0_nam, parm_nam, init_nam) = 
        make_system(exported_nam, sys_nam, bcfaces, srccells)

    timesteps = diff(LinRange(0, 10, 10)[2:end])
    
    groups = nothing
    model = MultiModel((CC = model_cc, NAM = model_nam), groups = groups)
    init_cc[:BCCharge]  = 0.0  
    init = Dict(
        :CC => init_cc,
        :NAM => init_nam
    )

    state0 = setup_state(model, init)
    forces = Dict(
        :CC => nothing,
        :NAM => nothing
    )
    parameters = Dict(
        :CC => parm_cc,
        :NAM => parm_nam
    )

    target = Dict(:model => :NAM, :equation => :charge_conservation)
    source = Dict(:model => :CC,:equation => :charge_conservation)
    intersection = ([10], [1], Cells(), Cells())
    crosstermtype = InjectiveCrossTerm
    issym = true
    coupling = 
        MultiModelCoupling(source,target, intersection; crosstype = crosstermtype, issym = issym)
    push!(model.couplings,coupling)

    sim = Simulator(model, state0 = state0,
         parameters = parameters, copy_state = true)
    cfg = simulator_config(sim)
    cfg[:info_level] = 2
    cfg[:debug_level] = 2
    cfg[:linear_solver] = nothing
    states, report = simulate(sim, timesteps, forces = forces, config = cfg)
    stateref = exported_all["states"]
    grids = Dict(:CC => G_cc,
                :NAM =>G_nam
                )
    return states, grids, state0, stateref, parameters, init, exported_all
end


@time states, grids, state0, stateref, parameters, init, exported_all = test_ac();

##

G = grids[:CC]
x = G["cells"]["centroids"]
xf = G["faces"]["centroids"][end]
xfi = G["faces"]["centroids"][2:end-1]
p1 = Plots.scatter([], []; title="Phi", size=(1000, 500), lw = 3)
p2 = Plots.scatter([], []; title="Flux", size=(1000, 500), lw=3)
p3 = Plots.plot(title="C", size=(1000, 500))
fields = ["CurrentCollector", "ElectrodeActiveComponent"]

for key in keys(grids)
    G = grids[key]
    x = G["cells"]["centroids"]
    xf= G["faces"]["centroids"][end]
    xfi= G["faces"]["centroids"][2:end-1]     
    p=plot(p1, p2, layout = (1, 2), legend = false)
    Plots.scatter!(p1, x, states[end][key].Phi)
    Plots.scatter!(p2, xfi, states[end][key].TPkGrad_Phi[1:2:end-1])
    if(haskey(states[end][key], :C))
        cc=states[end][key].C
        Plots.scatter!(p3, x, cc)
    end
end

for field in fields
    G = exported_all["model"]["NegativeElectrode"][field]["G"]
    x = G["cells"]["centroids"]
    xf= G["faces"]["centroids"][end]
    xfi= G["faces"]["centroids"][2:end-1]

    state = stateref[10]["NegativeElectrode"]
    phi_ref = state[field]["phi"]
    j_ref = state[field]["j"]
    Plots.plot!(p1, x, phi_ref; linecolor="black", lw=3)
    Plots.plot!(p2, xfi, j_ref; title="Flux", linecolor="black", lw=3)
    if haskey(state,"c")
        c = state[field]["c"]
        Plots.plot!(p3, xfi, j_ref; title="Flux", linecolor="red")
    end
end
display(plot!(p1, p2, p3, layout = (3, 1), legend = false))


closeall()