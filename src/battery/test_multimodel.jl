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
function make_system(exported,sys,bcfaces,srccells)
    #name="model1d"
    #fn = string(dirname(pathof(Terv)), "/../data/models/", name, ".mat")
    #exported_all = MAT.matread(fn)
    #exported = exported_all["model"]["NegativeElectrode"]["CurrentCollector"];
    ## for boundary
    # bcfaces = [1]
    # bcfaces = Int64.(bfaces)
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
    λ = exported["thermalConductivity"][1]

    S = model.secondary_variables
    S[:BoundaryPhi] = BoundaryPotential{Phi}()
    S[:BoundaryC] = BoundaryPotential{Phi}()
    S[:BoundaryT] = BoundaryPotential{T}()

    S[:BCCharge] = BoundaryCurrent{ChargeAcc}(srccells)
    S[:BCMass] = BoundaryCurrent{MassAcc}(srccells)
    S[:BCEnergy] = BoundaryCurrent{EnergyAcc}(srccells)

    phi0 = 0.
    init = Dict(
        :Phi                    => phi0,
        :C                      => C0,
        :T                      => T0,
        :Conductivity           => σ,
        :Diffusivity            => D,
        :ThermalConductivity    => λ,
        :BoundaryPhi            => bcvaluephi, 
        :BoundaryC              => bcvaluephi, 
        :BoundaryT              => bcvaluephi,
        :BCCharge               => -bcvaluesrc.*9.4575,
        :BCMass                 => bcvaluesrc,
        :BCEnergy               => bcvaluesrc,
        )

    state0 = setup_state(model, init)
    return model, G, state0, parameters, init
end

# function update_cross_term!(ct::InjectiveCrossTerm, eq::ScalarTestEquation, target_storage, source_storage, target_model, source_model, target, source, dt)
#    X_T = target_storage.state.XVar
#    X_S = source_storage.state.XVar
#    function f(X_S, X_T)
#        X_T - X_S
#    end
#    # Source term with AD context from source model - will end up as off-diagonal block
#    @. ct.crossterm_source = f(X_S, value(X_T))
#    # Source term with AD context from target model - will be inserted into equation
#    @. ct.crossterm_target = f(value(X_S), X_T)
#end


function test_ac()
    name="model1d"
    fn = string(dirname(pathof(Terv)), "/../data/models/", name, ".mat")
    exported_all = MAT.matread(fn)
    exported_cc = exported_all["model"]["NegativeElectrode"]["CurrentCollector"];
    # sys = ECComponent()
    # sys = ACMaterial();
    #sys = Grafite()
    sys_cc = CurrentCollector()
    bcfaces=[1]
    srccells = []
    (model_cc, G_cc, state0_cc, parm_cc,init_cc) = make_system(exported_cc,sys_cc, bcfaces, srccells)
    #sys_nam = CurrentCollector()
    sys_nam = Grafite()
    exported_nam = exported_all["model"]["NegativeElectrode"]["ElectrodeActiveComponent"];
    # sys = ECComponent()
    bcfaces=[]
    srccells = [10]
    (model_nam, G_nam, state0_nam, parm_nam, init_nam) = 
        make_system(exported_nam,sys_nam,bcfaces,srccells)

    timesteps = diff(LinRange(0, 10, 10)[2:end])
    
    groups = nothing
    model = MultiModel((CC = model_cc, NAM = model_nam), groups = groups)
    init_cc[:BCCharge]  = 0.0  
    #init_nam[:BoundaryPhi] = -3e-6
    init = Dict(
        :CC => init_cc,
        :NAM => init_nam
    )
    # state0 = setup_state(model, Dict(:CC => state0_cc, :NAM => state0_nam))
    state0 = setup_state(model, init)
    forces = Dict(
        :CC => nothing,
        :NAM => nothing
    )
    parameters = Dict(
        :CC => parm_cc,
        :NAM => parm_nam
    )

    target = Dict( :model => :NAM,
                   :equation => :charge_conservation
    )
    source = Dict( :model => :CC,
                :equation => :charge_conservation)
    intersection = ( [10], [1], Cells(), Cells())
    crosstermtype = InjectiveCrossTerm
    issym = true
    coupling = MultiModelCoupling(source,target, intersection; crosstype = crosstermtype, issym = issym)
    push!(model.couplings,coupling)




    #parameters = setup_parameters(model) # Dict(:CC => parm_cc, :NAM => parm_cc))
    #forces = Dict(:CC => state0_cc, :NAM => state0_nam)
    sim = Simulator(model, state0 = state0,
         parameters = parameters, copy_state = true)
    cfg = simulator_config(sim)
    cfg[:linear_solver] = nothing
    states = simulate(sim, timesteps, forces = forces, config = cfg)
    stateref = exported_all["states"]
    grids = Dict(:CC => G_cc,
                :NAM =>G_nam
                )
    return states, grids, state0, stateref, parameters, init, exported_all
end


@time states, grids, state0, stateref, parameters, init, exported_all = test_ac();

## f= plot_interactive(G, states);
#fields = ["CurrentCollector","ElectrodeActiveMaterial"]

#phi_ref = stateref[10]["NegativeElectrode"]["CurrentCollector"]["phi"]
#j_ref = stateref[10]["NegativeElectrode"]["CurrentCollector"]["j"]
## set up for current collector first
G = grids[:CC]
x = G["cells"]["centroids"]
xf= G["faces"]["centroids"][end]
xfi= G["faces"]["centroids"][2:end-1]
#state0=states[1]
p1 = Plots.plot(x,state0[:CC][:Phi];title="Phi")
#Plots.plot!(p1,x,phi_ref;linecolor="red")
p2 = Plots.plot(xfi,states[1][:CC].TPkGrad_Phi[1:2:end-1];title="Flux",markershape=:circle)
p3 = Plots.plot(;title="C")
    #p2 = Plots.plot([xf[end]],[state0[:BCCharge][1]];title="Flux")
    #Plots.plot!(p2,xfi,j_ref;linecolor="red")
fields = ["CurrentCollector","ElectrodeActiveComponent"]
for field in fields
    G = exported_all["model"]["NegativeElectrode"][field]["G"]
    x = G["cells"]["centroids"]
    xf= G["faces"]["centroids"][end]
    xfi= G["faces"]["centroids"][2:end-1]
    #state0=states[1]
    state = stateref[10]["NegativeElectrode"]
    phi_ref = state[field]["phi"]
    j_ref = state[field]["j"]
    Plots.plot!(p1,x,phi_ref;linecolor="red")
    Plots.plot!(p2,xfi,j_ref;title="Flux",linecolor="red")
    if haskey(state,"c")
        c = state[field]["c"]
        Plots.plot!(p3,xfi,j_ref;title="Flux",linecolor="red")
    end
end
#display(plot!(p1, p2, layout = (1, 2), legend = false))
##
for key in keys(grids)
    G = grids[key]
    x = G["cells"]["centroids"]
    xf= G["faces"]["centroids"][end]
    xfi= G["faces"]["centroids"][2:end-1]     
    p=plot(p1, p2, layout = (1, 2), legend = false)
    Plots.plot!(p1,x,states[end][key].Phi;markershape=:circle)
    Plots.plot!(p2,xfi,states[end][key].TPkGrad_Phi[1:2:end-1];markershape=:circle)
    if(haskey(states[end][key],:C))
        cc=states[end][key].C
        Plots.plot!(p3,x,cc;markershape=:circle)
    end
    display(plot!(p1, p2,p3,layout = (3, 1), legend = false))
end
##
#for (n,state) in enumerate(states)
#        println(n)
#        Plots.plot!(p1,x,states[n].Phi)
#        Plots.plot!(p2,xfi,states[n].TPkGrad_Phi[1:2:end-1])
#        display(plot!(p1, p2, layout = (1, 2), legend = false))
#end
##