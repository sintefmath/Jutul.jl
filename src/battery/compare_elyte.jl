#=
Compare the simulation of electrolyte in Julia and Matlab
=#

using Terv

function test_simple_elyte()
    name="modelElectrolyte"
    fn = string(dirname(pathof(Terv)), "/../data/models/", name, ".mat")
    exported = MAT.matread(fn)
    model = exported["model"]
    boundary = model["bcstruct"]["dirichlet"]

    b_phi = boundary["phi"][:, 1]
    b_c = boundary["conc"][:, 1]
    b_faces = Int64.(boundary["faces"])

    T_all = model["operators"]["T_all"]
    N_all = Int64.(model["G"]["faces"]["neighbors"])
    isboundary = (N_all[b_faces, 1].==0) .| (N_all[b_faces, 2].==0)
    @assert all(isboundary)
    bc_cells = N_all[b_faces,1] + N_all[b_faces,2]
    b_T_hf   = T_all[b_faces]

    domain = exported_model_to_domain(model, bc=bc_cells, b_T_hf=b_T_hf)

    #? Er value riktig?
    timesteps = exported["schedule"]["step"]["val"][:, 1]
    
    G = model["G"]
    sys = SimpleElyte()
    model = SimulationModel(domain, sys, context = DefaultContext())
    parameters = setup_parameters(model)

    S = model.secondary_variables
    S[:BoundaryPhi] = BoundaryPotential{Phi}()
    S[:BoundaryC] = BoundaryPotential{C}()

    state0 = exported["state0"]
    init = Dict(
          :Phi                    => state0["phi"][:, 1],
          :C                      => state0["cs"][1][:, 1], # ? Why is this different?
          :T                      => state0["T"][:, 1],
          :BoundaryPhi            => b_phi,
          :BoundaryC              => b_c,
          )
  
      state0 = setup_state(model, init)
  
      sim = Simulator(model, state0=state0, parameters=parameters)
      cfg = simulator_config(sim)
      cfg[:linear_solver] = nothing
      states = simulate(sim, timesteps, config = cfg)
      return states, G
  end

states, G = test_simple_elyte();

##
f = plot_interactive(G, states)
display(f)
