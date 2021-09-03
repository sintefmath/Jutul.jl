using Terv
using LinearAlgebra, Krylov
using Printf
using ForwardDiff
using CUDA, Test
# Turn on debugging to show output and timing.
# Turn on by uncommenting or running the following:
ENV["JULIA_DEBUG"] = Terv
# To disable the debug output:
# ENV["JULIA_DEBUG"] = nothing
# casename = "pico"
# casename = "spe10"
# casename = "tarbert_layer" 
# casename = "spe10_symrmc"
# casename = "tarbert"
 
# target = "cuda"
# target = "cpukernel"
# target = "cpu"
function test_single_phase_gpu(casename = "pico"; float_type = Float32, pvfrac=0.05, tstep = [1.0])#[1.0, 2.0])
    @time G, mrst_data = get_minimal_tpfa_grid_from_mrst(casename, extraout = true)
    println("Setting up simulation case.")
    nc = number_of_cells(G)
    # Parameters
    bar = 1e5
    p0 = 100*bar # 100 bar

    # Single-phase liquid system (compressible pressure equation)
    phase = LiquidPhase()
    sys = SinglePhaseSystem(phase)
    # Simulation model wraps grid and system together with context (which will be used for GPU etc)
    ctx = SingleCUDAContext(float_type)
    linsolve = CuSparseSolver("??", 1e-3)
    linsolve = GenericKrylov(Krylov.dqgmres, preconditioner = ILUZeroPreconditioner())
    model = SimulationModel(G, sys, context = ctx)

    # System state
    pv = model.domain.grid.pore_volumes
    timesteps = tstep*3600*24 # 1 day, 2 days
    tot_time = sum(timesteps)  
    irate = pvfrac*sum(pv)/tot_time
    src = [SourceTerm(1, irate), 
           SourceTerm(nc, -irate)]
    src = CuArray(src)
    forces = build_forces(model, sources = src)

    # State is dict with pressure in each cell
    init = Dict(:Pressure => p0)
    state0 = setup_state(model, init)
    # Model parameters
    parameters = setup_parameters(model)
    parameters[:reference_densities] = CuArray(float_type.([1.0]))

    # linsolve = nothing
    sim = Simulator(model, state0 = state0, parameters = parameters)
    info_level = 2
    debug_level = 2
    simulate(sim, timesteps, forces = forces, linear_solver = linsolve, 
            debug_level = debug_level, info_level = info_level, output_states = false)
    return true
end
if has_cuda_gpu()
    CUDA.allowscalar(false)

    @testset "GPU multiphase" begin
        @testset "Basic flow - single precision" begin
            @test test_single_phase_gpu(casename, float_type = Float32)
        end
        @testset "Basic flow - double precision" begin
            @test test_single_phase_gpu(casename, float_type = Float64)
        end
    end
end
