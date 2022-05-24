using Jutul
using LinearAlgebra
using Printf
using ForwardDiff
using CUDA
# Turn on debugging to show output and timing.
# Turn on by uncommenting or running the following:
ENV["JULIA_DEBUG"] = Jutul
# To disable the debug output:
# ENV["JULIA_DEBUG"] = nothing
casename = "pico"
# casename = "spe10"
# casename = "tarbert_layer"
# casename = "spe10_symrmc"
# casename = "tarbert"

target = "cuda"
# target = "cpukernel"
# target = "cpu"
CUDA.allowscalar(false)
begin
pvfrac=0.05
tstep = [1.0, 2.0]
@time G, mrst_data = get_minimal_tpfa_grid_from_mrst(casename, extraout = true)
println("Setting up simulation case.")
nc = number_of_cells(G)
nf = number_of_faces(G)
# Parameters
bar = 1e5
p0 = 100*bar # 100 bar
mu = 1e-3    # 1 cP
cl = 1e-5/bar
pRef = 100*bar
rhoLS = 1000.0
# Single-phase liquid system (compressible pressure equation)
phase = LiquidPhase()
sys = SinglePhaseSystem(phase)
# Simulation model wraps grid and system together with context (which will be used for GPU etc)
if target == "cuda"
    ctx = SingleCUDAContext()
    # ctx = SingleCUDAContext(Float64, Int64)
    linsolve = CuSparseSolver("??", 1e-3)
else
    ctx = DefaultContext()
    linsolve = AMGSolver("RugeStuben", 1e-3)
end
model = SimulationModel(G, sys, context = ctx)

# System state
timesteps = tstep*3600*24 # 1 day, 2 days
tot_time = sum(timesteps)
irate = pvfrac*sum(G.pv)/tot_time
src = sources = [SourceTerm(1, irate, fractional_flow = [1.0]), 
                    SourceTerm(nc, -irate)]
# State is dict with pressure in each cell
state0 = setup_state(model, p0)
# Model parameters


sim = Simulator(model, state0 = state0)

# linsolve = nothing
states = simulate(sim, timesteps, sources = src, linsolve = linsolve)
end
nothing
# display(sim.storage)
