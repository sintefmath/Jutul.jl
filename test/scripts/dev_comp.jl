@time using Terv
@time using MultiComponentFlash
ENV["JULIA_DEBUG"] = Terv
# Pkg.develop(PackageSpec(path = "D:/jobb\\bitbucket\\moyner.jl\\pvt\\MultiComponentFlash"))
##
# Multicomponent flow
# Variables:
# T as secondary variable (constant)
# z as primary variable

co2 = MolecularProperty(0.0440, 7.38e6, 304.1, 9.412e-5, 0.224)
c1 = MolecularProperty(0.0160, 4.60e6, 190.6, 9.863e-5, 0.011)
c10 = MolecularProperty(0.0142, 2.10e6, 617.7, 6.098e-4, 0.488)

mixture = MultiComponentMixture([co2, c1, c10], names = ["CO2", "C1", "C10"])
##
m = SSIFlash()
z0 = [0.5, 0.3, 0.2]
zi = [0.99, 0.01, 0.0]
p0 = 10e5
T0 = 300.0
n = length(z0)


eos = GenericCubicEOS(mixture)
##
nc = 5
G = get_1d_reservoir(nc)
nc = number_of_cells(G)
# Definition of fluid phases
rhoLS, rhoVS = 1000.0, 100.0
L, V = LiquidPhase(), VaporPhase()
# Define system and realize on grid
sys = TwoPhaseCompositionalSystem([L, V], eos)
ctx = DefaultContext(matrix_layout = BlockMajorLayout())
ctx = DefaultContext()
model = SimulationModel(G, sys, context = ctx)
model.primary_variables[:Pressure] = Pressure(max_rel = 0.2, minimum = 1e5, )
p = zeros(nc)
p .= p0
# p[1] *= 10
# p[end] /= 10

state0 = setup_state(model, Dict(:Pressure => p0, :OverallCompositions => z0))
tstep = [1.0]
timesteps = tstep*3600*24 # Convert time-steps from days to seconds
# Simulate and return

tot_time = sum(timesteps)
pvfrac = 1.0
pv = G.grid.pore_volumes
irate = pvfrac*sum(pv)/tot_time
src = [SourceTerm(1, irate, fractional_flow = zi), 
       SourceTerm(nc, -irate, fractional_flow = zi)]
forces = build_forces(model, sources = src)

sim = Simulator(model, state0 = state0)
states, report = simulate(sim, timesteps, info_level = 3, forces = forces)
 