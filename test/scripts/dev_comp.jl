@time using Jutul
@time using MultiComponentFlash
ENV["JULIA_DEBUG"] = Jutul
ENV["JULIA_DEBUG"] = nothing
# using Pkg; Pkg.develop(PackageSpec(path = "D:/jobb\\bitbucket\\moyner.jl\\pvt\\MultiComponentFlash"))
##
# Multicomponent flow
# Variables:
# T as secondary variable (constant)
# z as primary variable

co2 = MolecularProperty(0.0440, 7.38e6, 304.1, 9.412e-5, 0.224)
c1 = MolecularProperty(0.0160, 4.60e6, 190.6, 9.863e-5, 0.011)
c10 = MolecularProperty(0.0142, 2.10e6, 617.7, 6.098e-4, 0.488)

##
ϵ = 0.01
ϵ = 0
m = SSIFlash()
if true
       z0 = [0.5, 0.3, 0.2]
       zi = [0.99, 0.01-1e-3, 1e-3]
       zi = [1.0, 0, 0]
       mixture = MultiComponentMixture([co2, c1, c10], names = ["CO2", "C1", "C10"])
elseif true
       z0 = [0.001, 0.001, 1.0-0.002]
       zi = [0.98, 0.01, 0.01]
       mixture = MultiComponentMixture([co2, c1, c10], names = ["CO2", "C1", "C10"])
elseif false
       z0 = [0.5, 0.5]
       zi = [1-ϵ, ϵ]
       mixture = MultiComponentMixture([co2, c10], names = ["CO2", "C10"])
else
       z0 = [0.5, 0.5]
       zi = [1-ϵ, ϵ]
       mixture = MultiComponentMixture([c10, c10], names = ["CO2", "C10"])
end
p0 = 10e5
T0 = 300.0
n = length(z0)
 
eos = GenericCubicEOS(mixture)
##
nc = 10000
nc = 100
# G = get_1d_reservoir(nc)
mesh = CartesianMesh((nc, 1), (100.0, 1.0))
# nc = 50
# mesh = CartesianMesh((nc, nc), (100.0, 100.0))
geo = tpfv_geometry(mesh)
G = discretized_domain_tpfv_flow(geo)


nc = number_of_cells(G)
# Definition of fluid phases
rhoLS, rhoVS = 1000.0, 100.0
L, V = LiquidPhase(), VaporPhase()
# Define system and realize on grid
sys = MultiPhaseCompositionalSystemLV(eos, (L, V))
ctx = DefaultContext(matrix_layout = BlockMajorLayout())
ctx = DefaultContext()
model = SimulationModel(G, sys, context = ctx)
parameters = setup_parameters(model)
parameters[:reference_densities] .= 500.0
# pvar = Pressure(max_rel = 0.2, minimum = 1e5, scale = 1.0)
pvar = Pressure(minimum = 0.0, scale = 1.0, max_rel = 0.25)
model.primary_variables[:Pressure] = pvar

state0 = setup_state(model, Dict(:Pressure => p0, :OverallMoleFractions => z0))
tstep = ones(100)
timesteps = tstep*3600*24 # Convert time-steps from days to seconds
# Simulate and return

tot_time = sum(timesteps)
pvfrac = 1.0
pvfrac = 1e-1
pvfrac = 0.4
# pvfrac = 0.0
pv = G.grid.pore_volumes
irate = pvfrac*sum(pv)/tot_time
src = [SourceTerm(1, irate, fractional_flow = zi, type = StandardVolumeSource), 
       SourceTerm(nc, -irate, fractional_flow = zi, type = StandardVolumeSource)]
forces = build_forces(model, sources = src)
# forces = nothing
sim = Simulator(model, state0 = state0, parameters = parameters)
states, report = simulate(sim, timesteps, info_level = 1, forces = forces, max_timestep_cuts = 0, max_nonlinear_iterations = 10);
##
plot_interactive(mesh, states)
##
using GLMakie
x = 1:nc
lines(x, states[end][:Pressure] ./ 1e5)
##
z = states[10][:OverallMoleFractions]
f = lines(x, vec(z[1, :]))
for i = 2:size(z, 1)
    lines!(x, vec(z[i, :]))
end
f
##
cond = (p = 1.1293247790198943e7, T = 303.15, z = [0.1549999544005748, 0.8450000455026254, 9.679976144658433e-11])
flash_2ph(eos, cond)

##
cond = (p = 1.1293247790198943e7, T = 303.15, z = [0.1549999544005748, 0.8450000455026254, 9.679976144658433e-11])
flash_2ph(eos, cond)
##
cond = (p = p0, T = 303.15, z = z0)
flash_2ph(eos, cond)

##
# d = rand(10, 10)
d = ones(10, 10)
values = Node(d)
f = Figure()
ax = Axis(f[1, 1])
heatmap!(ax, values)
Colorbar(f[1, 2]) 
f
##
values[] = rand(10, 10)
##
values[] = ones(10, 10)