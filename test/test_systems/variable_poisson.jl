using Jutul
using Test

# function test_poisson(nx = 3, ny = nx)
nx = 3
ny = 1
sys = VariablePoissonSystem()
# Unit square
g = CartesianMesh((nx, ny), (1.0, 1.0))
# Set up a model with the grid and system
discretization = (poisson = Jutul.PoissonDiscretization(g), )
D = DiscretizedDomain(g, discretization)
model = SimulationModel(D, sys)
# Initial condition is random values
nc = number_of_cells(g)
U0 = ones(nc)
state0 = setup_state(model, Dict(:U=>U0))
param = setup_parameters(model)

nc = number_of_cells(g)
pos_src = PoissonSource(1, 1.0)
neg_src = PoissonSource(nc, -1.0)
forces = setup_forces(model, sources = [pos_src, neg_src])

sim = Simulator(model, state0 = state0, parameters = param)
states, = simulate(sim, [1.0], info_level = 1, forces = forces)
#     return states
# end

# @testset "Poisson 2D" begin
#     @test begin
#         states = test_poisson(4, 4)
#         true
#     end
# end
