using Jutul
using Test

# function test_poisson(nx = 3, ny = nx)
nx = ny = 3
sys = VariablePoissonSystem()
# Unit square
g = CartesianMesh((nx, ny), (1.0, 1.0))
# Set up a model with the grid and system
discretization = (poisson = Jutul.PoissonDiscretization(g), )
D = DiscretizedDomain(g, discretization)
model = SimulationModel(D, sys)
# Initial condition is random values
nc = number_of_cells(g)
U0 = zeros(nc)
state0 = setup_state(model, Dict(:U=>U0))
sim = Simulator(model, state0 = state0)
states, = simulate(sim, [1.0], info_level = 1)
#     return states
# end

# @testset "Poisson 2D" begin
#     @test begin
#         states = test_poisson(4, 4)
#         true
#     end
# end
