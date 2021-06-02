using Terv
using Test

function test_twophase(grid = "pico"; kwarg...)
    state0, model, prm, f, t = get_test_setup(grid, case_name = "two_phase_simple"; kwarg...)
    sim = Simulator(model, state0 = state0, parameters = prm)
    simulate(sim, t, forces = f)
    return true
end


@testset "Basic multiphase flow" begin
    @test test_twophase()
end

bctx = DefaultContext(matrix_layout = BlockMajorLayout())
@testset "Block assembly" begin
    @test test_twophase(context = bctx)
end
