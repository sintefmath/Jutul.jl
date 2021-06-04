using Terv
using Test

function test_twophase(grid = "pico"; debug_level = 1, kwarg...)
    state0, model, prm, f, t = get_test_setup(grid, case_name = "two_phase_simple"; kwarg...)
    sim = Simulator(model, state0 = state0, parameters = prm)
    cfg = simulator_config(sim, debug_level = debug_level)
    simulate(sim, t, forces = f, config = cfg)
    return true
end

bctx = DefaultContext(matrix_layout = BlockMajorLayout())
ctx = DefaultContext(matrix_layout = UnitMajorLayout())

@testset "Two phase flow" begin
    @testset "Basic flow" begin
        @test test_twophase()
    end
    @testset "Block assembly" begin
        @test test_twophase(context = bctx)
    end
    # @testset "Unit major assembly" begin
    #    @test test_twophase(context = ctx)
    # end
end