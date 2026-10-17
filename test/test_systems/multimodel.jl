using Jutul
using Test

function test_multi(; use_groups = false, specialize_model = false, specialize_storage = false, kwarg...)
    sys = ScalarTestSystem()
    # Model A
    A = ScalarTestDomain()
    modelA = SimulationModel(A, sys)
    sourceA = ScalarTestForce(1.0)
    forcesA = setup_forces(modelA, sources = sourceA)
    state0A = setup_state(modelA, Dict(:XVar => 0.0))
    # Model B
    B = ScalarTestDomain()
    modelB = SimulationModel(B, sys)
    sourceB = ScalarTestForce(-1.0)
    forcesB = setup_forces(modelB, sources = sourceB)
    state0B = setup_state(modelB, Dict(:XVar => 0.0))

    # Make a multimodel
    if use_groups
        groups = [1, 2]
    else
        groups = nothing
    end
    model = MultiModel((A = modelA, B = modelB), groups = groups, specialize = specialize_model)
    add_cross_term!(model, ScalarTestCrossTerm(), target = :A, source = :B, equation = :test_equation)
    # Set up joint state and forces
    state0 = setup_state(model, A = state0A, B = state0B)
    forces = setup_forces(model, A = forcesA, B = forcesB)
    # Set up simulator, and run simulation
    sim = Simulator(model, state0 = state0, specialize = specialize_storage)
    states, = simulate(sim, [1.0], forces = forces, info_level = -1; kwarg...)

    XA = states[end][:A][:XVar]
    XB = states[end][:B][:XVar]

    return XA[] ≈ 1 / 3 && XB[] ≈ -1 / 3
end

group_precond = GroupWisePreconditioner([TrivialPreconditioner(), TrivialPreconditioner()])

@testset "Multi-model: Scalar test system" begin
    @testset "Single sparse matrix" begin
        @test test_multi(use_groups = false)
        @test test_multi(use_groups = false, specialize_storage = true, specialize_model = false)
        @test test_multi(use_groups = false, specialize_storage = true, specialize_model = true)
        @test test_multi(use_groups = false, specialize_storage = false, specialize_model = false)
    end
    @testset "Multiple sparse matrices" begin
        @test test_multi(use_groups = true, linear_solver = GenericKrylov())
        # @test test_multi(use_groups = true, linear_solver = GenericKrylov(preconditioner = group_precond))
    end
end

struct PreparationTestVariable <: Jutul.ScalarVariable
    coefficient::Float64
end

struct PreparationTestCrossTerm{T} <: Jutul.AdditiveCrossTerm
    coefficient::T
end

Jutul.symmetry(::PreparationTestCrossTerm) = Jutul.CTSkewSymmetry()
Jutul.cross_term_entities(::PreparationTestCrossTerm, eq::Jutul.JutulEquation, model) = [1]

function Jutul.prepare_cross_term(ct::PreparationTestCrossTerm{Nothing}, target, source)
    return PreparationTestCrossTerm(target[:XVar].coefficient)
end

function Jutul.update_cross_term_in_entity!(out, i, state_t, state0_t,
        state_s, state0_s, model_t, model_s, ct::PreparationTestCrossTerm,
        eq::Jutul.JutulEquation, dt, ldisc = Jutul.local_discretization(ct, i))
    out[1] = ct.coefficient*(state_t.XVar[1] - state_s.XVar[1])
    return out
end

@testset "Cross terms capture finalized model definitions" begin
    for immutable in (false, true)
        left = SimulationModel(ScalarTestDomain(), ScalarTestSystem())
        right = SimulationModel(ScalarTestDomain(), ScalarTestSystem())
        set_primary_variables!(left; XVar = PreparationTestVariable(2.0))
        model = MultiModel((Left = left, Right = right))
        add_cross_term!(model, PreparationTestCrossTerm(nothing);
            target = :Left, source = :Right, equation = :test_equation)
        # Configure the variable after assembling the MultiModel. Preparation
        # must use the final definition and leave the input cross term alone.
        set_primary_variables!(left; XVar = PreparationTestVariable(3.0))
        if immutable
            model = convert_to_immutable_storage(model)
        end
        state0 = setup_state(model; Left = Dict(:XVar => 0.0), Right = Dict(:XVar => 0.0))
        forces = setup_forces(model;
            Left = setup_forces(left; sources = ScalarTestForce(1.0)),
            Right = setup_forces(right; sources = ScalarTestForce(-1.0)))
        simulator = Simulator(model; state0)
        @test only(model.cross_terms).cross_term.coefficient === nothing
        @test only(simulator.model.cross_terms).cross_term.coefficient == 3.0
        @test Jutul.prepare_model_for_simulation(simulator.model) === simulator.model
        states, = simulate!(simulator, [1.0]; forces, info_level = -1)
        @test only(states[end][:Left][:XVar]) ≈ 1/7
        @test only(states[end][:Right][:XVar]) ≈ -1/7
    end
end
