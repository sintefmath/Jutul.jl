using Jutul, Test

@testset "Adjoint utilities" begin
    struct TestSystem <: Jutul.JutulSystem

    end

    struct ScalarTestVariable <: Jutul.ScalarVariable
    end

    struct FractionTestVariables <: Jutul.FractionVariables
        n::Int
    end

    struct MatrixTestVariable <: Jutul.JutulVariables
        n::Int
    end

    Jutul.values_per_entity(model, v::FractionTestVariables) = v.n
    Jutul.values_per_entity(model, v::MatrixTestVariable) = v.n


    function setup_test_model(layout)
        nx = 2
        ny = 1
        cm = CartesianMesh((nx, ny))
        sys = TestSystem()
        m = SimulationModel(cm, sys, context = DefaultContext(matrix_layout = layout))
        m.primary_variables[:Scalar] = ScalarTestVariable()
        m.primary_variables[:Fraction] = FractionTestVariables(3)
        m.primary_variables[:Matrix] = MatrixTestVariable(4)
        return m
    end

    eq_model = setup_test_model(EquationMajorLayout())
    block_model = setup_test_model(BlockMajorLayout())

    scalar   = [
        1.0, 2.0
    ]
    fraction = [
        0.1 0.2;
        0.3 0.4;
        0.6 0.2
    ]
    matrix   = [
        10 20;
        30 40;
        50 60;
        70 80
    ]
    state = setup_state(eq_model, Scalar = scalar, Fraction = fraction, Matrix = matrix)

    eq_x = vectorize_variables(eq_model, state, :primary)
    block_x = vectorize_variables(block_model, state, :primary)

    eq_x_ref = [1.0, 2.0, 0.1, 0.2, 0.3, 0.4, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]

    @test eq_x == eq_x_ref
    @test eq_x == block_x

    eq_y = Jutul.adjoint_transfer_canonical_order(eq_x, eq_model; to_canonical = false)
    @test eq_y == eq_x
    @test Jutul.adjoint_transfer_canonical_order(eq_y, eq_model; to_canonical = true) == eq_x

    block_y = Jutul.adjoint_transfer_canonical_order(block_x, block_model; to_canonical = false)
    block_y_ref = [1.0, 0.1, 0.3, 10, 30, 50, 70, 2.0, 0.2, 0.4, 20, 40, 60, 80]
    @test block_y == block_y_ref
    @test Jutul.adjoint_transfer_canonical_order(block_y, block_model; to_canonical = true) == block_x

    x = Dict(
        "a" => Dict("b" => Dict(:c => 42)),
        :d => Dict(:e => Dict(:f => 3.14))
    )

    @test Jutul.DictOptimization.convert_key("a.b.c", x) == ["a", "b", :c]
    @test Jutul.DictOptimization.convert_key(:d, x) == [:d]
    @test Jutul.DictOptimization.convert_key("d.e.f", x) == [:d, :e, :f]

    x = Dict(
        "a" => Dict("b" => Dict("cd" => 42)),
    )

    @test Jutul.DictOptimization.convert_key("a.b.cd", x) == ["a", "b", "cd"]
end
