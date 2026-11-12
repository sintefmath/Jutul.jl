using Jutul, Test

struct AdjointTestForce{T} <: Jutul.JutulForce
    value::T
end

function Jutul.apply_forces_to_equation!(d, storage, model, eq::Jutul.ScalarTestEquation, eq_s, f::AdjointTestForce, time)
    return Jutul.apply_forces_to_equation!(d, storage, model, eq, eq_s, ScalarTestForce(f.value), time)
end

function Jutul.vectorize_force!(v, model, f::AdjointTestForce, name, variant)
    v[1] = f.value
    return (names = [:value],)
end

function Jutul.devectorize_force(f::AdjointTestForce, model, x, meta, name, variant)
    return AdjointTestForce(only(x))
end

function finite_difference_force_test(objective, x; h = 1.0e-5)
    return map(eachindex(x)) do i
        xp, xm = copy(x), copy(x)
        xp[i] += h
        xm[i] -= h
        (objective(xp) - objective(xm)) / (2h)
    end
end

@testset "Force adjoint with distinct controls and ministeps" begin
    a = SimulationModel(ScalarTestDomain(use_manual = false), ScalarTestSystem())
    b = SimulationModel(ScalarTestDomain(use_manual = false), ScalarTestSystem())
    model = MultiModel((A = a, B = b))
    state0 = setup_state(model; A = setup_state(a; XVar = 0.0), B = setup_state(b; XVar = 0.0))
    dt = [1.0, 1.0]
    function make_case(x)
        forces = [
            setup_forces(model; A = setup_forces(a; sources = AdjointTestForce(qa)), B = setup_forces(b; sources = AdjointTestForce(qb)))
                for (qa, qb) in eachcol(reshape(x, 2, :))
        ]
        return JutulCase(model, dt, forces; state0)
    end
    function objective(m, state, dt, step, forces)
        return dt * (forces[:A].sources.value * only(state[:A][:XVar]) + 2 * forces[:B].sources.value * only(state[:B][:XVar]))
    end
    forward(case) = simulate(case; info_level = -1, output_substates = true, max_timestep = 0.5)
    x = [2.0, 5.0, 3.0, 7.0]
    case = make_case(x)
    result = forward(case)
    @test report_timesteps(result.reports; ministeps = true) == fill(0.5, 4)
    dforces, mapping, gradient = Jutul.solve_adjoint_forces(case, result, objective)
    numerical = finite_difference_force_test(x) do y
        perturbed = make_case(y)
        Jutul.evaluate_objective(objective, perturbed, forward(perturbed))
    end
    @test mapping == [1, 1, 2, 2]
    @test vcat(gradient...) ≈ numerical rtol = 1.0e-6
    @test [f[k].sources.value for f in dforces for k in (:A, :B)] ≈ numerical rtol = 1.0e-6
end

@testset "Generic adjoint uses the forces of the setup case" begin
    model = SimulationModel(ScalarTestDomain(use_manual = false), ScalarTestSystem())
    state0 = setup_state(model; XVar = 0.0)
    dt = [1.0, 1.0]
    force(q) = setup_forces(model; sources = ScalarTestForce(q))
    setup_case(x, step) = JutulCase(model, dt, force(only(x)); state0)
    objective(m, state, dt, step, forces) = dt * forces.sources.value * only(state[:XVar])
    x = [2.0]
    case = setup_case(x, missing)
    result = simulate(case; info_level = -1)
    # The full case returned by setup_case supplies the forces for both report steps.
    packed_forces = force.([2.0, 5.0])
    gradient = Jutul.AdjointsDI.solve_adjoint_generic(
        x, setup_case, result.states, result.reports, objective; state0, forces = packed_forces
    )
    numerical = finite_difference_force_test(x) do y
        perturbed = setup_case(y, missing)
        Jutul.evaluate_objective(objective, perturbed, simulate(perturbed; info_level = -1))
    end
    @test gradient ≈ numerical rtol = 1.0e-6
end
