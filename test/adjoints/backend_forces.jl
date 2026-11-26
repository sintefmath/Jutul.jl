using Jutul, Test

const AdjointForcesAdapt = Jutul.KernelExecution.Adapt
const AdjointForcesKA = Jutul.KernelExecution.KernelAbstractions

struct AdjointBackendTestForce <: Jutul.JutulForce
    value::Float64
    prepared::Bool
end

struct AdjointTestCoefficient <: ScalarVariable end
struct AdjointTestProperty <: ScalarVariable end

@jutul_secondary function adjoint_test_property!(
        out, property::AdjointTestProperty, model, XVar, Coefficient, indices
    )
    for i in indices
        out[i] = Coefficient[i] * XVar[i]^2
    end
end

@testset "Adjoint secondary properties on the selected backend" begin
    model = SimulationModel(
        ScalarTestDomain(use_manual = false), ScalarTestSystem();
        context = ParallelCSRContext(1)
    )
    set_parameters!(model; Coefficient = AdjointTestCoefficient())
    set_secondary_variables!(model; Property = AdjointTestProperty())
    state0 = setup_state(model; XVar = 2.0)
    parameters = setup_parameters(model; Coefficient = 3.0)
    forces = setup_forces(model)
    dt = [1.0, 1.0]
    states = [Jutul.JutulStorage(deepcopy(state0)) for _ in dt]
    G(model, state, dt, step_info, forces) = only(state.Property)

    sim = Simulator(model; state0, parameters)
    backend_sim = transfer_to_backend(sim, AdjointForcesKA.CPU())
    for execution_model in (model, backend_sim.model)
        storage = setup_adjoint_storage(
            model;
            state0,
            parameters,
            execution_model,
            include_state0 = true,
            linear_solver = GenericKrylov(:gmres, rtol = 1.0e-12)
        )
        if execution_model === backend_sim.model
            @test storage.parameter.model.context isa KernelAbstractionsContext
        end
        gradient = zeros(1)
        for _ in 1:2
            solve_adjoint_sensitivities!(
                gradient, storage, states, state0, dt, G; forces, info_level = -1
            )
            @test only(gradient) ≈ 8.0
            @test only(storage.dstate0) ≈ 24.0
        end
    end
end

# Mark transfers explicitly so a CPU backend can detect a missing device transfer.
function AdjointForcesAdapt.adapt_storage(::KernelAbstractionsContext, forces::Vector{AdjointBackendTestForce})
    return [AdjointBackendTestForce(force.value, true) for force in forces]
end

function Jutul.apply_forces_to_equation!(
        d, storage, model, eq::Jutul.ScalarTestEquation, eq_s,
        forces::AbstractVector{AdjointBackendTestForce}, time
    )
    backend = model.context isa KernelAbstractionsContext
    @assert all(force -> force.prepared == backend, forces) "Forces do not match the assembly backend"
    source = ScalarTestForce(sum(force.value for force in forces))
    return Jutul.apply_forces_to_equation!(d, storage, model, eq, eq_s, source, time)
end

@testset "Adjoint forces follow each simulator's backend" begin
    model = SimulationModel(
        ScalarTestDomain(use_manual = false), ScalarTestSystem();
        context = ParallelCSRContext(1)
    )
    state = setup_state(model; XVar = 0.0)
    forces = setup_forces(model; sources = [AdjointBackendTestForce(2.0, false)])
    backend_sim = transfer_to_backend(Simulator(model; state0 = state), AdjointForcesKA.CPU())
    backend_forces = Jutul.preprocess_forces(backend_sim, forces).forces
    wrapped = Jutul.KernelExecution.BackendForces(forces, backend_forces)
    @test !only(forces.sources).prepared
    @test only(backend_forces.sources).prepared

    for mode in (:forward, :reverse), backend in (true, false), input in (forces, wrapped)
        sim = Simulator(model; state0 = state, mode)
        if backend
            sim = transfer_to_backend(sim, AdjointForcesKA.CPU())
        end
        Jutul.adjoint_reassemble!(sim, state, state, 1.0, input, 0.0)
        @test only(Jutul.vector_residual(sim.storage.LinearizedSystem)) ≈ -2.0
        @test !only(forces.sources).prepared
    end
end
