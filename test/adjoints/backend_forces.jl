using Jutul, Test

const AdjointForcesAdapt = Jutul.KernelExecution.Adapt
const AdjointForcesKA = Jutul.KernelExecution.KernelAbstractions

struct AdjointBackendTestForce <: Jutul.JutulForce
    value::Float64
    prepared::Bool
end

# Mark transfers explicitly so a CPU backend can detect a missing device transfer.
function AdjointForcesAdapt.adapt_storage(::KernelAbstractionsContext, forces::Vector{AdjointBackendTestForce})
    return [AdjointBackendTestForce(force.value, true) for force in forces]
end

function Jutul.apply_forces_to_equation!(d, storage, model, eq::Jutul.ScalarTestEquation, eq_s,
        forces::AbstractVector{AdjointBackendTestForce}, time)
    backend = model.context isa KernelAbstractionsContext
    @assert all(force -> force.prepared == backend, forces) "Forces do not match the assembly backend"
    source = ScalarTestForce(sum(force.value for force in forces))
    return Jutul.apply_forces_to_equation!(d, storage, model, eq, eq_s, source, time)
end

@testset "Adjoint forces follow each simulator's backend" begin
    model = SimulationModel(ScalarTestDomain(use_manual = false), ScalarTestSystem();
        context = ParallelCSRContext(1))
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
