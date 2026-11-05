using Jutul, Test

# Behaviour that applications (BattMo, JutulDarcy, Mocca, Fimbul) rely on when they
# extend the solver loop: hook multiplicity and order, report structure, convergence
# criteria and tolerance lookup. A small model with two generic equations of different
# types, a parameter with a before-step hook, a force and a skew-symmetric cross term
# records the current behaviour, so that changes to how the solver loop traverses
# equations, variables and cross terms are deliberate.

struct InterfaceTestSystem <: JutulSystem end
struct InterfaceVariableA <: ScalarVariable end
struct InterfaceVariableB <: ScalarVariable end
struct InterfaceParameter <: ScalarVariable end
struct InterfaceEquationA <: JutulEquation end
struct InterfaceEquationB <: JutulEquation end
struct InterfaceForce end
struct InterfaceCrossTerm <: AdditiveCrossTerm end

# Entity loops may run on several threads, so the hooks count with atomics
const INTERFACE_CALLS = (
    parameter_hook = Threads.Atomic{Int}(0),
    prepare = Threads.Atomic{Int}(0),
    update_equation_b = Threads.Atomic{Int}(0),
    linearize_b = Threads.Atomic{Int}(0),
    force = Threads.Atomic{Int}(0),
    cross_term = Threads.Atomic{Int}(0),
    prepare_cross_term = Threads.Atomic{Int}(0),
)
const INTERFACE_EQUATION_ORDER = Symbol[]
const INTERFACE_UPDATE_REPORTS = []

count_interface_call!(name) = Threads.atomic_add!(INTERFACE_CALLS[name], 1)

function reset_interface_log!()
    foreach(counter -> counter[] = 0, INTERFACE_CALLS)
    empty!(INTERFACE_EQUATION_ORDER)
    empty!(INTERFACE_UPDATE_REPORTS)
    return nothing
end

function Jutul.select_primary_variables!(S, ::InterfaceTestSystem, model::SimulationModel)
    S[:A] = InterfaceVariableA()
    S[:B] = InterfaceVariableB()
    return S
end

function Jutul.select_parameters!(S, ::InterfaceTestSystem, model::SimulationModel)
    S[:P] = InterfaceParameter()
    return S
end

function Jutul.select_equations!(eqs, ::InterfaceTestSystem, model::SimulationModel)
    eqs[:eq_a] = InterfaceEquationA()
    eqs[:eq_b] = InterfaceEquationB()
    return eqs
end

Jutul.default_value(model, ::InterfaceParameter) = 2.0
Jutul.local_discretization(::Union{InterfaceEquationA, InterfaceEquationB}, i) = nothing

function Jutul.setup_forces(model::SimulationModel{<:Any, InterfaceTestSystem}; force = nothing)
    return (force = force,)
end

# dA/dt = B - A and B = P, so that A relaxes towards the parameter value
function Jutul.update_equation_in_entity!(v, i, state, state0, eq::InterfaceEquationA, model, dt, ldisc = nothing)
    A = state.A[i]
    return v[] = (A - state0.A[i]) / dt + A - state.B[i]
end

function Jutul.update_equation_in_entity!(v, i, state, state0, eq::InterfaceEquationB, model, dt, ldisc = nothing)
    return v[] = state.B[i] - state.P[i]
end

function Jutul.update_parameter_before_step!(value, ::InterfaceParameter, storage, model, dt, forces)
    count_interface_call!(:parameter_hook)
    return value
end

function Jutul.prepare_equation_in_entity!(i, eq::Union{InterfaceEquationA, InterfaceEquationB}, eq_s, state, state0, model, dt)
    count_interface_call!(:prepare)
    if i == 1
        push!(INTERFACE_EQUATION_ORDER, eq isa InterfaceEquationA ? :eq_a : :eq_b)
    end
    return nothing
end

# Overrides in the style of JutulDarcy that wrap the generic implementation
function Jutul.update_equation!(eq_s, eq::InterfaceEquationB, storage, model, dt)
    count_interface_call!(:update_equation_b)
    return invoke(Jutul.update_equation!, Tuple{Any, JutulEquation, Any, Any, Any}, eq_s, eq, storage, model, dt)
end

function Jutul.update_linearized_system_equation!(nz::AbstractArray, r, model, eq::InterfaceEquationB, caches::NamedTuple)
    count_interface_call!(:linearize_b)
    return invoke(Jutul.update_linearized_system_equation!, Tuple{Any, Any, Any, JutulEquation, Any}, nz, r, model, eq, caches)
end

function Jutul.apply_forces_to_equation!(diag_part, storage, model, eq::InterfaceEquationA, eq_s, force::InterfaceForce, time)
    count_interface_call!(:force)
    return diag_part
end

# Two named criteria, and a record of the increment report passed in by the solver
function Jutul.convergence_criterion(model, storage, eq::InterfaceEquationB, eq_s, r; dt = 1.0, update_report = missing)
    ismissing(update_report) || push!(INTERFACE_UPDATE_REPORTS, update_report)
    e = maximum(abs, r)
    return (AbsMax = (errors = [e], names = "R"), Scaled = (errors = [e / 2], names = "R"))
end

Jutul.symmetry(::InterfaceCrossTerm) = Jutul.CTSkewSymmetry()

function Jutul.prepare_cross_term_in_entity!(i, state_t, state0_t, state_s, state0_s, model_t, model_s, ct::InterfaceCrossTerm, eq, dt)
    count_interface_call!(:prepare_cross_term)
    return nothing
end

function Jutul.update_cross_term_in_entity!(out, i, state_t, state0_t, state_s, state0_s, model_t, model_s, ct::InterfaceCrossTerm, eq, dt, ldisc = nothing)
    count_interface_call!(:cross_term)
    return out[] = state_t.A[i] - state_s.A[i]
end

interface_model(n) = SimulationModel(DiscretizedDomain(CartesianMesh((n,))), InterfaceTestSystem())

function run_interface_simulation(model, state0, forces; specialize = false, kwarg...)
    sim = Simulator(model, state0 = state0, specialize = specialize)
    # Setup evaluates equations and cross terms for sparsity detection, count the solve only
    reset_interface_log!()
    states, reports = simulate(sim, [0.5, 0.5]; forces = forces, info_level = -1, kwarg...)
    ministeps = [ministep for report in reports for ministep in report[:ministeps]]
    return (
        sim = sim,
        states = states,
        reports = reports,
        ministeps = length(ministeps),
        linearizations = sum(ministep -> length(ministep[:steps]), ministeps),
        calls = map(counter -> counter[], INTERFACE_CALLS),
        equation_order = copy(INTERFACE_EQUATION_ORDER),
        update_reports = copy(INTERFACE_UPDATE_REPORTS),
    )
end

function simulate_interface_model(; n = 3, force = InterfaceForce(), kwarg...)
    model = interface_model(n)
    state0 = setup_state(model, Dict(:A => 0.0, :B => 0.0))
    return run_interface_simulation(model, state0, setup_forces(model, force = force); kwarg...)
end

function simulate_interface_multimodel(; n = 3, groups = nothing, kwarg...)
    models = (M1 = interface_model(n), M2 = interface_model(n))
    model = MultiModel(models, groups = groups)
    add_cross_term!(model, InterfaceCrossTerm(), target = :M1, source = :M2, equation = :eq_a)
    state0 = setup_state(
        model,
        M1 = setup_state(models.M1, Dict(:A => 0.0, :B => 0.0)),
        M2 = setup_state(models.M2, Dict(:A => 1.0, :B => 0.0)),
    )
    forces = setup_forces(model, M1 = setup_forces(models.M1, force = InterfaceForce()))
    return run_interface_simulation(model, state0, forces; kwarg...)
end

@testset "Application API: single model" begin
    n = 3
    results = Dict(specialize => simulate_interface_model(; n, specialize) for specialize in (false, true))
    for (specialize, result) in results
        @testset "specialize = $specialize" begin
            (; calls, linearizations, ministeps) = result
            # Implicit Euler for dA/dt = B - A with B = P = 2 and two steps of 0.5
            @test isapprox(result.states[end][:A], fill(10 / 9, n), rtol = 1.0e-10)
            @test isapprox(result.states[end][:B], fill(2.0, n), rtol = 1.0e-10)
            # update_before_step! calls the parameter hook once per ministep
            @test calls.parameter_hook == ministeps
            # Equations are updated in model order and prepared once per entity
            @test result.equation_order == repeat([:eq_a, :eq_b], linearizations)
            @test calls.prepare == 2 * n * linearizations
            # Overrides of update_equation!, update_linearized_system_equation! and
            # apply_forces_to_equation! are called once per linearization
            @test calls.update_equation_b == linearizations
            @test calls.linearize_b == linearizations
            @test calls.force == linearizations
        end
    end
    @test results[false].sim.storage isa Jutul.JutulStorage
    @test results[true].sim.storage isa Jutul.ImmutableJutulStorage
    @test results[true].calls == results[false].calls
    # A force entry with value nothing does not reach the equation hook
    @test simulate_interface_model(force = nothing).calls.force == 0
end

@testset "Application API: reports" begin
    iteration_hook = (converged, report, storage, model, dt, forces, config, iteration) -> begin
        report[:user_iteration] = iteration
        return converged
    end
    ministep_hook = (done, report, sim, dt, forces, max_iter, config) -> begin
        report[:user_dt] = dt
        return (done, report)
    end
    result = simulate_interface_model(post_iteration_hook = iteration_hook, post_ministep_hook = ministep_hook)
    step_keys = (:secondary_time, :equations_time, :linear_system_time, :convergence_time, :solved, :converged, :errors)
    solve_keys = (:update, :linear_solver, :linear_iterations, :linear_solve_time, :update_time)
    for report in result.reports
        @test haskey(report, :total_time)
        for ministep in report[:ministeps]
            @test all(k -> haskey(ministep, k), (:dt, :steps, :success, :prepare_time, :finalize_time, :post_update))
            steps = ministep[:steps]
            # Keys added by hooks are kept and every iteration owns its report
            @test ministep[:user_dt] == ministep[:dt]
            @test [step[:user_iteration] for step in steps] == collect(1:length(steps))
            @test all(step -> all(k -> haskey(step, k), step_keys), steps)
            @test all(step -> all(k -> haskey(step, k), solve_keys), filter(step -> step[:solved], steps))
            stats = Jutul.stats_ministep(steps)
            @test stats.linearizations == length(steps)
            @test stats.newtons == count(step -> step[:solved], steps)
            # Variable change report read by VariableChangeTimestepSelector
            change = ministep[:post_update][:A]
            @test change.n == 3
            @test change.x.max >= change.dx.max > 0
            @test change.dx.sum >= change.dx.max
        end
    end
    @test Jutul.report_stats(result.reports).newtons == sum(step -> step[:solved], (step for report in result.reports for ministep in report[:ministeps] for step in ministep[:steps]))
end

@testset "Application API: convergence criteria" begin
    tolerances = Dict(:default => 1.0e-3, :eq_a => 1.0e-8, :eq_b => Dict(:AbsMax => 1.0e-9))
    result = simulate_interface_model(tolerances = tolerances)
    errors = result.reports[1][:ministeps][1][:steps][1][:errors]
    @test [entry.name for entry in errors] == [:eq_a, :eq_b]
    @test keys(errors[1].criterions) == (:AbsMax,)
    @test keys(errors[2].criterions) == (:AbsMax, :Scaled)
    # Tolerance lookup: a number per equation, then per criterion name, then the default
    @test isapprox(errors[1].tolerances[:AbsMax], 1.0e-8, rtol = 1.0e-12)
    @test isapprox(errors[2].tolerances[:AbsMax], 1.0e-9, rtol = 1.0e-12)
    @test isapprox(errors[2].tolerances[:Scaled], 1.0e-3, rtol = 1.0e-12)
    # From the second iteration on, criteria receive the increment report of the last solve
    @test length(result.update_reports) == result.linearizations - result.ministeps
    @test all(update -> update[:A].sum >= update[:A].max >= 0, result.update_reports)
end

@testset "Application API: multimodel with cross term" begin
    n = 3
    variants = [
        (groups = nothing, specialize = false, linear_solver = nothing),
        (groups = nothing, specialize = true, linear_solver = nothing),
        (groups = [1, 2], specialize = false, linear_solver = GenericKrylov(relative_tolerance = 1.0e-12)),
    ]
    for variant in variants
        @testset "groups = $(variant.groups), specialize = $(variant.specialize)" begin
            result = simulate_interface_multimodel(; n, variant...)
            (; calls, linearizations, ministeps) = result
            # The skew-symmetric cross term cancels in the sum of the two A equations
            A_sum = result.states[end][:M1][:A] + result.states[end][:M2][:A]
            @test isapprox(A_sum, fill(8 / 3, n), rtol = 1.0e-8)
            @test calls.parameter_hook == 2 * ministeps
            @test result.equation_order == repeat([:eq_a, :eq_b, :eq_a, :eq_b], linearizations)
            @test calls.update_equation_b == 2 * linearizations
            @test calls.linearize_b == 2 * linearizations
            @test calls.force == linearizations
            # One preparation per target entity, one evaluation each with target and
            # source derivatives
            @test calls.prepare_cross_term == n * linearizations
            @test calls.cross_term == 2 * n * linearizations
            step = result.reports[1][:ministeps][1][:steps][1]
            @test collect(keys(step[:errors])) == [:M1, :M2]
            @test all(entries -> [entry.name for entry in entries] == [:eq_a, :eq_b], values(step[:errors]))
            @test haskey(result.reports[1][:ministeps][1][:post_update][:M1], :A)
            @test all(update -> haskey(update, :A), result.update_reports)
        end
    end
end
