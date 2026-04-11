using Jutul
using Jutul.JutulKernelAbstractions: transfer_to_device
using Test
using SparseArrays
using JLArrays
using KernelAbstractions
using GPUArrays

@testset "KernelAbstractions device simulation" begin
    backend = JLBackend()

    @testset "SimpleHeatSystem on device" begin
        sys = SimpleHeatSystem()
        g = CartesianMesh((4, 4), (1.0, 1.0))
        D = DiscretizedDomain(g)
        model = SimulationModel(D, sys, context = ParallelCSRContext(1))
        nc = number_of_cells(g)
        T0 = collect(range(1.0, 10.0, length = nc))
        state0 = setup_state(model, Dict(:T => T0))

        # Run on CPU for reference
        sim_cpu = Simulator(model, state0 = state0)
        states_cpu, = simulate!(sim_cpu, [1.0]; info_level = -1)
        T_cpu = states_cpu[1][:T]

        # Run on device
        sim2 = Simulator(model, state0 = state0)
        sim_dev = transfer_to_device(sim2, backend)
        states_dev, = simulate!(sim_dev, [1.0]; info_level = -1)

        @test length(states_dev) == 1
        T_dev = Array(states_dev[1][:T])
        @test T_dev ≈ T_cpu rtol=1e-10
    end

    @testset "VariablePoissonSystem (steady-state) on device" begin
        sys = VariablePoissonSystem()
        g = CartesianMesh((3, 1), (1.0, 1.0))
        domain = DataDomain(g, poisson_coefficient = 1.0)
        model = SimulationModel(domain, sys, context = ParallelCSRContext(1))
        state0 = setup_state(model, U = 1.0)
        param = setup_parameters(model)

        nc = number_of_cells(g)
        pos_src = PoissonSource(1, 1.0)
        neg_src = PoissonSource(nc, -1.0)
        forces = setup_forces(model, sources = [pos_src, neg_src])

        # Run on CPU for reference
        case = JutulCase(model, [1.0], forces, state0 = state0, parameters = param)
        states_cpu, = simulate(case, info_level = -1)
        U_cpu = states_cpu[1][:U]
        U_cpu_norm = U_cpu .- U_cpu[1]

        # Run on device
        sim = Simulator(model, state0 = state0, parameters = param)
        sim_dev = transfer_to_device(sim, backend)
        states_dev, = simulate!(sim_dev, [1.0]; info_level = -1, forces = forces)

        @test length(states_dev) == 1
        U_dev = Array(states_dev[1][:U])
        U_dev_norm = U_dev .- U_dev[1]
        # Check known analytical result
        @test U_dev_norm ≈ [0.0, 1/3, 2/3] atol=1e-6
        # Check matches CPU
        @test U_dev_norm ≈ U_cpu_norm rtol=1e-10
    end

    @testset "VariablePoissonSystem (time-dependent) on device" begin
        sys = VariablePoissonSystem(time_dependent = true)
        g = CartesianMesh((2, 2), (1.0, 1.0))
        discretization = (poisson = Jutul.PoissonDiscretization(g), )
        D = DiscretizedDomain(g, discretization)
        model = SimulationModel(D, sys, context = ParallelCSRContext(1))
        state0 = setup_state(model, Dict(:U => ones(4)))
        K = Jutul.compute_face_trans(g, 1.0)
        param = setup_parameters(model, K = K)

        nc = number_of_cells(g)
        pos_src = PoissonSource(1, 1.0)
        neg_src = PoissonSource(nc, -1.0)
        forces = setup_forces(model, sources = [pos_src, neg_src])

        dt = [0.1, 0.9]

        # Run on CPU for reference
        case = JutulCase(model, dt, forces, state0 = state0, parameters = param)
        states_cpu, = simulate(case, info_level = -1)

        # Run on device
        sim = Simulator(model, state0 = state0, parameters = param)
        sim_dev = transfer_to_device(sim, backend)
        states_dev, = simulate!(sim_dev, dt; info_level = -1, forces = forces)

        @test length(states_dev) == 2
        for i in 1:2
            U_dev = Array(states_dev[i][:U])
            U_cpu = states_cpu[i][:U]
            @test U_dev ≈ U_cpu rtol=1e-10
        end
    end

    @testset "Larger VariablePoissonSystem on device" begin
        sys = VariablePoissonSystem()
        g = CartesianMesh((5, 5), (1.0, 1.0))
        domain = DataDomain(g, poisson_coefficient = 1.0)
        model = SimulationModel(domain, sys, context = ParallelCSRContext(1))
        state0 = setup_state(model, U = 0.0)
        param = setup_parameters(model)

        nc = number_of_cells(g)
        pos_src = PoissonSource(1, 1.0)
        neg_src = PoissonSource(nc, -1.0)
        forces = setup_forces(model, sources = [pos_src, neg_src])

        # Run on CPU
        case = JutulCase(model, [1.0], forces, state0 = state0, parameters = param)
        states_cpu, = simulate(case, info_level = -1)

        # Run on device
        sim = Simulator(model, state0 = state0, parameters = param)
        sim_dev = transfer_to_device(sim, backend)
        states_dev, = simulate!(sim_dev, [1.0]; info_level = -1, forces = forces)

        @test length(states_dev) == 1
        U_dev = Array(states_dev[1][:U])
        U_cpu = states_cpu[1][:U]
        @test U_dev ≈ U_cpu rtol=1e-3
    end
end
