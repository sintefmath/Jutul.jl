using Jutul
using Test
using LinearAlgebra
using SparseArrays

@testset "VEM Linear Elasticity" begin
    @testset "VEMElasticitySetup" begin
        @testset "3D setup" begin
            mesh = UnstructuredMesh(CartesianMesh((2, 2, 2), (1.0, 1.0, 1.0)))
            setup = VEMElasticitySetup(mesh)
            @test setup.dim == 3
            @test setup.num_nodes == length(mesh.node_points)
            @test setup.num_cells == number_of_cells(mesh)
            @test length(setup.cell_data) == number_of_cells(mesh)
            @test length(setup.boundary_nodes) > 0
            # All cell data should have positive volume
            for cd in setup.cell_data
                @test cd.volume > 0
            end
        end

        @testset "2D setup" begin
            mesh = UnstructuredMesh(CartesianMesh((3, 3), (1.0, 1.0)))
            setup = VEMElasticitySetup(mesh)
            @test setup.dim == 2
            @test setup.num_nodes == length(mesh.node_points)
            @test setup.num_cells == number_of_cells(mesh)
            for cd in setup.cell_data
                @test cd.volume > 0
            end
        end
    end

    @testset "Assembly and solve 3D" begin
        mesh = UnstructuredMesh(CartesianMesh((2, 2, 2), (1.0, 1.0, 1.0)))
        nc = number_of_cells(mesh)
        nn = length(mesh.node_points)
        E = fill(1e9, nc)
        nu = fill(0.3, nc)
        # Non-uniform pressure to get nonzero displacement
        dp = [Float64(i) * 1e5 for i in 1:nc]

        setup = VEMElasticitySetup(mesh)
        K, rhs = assemble_vem_elasticity(setup, E, nu, dp)

        @test size(K) == (3 * nn, 3 * nn)
        @test length(rhs) == 3 * nn
        @test issparse(K)

        @testset "Stiffness matrix symmetry" begin
            K_full = Matrix(K)
            @test norm(K_full - K_full') / norm(K_full) < 1e-10
        end

        @testset "Zero displacement at boundaries" begin
            result = solve_vem_elasticity(setup, E, nu, dp)
            u = result.displacement
            for node in setup.boundary_nodes
                for d in 1:3
                    dof = 3 * (node - 1) + d
                    @test abs(u[dof]) < 1e-14
                end
            end
        end

        @testset "Zero pressure gives zero displacement" begin
            result0 = solve_vem_elasticity(setup, E, nu, zeros(nc))
            @test norm(result0.displacement) < 1e-14
        end

        @testset "Uniform pressure gives zero displacement" begin
            result_unif = solve_vem_elasticity(setup, E, nu, fill(1e6, nc))
            @test norm(result_unif.displacement) < 1e-14
        end

        @testset "Linear scaling with pressure" begin
            result1 = solve_vem_elasticity(setup, E, nu, dp)
            result2 = solve_vem_elasticity(setup, E, nu, 2 .* dp)
            @test norm(result1.displacement) > 0
            @test norm(result2.displacement) / norm(result1.displacement) ≈ 2.0 atol=1e-10
        end

        @testset "Stiffer material gives smaller displacement" begin
            result_soft = solve_vem_elasticity(setup, fill(1e9, nc), nu, dp)
            result_stiff = solve_vem_elasticity(setup, fill(1e10, nc), nu, dp)
            ratio = norm(result_soft.displacement) / norm(result_stiff.displacement)
            @test ratio ≈ 10.0 atol=1e-8
        end
    end

    @testset "Assembly and solve 2D" begin
        mesh = UnstructuredMesh(CartesianMesh((3, 3), (1.0, 1.0)))
        nc = number_of_cells(mesh)
        nn = length(mesh.node_points)
        E = fill(1e9, nc)
        nu = fill(0.3, nc)
        # Non-uniform pressure
        dp = [Float64(i) * 1e5 for i in 1:nc]

        setup = VEMElasticitySetup(mesh)

        @testset "Stiffness matrix symmetry" begin
            K, rhs = assemble_vem_elasticity(setup, E, nu, dp)
            K_full = Matrix(K)
            @test norm(K_full - K_full') / norm(K_full) < 1e-10
        end

        @testset "Zero displacement at boundaries" begin
            result = solve_vem_elasticity(setup, E, nu, dp)
            for node in setup.boundary_nodes
                for d in 1:2
                    dof = 2 * (node - 1) + d
                    @test abs(result.displacement[dof]) < 1e-14
                end
            end
        end

        @testset "Zero pressure gives zero displacement" begin
            result0 = solve_vem_elasticity(setup, E, nu, zeros(nc))
            @test norm(result0.displacement) < 1e-14
        end

        @testset "Uniform pressure gives zero displacement" begin
            result_unif = solve_vem_elasticity(setup, E, nu, fill(1e6, nc))
            @test norm(result_unif.displacement) < 1e-14
        end

        @testset "Linear scaling with pressure" begin
            result1 = solve_vem_elasticity(setup, E, nu, dp)
            result2 = solve_vem_elasticity(setup, E, nu, 3 .* dp)
            @test norm(result1.displacement) > 0
            @test norm(result2.displacement) / norm(result1.displacement) ≈ 3.0 atol=1e-10
        end
    end

    @testset "Setup reuse" begin
        mesh = UnstructuredMesh(CartesianMesh((3, 3, 3), (1.0, 1.0, 1.0)))
        nc = number_of_cells(mesh)
        setup = VEMElasticitySetup(mesh)

        # Solve with different material properties using same setup
        E1 = fill(1e9, nc)
        E2 = fill(2e9, nc)
        nu = fill(0.25, nc)
        # Non-uniform pressure
        dp = [Float64(i) * 1e5 for i in 1:nc]

        result1 = solve_vem_elasticity(setup, E1, nu, dp)
        result2 = solve_vem_elasticity(setup, E2, nu, dp)

        @test norm(result1.displacement) > norm(result2.displacement)
    end

    @testset "Convenience function (mesh input)" begin
        mesh = UnstructuredMesh(CartesianMesh((2, 2, 2), (1.0, 1.0, 1.0)))
        nc = number_of_cells(mesh)
        E = fill(1e9, nc)
        nu = fill(0.3, nc)
        dp = [Float64(i) * 1e5 for i in 1:nc]

        result = solve_vem_elasticity(mesh, E, nu, dp)
        @test hasproperty(result, :displacement)
        @test hasproperty(result, :setup)
        @test hasproperty(result, :K)
        @test hasproperty(result, :rhs)
        @test length(result.displacement) == 3 * length(mesh.node_points)
    end

    @testset "Biot coefficient" begin
        mesh = UnstructuredMesh(CartesianMesh((2, 2, 2), (1.0, 1.0, 1.0)))
        nc = number_of_cells(mesh)
        setup = VEMElasticitySetup(mesh)
        E = fill(1e9, nc)
        nu = fill(0.3, nc)
        # Non-uniform pressure
        dp = [Float64(i) * 1e5 for i in 1:nc]

        result_biot1 = solve_vem_elasticity(setup, E, nu, dp, biot_coefficient = 1.0)
        result_biot05 = solve_vem_elasticity(setup, E, nu, dp, biot_coefficient = 0.5)

        @test norm(result_biot1.displacement) > 0
        @test norm(result_biot1.displacement) / norm(result_biot05.displacement) ≈ 2.0 atol=1e-10
    end

    @testset "Linear displacement patch test 3D" begin
        # Prescribe u(x) = Ax + b on boundary, verify interior follows
        mesh = UnstructuredMesh(CartesianMesh((3, 3, 3), (1.0, 1.0, 1.0)))
        nc = number_of_cells(mesh)
        nn = length(mesh.node_points)
        setup = VEMElasticitySetup(mesh)
        E = fill(1e9, nc)
        nu = fill(0.3, nc)

        # Test 1: u_x = 0.001*x, u_y = 0, u_z = 0
        bc = zeros(3 * nn)
        for (i, pt) in enumerate(mesh.node_points)
            bc[3*(i-1)+1] = 0.001 * pt[1]
        end
        result = solve_vem_elasticity(setup, E, nu, zeros(nc), boundary_displacement = bc)
        for (i, pt) in enumerate(mesh.node_points)
            @test abs(result.displacement[3*(i-1)+1] - 0.001 * pt[1]) < 1e-14
            @test abs(result.displacement[3*(i-1)+2]) < 1e-14
            @test abs(result.displacement[3*(i-1)+3]) < 1e-14
        end

        # Mixed linear displacement: u_x = 0.001*y, u_y = 0.002*z, u_z = 0.0005*x
        bc2 = zeros(3 * nn)
        for (i, pt) in enumerate(mesh.node_points)
            bc2[3*(i-1)+1] = 0.001 * pt[2]
            bc2[3*(i-1)+2] = 0.002 * pt[3]
            bc2[3*(i-1)+3] = 0.0005 * pt[1]
        end
        result2 = solve_vem_elasticity(setup, E, nu, zeros(nc), boundary_displacement = bc2)
        for (i, pt) in enumerate(mesh.node_points)
            @test abs(result2.displacement[3*(i-1)+1] - 0.001 * pt[2]) < 1e-13
            @test abs(result2.displacement[3*(i-1)+2] - 0.002 * pt[3]) < 1e-13
            @test abs(result2.displacement[3*(i-1)+3] - 0.0005 * pt[1]) < 1e-13
        end
    end

    @testset "Linear displacement patch test 2D" begin
        mesh = UnstructuredMesh(CartesianMesh((4, 4), (1.0, 1.0)))
        nc = number_of_cells(mesh)
        nn = length(mesh.node_points)
        setup = VEMElasticitySetup(mesh)
        E = fill(1e9, nc)
        nu = fill(0.3, nc)

        # u_x = 0.001*x, u_y = 0.0005*y
        bc = zeros(2 * nn)
        for (i, pt) in enumerate(mesh.node_points)
            bc[2*(i-1)+1] = 0.001 * pt[1]
            bc[2*(i-1)+2] = 0.0005 * pt[2]
        end
        result = solve_vem_elasticity(setup, E, nu, zeros(nc), boundary_displacement = bc)
        for (i, pt) in enumerate(mesh.node_points)
            @test abs(result.displacement[2*(i-1)+1] - 0.001 * pt[1]) < 1e-13
            @test abs(result.displacement[2*(i-1)+2] - 0.0005 * pt[2]) < 1e-13
        end
    end
end
