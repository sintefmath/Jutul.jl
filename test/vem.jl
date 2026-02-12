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

    @testset "Displacement output shape" begin
        @testset "3D output" begin
            mesh = UnstructuredMesh(CartesianMesh((2, 2, 2), (1.0, 1.0, 1.0)))
            nc = number_of_cells(mesh)
            nn = length(mesh.node_points)
            setup = VEMElasticitySetup(mesh)
            dp = [Float64(i) * 1e5 for i in 1:nc]
            result = solve_vem_elasticity(setup, fill(1e9, nc), fill(0.3, nc), dp)
            @test size(result.displacement) == (3, nn)
        end

        @testset "2D output" begin
            mesh = UnstructuredMesh(CartesianMesh((3, 3), (1.0, 1.0)))
            nc = number_of_cells(mesh)
            nn = length(mesh.node_points)
            setup = VEMElasticitySetup(mesh)
            dp = [Float64(i) * 1e5 for i in 1:nc]
            result = solve_vem_elasticity(setup, fill(1e9, nc), fill(0.3, nc), dp)
            @test size(result.displacement) == (2, nn)
        end
    end

    @testset "Boundary node helpers" begin
        @testset "3D boundary helpers" begin
            mesh = UnstructuredMesh(CartesianMesh((3, 3, 3), (1.0, 1.0, 1.0)))
            setup = VEMElasticitySetup(mesh)
            pts = mesh.node_points

            bnodes = boundary_nodes(setup)
            @test length(bnodes) == length(setup.boundary_nodes)
            @test bnodes == setup.boundary_nodes

            # Test each side
            for (dir, dim_idx, cmp) in [
                (:xmin, 1, 0.0), (:xmax, 1, 1.0),
                (:ymin, 2, 0.0), (:ymax, 2, 1.0),
                (:zmin, 3, 0.0), (:zmax, 3, 1.0)
            ]
                side = boundary_nodes_on_side(setup, dir)
                @test length(side) > 0
                for n in side
                    @test n in bnodes
                    @test abs(pts[n][dim_idx] - cmp) < 1e-10
                end
            end

            # xmin and xmax should not overlap
            xmin_nodes = boundary_nodes_on_side(setup, :xmin)
            xmax_nodes = boundary_nodes_on_side(setup, :xmax)
            @test isempty(intersect(xmin_nodes, xmax_nodes))
        end

        @testset "2D boundary helpers" begin
            mesh = UnstructuredMesh(CartesianMesh((3, 3), (1.0, 1.0)))
            setup = VEMElasticitySetup(mesh)
            pts = mesh.node_points

            for (dir, dim_idx, cmp) in [
                (:xmin, 1, 0.0), (:xmax, 1, 1.0),
                (:ymin, 2, 0.0), (:ymax, 2, 1.0)
            ]
                side = boundary_nodes_on_side(setup, dir)
                @test length(side) > 0
                for n in side
                    @test abs(pts[n][dim_idx] - cmp) < 1e-10
                end
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
            for node in boundary_nodes(setup)
                for d in 1:3
                    @test abs(u[d, node]) < 1e-14
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
            for node in boundary_nodes(setup)
                for d in 1:2
                    @test abs(result.displacement[d, node]) < 1e-14
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
        nn = length(mesh.node_points)
        E = fill(1e9, nc)
        nu = fill(0.3, nc)
        dp = [Float64(i) * 1e5 for i in 1:nc]

        result = solve_vem_elasticity(mesh, E, nu, dp)
        @test hasproperty(result, :displacement)
        @test hasproperty(result, :setup)
        @test hasproperty(result, :K)
        @test hasproperty(result, :rhs)
        @test size(result.displacement) == (3, nn)
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
        # Prescribe u(x) = Ax + b on boundary nodes only, verify interior follows
        mesh = UnstructuredMesh(CartesianMesh((3, 3, 3), (1.0, 1.0, 1.0)))
        nc = number_of_cells(mesh)
        nn = length(mesh.node_points)
        setup = VEMElasticitySetup(mesh)
        E = fill(1e9, nc)
        nu = fill(0.3, nc)
        bnodes = boundary_nodes(setup)

        # u_x = 0.001*x, u_y = 0, u_z = 0 — set only on boundary nodes
        bc = zeros(3 * nn)
        for n in bnodes
            pt = mesh.node_points[n]
            bc[3*(n-1)+1] = 0.001 * pt[1]
        end
        result = solve_vem_elasticity(setup, E, nu, zeros(nc), boundary_displacement = bc)
        for (i, pt) in enumerate(mesh.node_points)
            @test abs(result.displacement[1, i] - 0.001 * pt[1]) < 1e-14
            @test abs(result.displacement[2, i]) < 1e-14
            @test abs(result.displacement[3, i]) < 1e-14
        end

        # Mixed linear displacement — set only on boundary nodes
        bc2 = zeros(3 * nn)
        for n in bnodes
            pt = mesh.node_points[n]
            bc2[3*(n-1)+1] = 0.001 * pt[2]
            bc2[3*(n-1)+2] = 0.002 * pt[3]
            bc2[3*(n-1)+3] = 0.0005 * pt[1]
        end
        result2 = solve_vem_elasticity(setup, E, nu, zeros(nc), boundary_displacement = bc2)
        for (i, pt) in enumerate(mesh.node_points)
            @test abs(result2.displacement[1, i] - 0.001 * pt[2]) < 1e-13
            @test abs(result2.displacement[2, i] - 0.002 * pt[3]) < 1e-13
            @test abs(result2.displacement[3, i] - 0.0005 * pt[1]) < 1e-13
        end
    end

    @testset "Linear displacement patch test 2D" begin
        mesh = UnstructuredMesh(CartesianMesh((4, 4), (1.0, 1.0)))
        nc = number_of_cells(mesh)
        nn = length(mesh.node_points)
        setup = VEMElasticitySetup(mesh)
        E = fill(1e9, nc)
        nu = fill(0.3, nc)
        bnodes = boundary_nodes(setup)

        # u_x = 0.001*x, u_y = 0.0005*y — set only on boundary nodes
        bc = zeros(2 * nn)
        for n in bnodes
            pt = mesh.node_points[n]
            bc[2*(n-1)+1] = 0.001 * pt[1]
            bc[2*(n-1)+2] = 0.0005 * pt[2]
        end
        result = solve_vem_elasticity(setup, E, nu, zeros(nc), boundary_displacement = bc)
        for (i, pt) in enumerate(mesh.node_points)
            @test abs(result.displacement[1, i] - 0.001 * pt[1]) < 1e-13
            @test abs(result.displacement[2, i] - 0.0005 * pt[2]) < 1e-13
        end
    end

    @testset "Boundary side helpers with solve" begin
        mesh = UnstructuredMesh(CartesianMesh((3, 3, 3), (1.0, 1.0, 1.0)))
        nc = number_of_cells(mesh)
        nn = length(mesh.node_points)
        setup = VEMElasticitySetup(mesh)
        pts = mesh.node_points

        xmin_nodes = boundary_nodes_on_side(setup, :xmin)
        @test length(xmin_nodes) > 0
        for n in xmin_nodes
            @test abs(pts[n][1]) < 1e-10
        end

        xmax_nodes = boundary_nodes_on_side(setup, :xmax)
        @test length(xmax_nodes) > 0
        for n in xmax_nodes
            @test abs(pts[n][1] - 1.0) < 1e-10
        end
    end
end
