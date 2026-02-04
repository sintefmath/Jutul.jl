using Jutul, Test, LinearAlgebra
using Jutul.EmbeddedMeshes

@testset "EmbeddedMeshes" begin
    @testset "Simple 2D Fracture Network" begin
        # Create a 3x3x3 cartesian mesh to extract fractures from
        parent_dims = (3, 3, 3)
        parent_mesh = UnstructuredMesh(CartesianMesh(parent_dims, (4.0, 4.0, 4.0)))
        
        # Get face connectivity to select fractures
        neighbors = get_neighborship(parent_mesh)
        ijk = reinterpret(reshape, Int, map(c -> cell_ijk(parent_mesh, c), 1:number_of_cells(parent_mesh)))
        
        # Select faces representing a cross-shaped fracture pattern
        # Vertical fracture: separating cells with i=1 from i=2 (x-normal faces)
        # Horizontal fracture: separating cells with j=1 from j=2 (y-normal faces)
        face_mask = falses(size(neighbors, 2))
        
        # X-normal fracture at i=1.5 (between i=1 and i=2)
        for k = 1:parent_dims[3], j = 1:parent_dims[2]
            cell_mask_l = (ijk[1,:] .== 1) .& (ijk[2,:] .== j) .& (ijk[3,:] .== k)
            cell_mask_r = (ijk[1,:] .== 2) .& (ijk[2,:] .== j) .& (ijk[3,:] .== k)
            face_mask = face_mask .| vec(any(cell_mask_l[neighbors], dims = 1) .& any(cell_mask_r[neighbors], dims = 1))
        end
        
        # Y-normal fracture at j=1.5 (between j=1 and j=2)  
        for k = 1:parent_dims[3], i = 1:parent_dims[1]
            cell_mask_l = (ijk[1,:] .== i) .& (ijk[2,:] .== 1) .& (ijk[3,:] .== k)
            cell_mask_r = (ijk[1,:] .== i) .& (ijk[2,:] .== 2) .& (ijk[3,:] .== k)
            face_mask = face_mask .| vec(any(cell_mask_l[neighbors], dims = 1) .& any(cell_mask_r[neighbors], dims = 1))
        end
        
        fracture_faces = findall(face_mask)
        embedded_mesh = Jutul.EmbeddedMeshes.EmbeddedMesh(parent_mesh, fracture_faces)
        
        @testset "EmbeddedMesh Construction" begin
            # Create embedded mesh
            
            @test embedded_mesh isa Jutul.EmbeddedMeshes.EmbeddedMesh
            @test embedded_mesh.unstructured_mesh isa Jutul.UnstructuredMesh
            @test embedded_mesh.intersections isa Vector{Vector{Int}}
            
            # Test basic interface functions
            @test dim(embedded_mesh) == 3  # Embedded features are 2D, but have 3D coordinates
            @test number_of_cells(embedded_mesh) == length(fracture_faces)  # Each face should become a cell in the embedded mesh
            @test number_of_faces(embedded_mesh) == 18 + 3*6
            @test number_of_boundary_faces(embedded_mesh) == 24
        end
        
        @testset "Connectivity Verification" begin
            umesh = embedded_mesh.unstructured_mesh
                
            nc = number_of_cells(embedded_mesh)
            nf = number_of_faces(embedded_mesh)
            nbf = number_of_boundary_faces(embedded_mesh)
            
            # Basic connectivity checks
            @test length(umesh.faces.cells_to_faces) == nc
            @test length(umesh.boundary_faces.cells_to_faces) == nc
            @test length(umesh.faces.neighbors) == nf
            @test length(umesh.boundary_faces.neighbors) == nbf
            @test all(length.(embedded_mesh.intersections) .== 4)
            
            # Check neighbor consistency
            neighbors = get_neighborship(embedded_mesh)
            @test size(neighbors) == (2, nf)
            @test all(1 .<= neighbors[1, :] .<= nc)
            @test all(1 .<= neighbors[2, :] .<= nc)
            
            # Verify no self-connections
            for i in 1:nf
                @test neighbors[1, i] != neighbors[2, i]
            end
        end
        
        @testset "Geometric Consistency" begin
            nc = number_of_cells(embedded_mesh)
            # Test geometry computation
            for i in 1:nc
                centroid, volume = Jutul.compute_centroid_and_measure(embedded_mesh, Cells(), i)
                centroid_p, area_p = Jutul.compute_centroid_and_measure(parent_mesh, Faces(), fracture_faces[i])
                cell_normal = Jutul.EmbeddedMeshes.cell_normal(embedded_mesh, i)
                face_normal = Jutul.face_normal(parent_mesh, fracture_faces[i], Faces())
                @test all(isapprox.(centroid, centroid_p))
                @test all(isapprox.(volume, area_p))
                @test all(isapprox(abs(dot(cell_normal, face_normal)), 1.0))
            end
        end
    end
    
    @testset "Empty Embedded Mesh" begin
        # Test edge case with no faces (should handle gracefully)
        parent_mesh = UnstructuredMesh(CartesianMesh((2, 2, 2)))
        empty_faces = Int[]
        
        @test_throws Exception EmbeddedMesh(parent_mesh, empty_faces)
    end
    
    @testset "Triangulation and Plotting" begin
        # Test that triangulation works for visualization
        parent_mesh = UnstructuredMesh(CartesianMesh((2, 2, 2), (2.0, 2.0, 2.0)))
        neighbors = get_neighborship(parent_mesh)
        test_faces = [1, 2]
        
        embedded_mesh = Jutul.EmbeddedMeshes.EmbeddedMesh(parent_mesh, test_faces)
        
        # Test triangulation
        triangulation = triangulate_mesh(embedded_mesh; outer = false)
        
        @test triangulation isa NamedTuple
        @test haskey(triangulation, :mapper)
        @test haskey(triangulation, :points)
        @test haskey(triangulation, :triangulation)
        
        # Check that triangulation data is reasonable
        @test size(triangulation.points, 2) == 3  # Embedded mesh coordinates are 3D
        @test size(triangulation.triangulation, 2) == 3  # Triangles have 3 vertices
        
        # Test plot primitives
        primitives = Jutul.plot_primitives(embedded_mesh, :mesh)
        @test primitives !== nothing
    end
    
    @testset "Interface Consistency" begin
        # Test that embedded mesh behaves consistently with the FiniteVolumeMesh interface
        parent_mesh = UnstructuredMesh(CartesianMesh((3, 3, 3)))
        neighbors = get_neighborship(parent_mesh)
        test_faces = [1, 3, 5]
        
        embedded_mesh = Jutul.EmbeddedMeshes.EmbeddedMesh(parent_mesh, test_faces)
        
        # Test count_entities consistency
        @test count_entities(embedded_mesh, Cells()) == number_of_cells(embedded_mesh)
        @test count_entities(embedded_mesh, Faces()) == number_of_faces(embedded_mesh)
        @test count_entities(embedded_mesh, BoundaryFaces()) == number_of_boundary_faces(embedded_mesh)
        
        # Test that we can convert back to UnstructuredMesh
        umesh = UnstructuredMesh(embedded_mesh)
        @test umesh isa UnstructuredMesh
        @test dim(umesh) == dim(embedded_mesh)
        @test number_of_cells(umesh) == number_of_cells(embedded_mesh)
    end
    
    @testset "Transmissibility Computation" begin
        # Test specialized transmissibility calculations for embedded features
        parent_mesh = UnstructuredMesh(CartesianMesh((2, 2, 2), (2.0, 2.0, 2.0)))
        test_faces = [1, 2, 5, 6]
        embedded_mesh = Jutul.EmbeddedMeshes.EmbeddedMesh(parent_mesh, test_faces)
        
        # Set up basic finite volume geometry
        tpfv_geo = tpfv_geometry(embedded_mesh)
        nc = number_of_cells(embedded_mesh)
        nf = number_of_faces(embedded_mesh)
        
        # Test transmissibility computation
        perm_scalar = 1e-12  # 1 mD
        N = get_neighborship(embedded_mesh)
        nc = number_of_cells(embedded_mesh)
        faces, facepos = get_facepos(N, nc)
        T_hf = compute_half_face_trans(
            embedded_mesh,
            tpfv_geo.cell_centroids,
            tpfv_geo.face_centroids,
            tpfv_geo.areas,
            perm_scalar,
            faces,
            facepos
        )
        
        @test length(T_hf) == length(faces)
        @test all(T_hf .== 2.0e-12)  # Transmissibilities should be non-negative
    end
end