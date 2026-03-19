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
        embedded_mesh_remove = Jutul.EmbeddedMeshes.EmbeddedMesh(
            parent_mesh,
            fracture_faces;
            intersection_strategy = :remove
        )
        embedded_mesh_star = Jutul.EmbeddedMeshes.EmbeddedMesh(
            parent_mesh,
            fracture_faces;
            intersection_strategy = :star_delta
        )
        embedded_mesh_keep = Jutul.EmbeddedMeshes.EmbeddedMesh(
            parent_mesh,
            fracture_faces;
            intersection_strategy = :keep
        )
        
        @testset "EmbeddedMesh Construction" begin
            # Create embedded mesh
            
            @test embedded_mesh_remove isa Jutul.EmbeddedMeshes.EmbeddedMesh
            @test embedded_mesh_remove.unstructured_mesh isa Jutul.UnstructuredMesh
            @test embedded_mesh_remove.intersection_neighbors isa Vector{Vector{Int}}
            @test embedded_mesh_remove.intersection_edges isa Vector{Vector{Int}}
            @test embedded_mesh_remove.intersection_cells == Int[]
            @test embedded_mesh_star isa Jutul.EmbeddedMeshes.EmbeddedMesh
            @test embedded_mesh_star.intersection_neighbors == embedded_mesh_remove.intersection_neighbors
            @test all(!isempty, embedded_mesh_star.intersection_edges)
            @test embedded_mesh_star.intersection_cells == Int[]
            @test embedded_mesh_keep isa Jutul.EmbeddedMeshes.EmbeddedMesh
            @test embedded_mesh_keep.intersection_neighbors == embedded_mesh_remove.intersection_neighbors
            @test all(!isempty, embedded_mesh_keep.intersection_edges)
            @test length(embedded_mesh_keep.intersection_cells) == length(embedded_mesh_keep.intersection_neighbors)
            
            # Test basic interface functions
            @test dim(embedded_mesh_remove) == 3
            @test number_of_cells(embedded_mesh_remove) == length(fracture_faces)
            @test number_of_cells(embedded_mesh_star) == length(fracture_faces)
            @test number_of_cells(embedded_mesh_keep) == length(fracture_faces) + length(embedded_mesh_keep.intersection_cells)
            @test number_of_faces(embedded_mesh_remove) < number_of_faces(embedded_mesh_star)
            @test number_of_faces(embedded_mesh_keep) == number_of_faces(embedded_mesh_remove) + sum(length, embedded_mesh_keep.intersection_neighbors)
            @test number_of_boundary_faces(embedded_mesh_remove) > number_of_boundary_faces(embedded_mesh_keep)
        end
        
        @testset "Connectivity Verification" begin
            umesh = embedded_mesh_remove.unstructured_mesh
                
            nc = number_of_cells(embedded_mesh_remove)
            nf = number_of_faces(embedded_mesh_remove)
            nbf = number_of_boundary_faces(embedded_mesh_remove)
            
            # Basic connectivity checks
            @test length(umesh.faces.cells_to_faces) == nc
            @test length(umesh.boundary_faces.cells_to_faces) == nc
            @test length(umesh.faces.neighbors) == nf
            @test length(umesh.boundary_faces.neighbors) == nbf
            @test all(length.(embedded_mesh_remove.intersection_neighbors) .== 4)
            @test all(length.(embedded_mesh_remove.intersection_edges) .== 4)

            umesh_connected = embedded_mesh_star.unstructured_mesh
            @test all(length.(embedded_mesh_star.intersection_neighbors) .== 4)
            @test length(umesh_connected.faces.neighbors) > length(umesh.faces.neighbors)
            
            # Check neighbor consistency
            neighbors = get_neighborship(embedded_mesh_remove)
            @test size(neighbors) == (2, nf)
            @test all(1 .<= neighbors[1, :] .<= nc)
            @test all(1 .<= neighbors[2, :] .<= nc)
            
            # Verify no self-connections
            for i in 1:nf
                @test neighbors[1, i] != neighbors[2, i]
            end

            # Remove strategy should disconnect the intersection into boundary faces.
            for (ix, ix_boundary_faces) in zip(embedded_mesh_remove.intersection_neighbors, embedded_mesh_remove.intersection_edges)
                ix_set = Set(ix)
                has_internal_ix_link = any(eachcol(neighbors)) do n
                    (n[1] in ix_set) && (n[2] in ix_set)
                end
                @test !has_internal_ix_link
                @test length(ix_boundary_faces) == length(ix)
            end

            # Star-delta strategy should include internal links at intersections.
            neighbors_connected = get_neighborship(embedded_mesh_star)
            total_ix_links = 0
            for ix in embedded_mesh_star.intersection_neighbors
                ix_set = Set(ix)
                total_ix_links += count(eachcol(neighbors_connected)) do n
                    (n[1] in ix_set) && (n[2] in ix_set)
                end
            end
            @test total_ix_links > 0

            # Keep strategy should attach duplicated edges to new intersection cells.
            neighbors_keep = get_neighborship(embedded_mesh_keep)
            for (ix_cells, ix_cell) in zip(embedded_mesh_keep.intersection_neighbors, embedded_mesh_keep.intersection_cells)
                connected_faces = findall(f -> ix_cell in neighbors_keep[:, f], 1:size(neighbors_keep, 2))
                connected_cells = Int[]
                for f in connected_faces
                    n = neighbors_keep[:, f]
                    other = n[1] == ix_cell ? n[2] : n[1]
                    push!(connected_cells, other)
                end
                sort!(connected_cells)
                @test connected_cells == sort(ix_cells)
                @test all(count(==(ix_cell), neighbors_keep[:, f]) == 1 for f in connected_faces)
            end
        end
        
        @testset "Geometric Consistency" begin
            nc = number_of_cells(embedded_mesh_remove)
            # Test geometry computation
            for i in 1:nc
                centroid, volume = Jutul.compute_centroid_and_measure(embedded_mesh_remove, Cells(), i)
                centroid_p, area_p = Jutul.compute_centroid_and_measure(parent_mesh, Faces(), fracture_faces[i])
                cell_normal = Jutul.EmbeddedMeshes.cell_normal(embedded_mesh_remove, i)
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