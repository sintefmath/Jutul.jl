export solve_vem_elasticity, VEMElasticitySetup, assemble_vem_elasticity
export boundary_nodes, boundary_nodes_on_side

"""
    VEMCellData

Precomputed per-cell data from mesh traversal, including local node indices,
coordinates, face-to-node connectivity, outward normals, and cell volume.
"""
struct VEMCellData
    "Global node indices for this cell (unique, sorted)"
    node_indices::Vector{Int}
    "Local node coordinates (D × num_nodes)"
    coordinates::Matrix{Float64}
    "Face-to-node connectivity: each entry is a vector of local node indices for a face"
    face_nodes::Vector{Vector{Int}}
    "Outward unit normals for each face (D × num_faces)"
    outward_normals::Matrix{Float64}
    "Cell volume"
    volume::Float64
    "Cell centroid (length D)"
    centroid::Vector{Float64}
end

"""
    VEMElasticitySetup

Precomputed setup data for VEM linear elasticity. Created once from a mesh
and reused for multiple solves with different material parameters.
"""
struct VEMElasticitySetup{D, M}
    "Spatial dimension"
    dim::Int
    "Number of nodes in the mesh"
    num_nodes::Int
    "Number of cells in the mesh"
    num_cells::Int
    "Per-cell precomputed data"
    cell_data::Vector{VEMCellData}
    "Indices of boundary nodes (unique, sorted)"
    boundary_nodes::Vector{Int}
    "Reference to the mesh (needed for face-based pressure RHS assembly)"
    mesh::M
end

"""
    VEMElasticitySetup(mesh::UnstructuredMesh)

Create a VEM elasticity setup by traversing the mesh once and precomputing
all geometric data needed for assembly.
"""
function VEMElasticitySetup(mesh::UnstructuredMesh)
    D = dim(mesh)
    @assert D == 2 || D == 3 "VEM elasticity only supports 2D and 3D meshes"
    nc = number_of_cells(mesh)
    nn = length(mesh.node_points)
    pts = mesh.node_points

    cell_data = Vector{VEMCellData}(undef, nc)
    for cell in 1:nc
        cell_data[cell] = _precompute_cell_data(mesh, cell, D)
    end

    # Identify boundary nodes
    boundary_nodes = _find_boundary_nodes(mesh)

    return VEMElasticitySetup{D, typeof(mesh)}(D, nn, nc, cell_data, boundary_nodes, mesh)
end

"""
    assemble_vem_elasticity(setup, youngs_modulus, poisson_ratio, pressure_change; biot_coefficient=1.0, boundary_displacement=nothing)

Assemble the VEM linear elasticity system (stiffness matrix and RHS) for given
per-cell material properties and pressure change.

Returns `(K, rhs)` where `K` is the sparse stiffness matrix and `rhs` is the
right-hand side vector, both with boundary conditions applied.

# Keyword arguments
- `biot_coefficient`: Biot coefficient (scalar or per-cell vector), default 1.0.
- `boundary_displacement`: If `nothing` (default), zero displacement on all boundary nodes.
  If a vector of length `D * num_nodes`, the values at boundary DOFs are used as prescribed displacement.
"""
function assemble_vem_elasticity(
    setup::VEMElasticitySetup{D},
    youngs_modulus::AbstractVector,
    poisson_ratio::AbstractVector,
    pressure_change::AbstractVector;
    biot_coefficient::Union{Real, AbstractVector} = 1.0,
    boundary_displacement::Union{Nothing, AbstractVector} = nothing
) where {D}
    nc = setup.num_cells
    nn = setup.num_nodes
    ndof = D * nn

    @assert length(youngs_modulus) == nc
    @assert length(poisson_ratio) == nc
    @assert length(pressure_change) == nc

    # Prepare triplets for sparse matrix assembly
    I_idx = Int[]
    J_idx = Int[]
    V_val = Float64[]

    # Right-hand side
    rhs = zeros(Float64, ndof)

    # Get biot coefficient per cell
    biot = biot_coefficient isa Real ? fill(Float64(biot_coefficient), nc) : Float64.(biot_coefficient)

    for cell in 1:nc
        cd = setup.cell_data[cell]
        num_nodes_cell = length(cd.node_indices)
        local_ndof = D * num_nodes_cell

        # Compute local stiffness matrix
        K_local = _assemble_local_stiffness(cd, youngs_modulus[cell], poisson_ratio[cell], D)

        # Scatter local stiffness matrix to global
        for i in 1:local_ndof
            gi = D * (cd.node_indices[(i - 1) ÷ D + 1] - 1) + (i - 1) % D + 1
            for j in 1:local_ndof
                gj = D * (cd.node_indices[(j - 1) ÷ D + 1] - 1) + (j - 1) % D + 1
                val = K_local[i, j]
                if val != 0.0
                    push!(I_idx, gi)
                    push!(J_idx, gj)
                    push!(V_val, val)
                end
            end
        end
    end

    # Assemble pressure RHS using face-based approach
    _assemble_pressure_rhs!(rhs, setup, pressure_change, biot, D)

    # Determine boundary DOFs and prescribed values
    bnd_dof_set = Set{Int}()
    bnd_dof_vals = Dict{Int, Float64}()
    for node in setup.boundary_nodes
        for d in 1:D
            dof = D * (node - 1) + d
            push!(bnd_dof_set, dof)
            if boundary_displacement !== nothing
                bnd_dof_vals[dof] = boundary_displacement[dof]
            else
                bnd_dof_vals[dof] = 0.0
            end
        end
    end

    # Build the full (unconstrained) stiffness matrix first to compute
    # RHS modification for nonzero BC
    K_full = sparse(I_idx, J_idx, V_val, ndof, ndof)

    # Modify RHS for nonzero boundary displacement:
    # For each boundary DOF j with prescribed value u_j,
    # subtract K[:,j] * u_j from the RHS for interior DOFs
    for (dof, val) in bnd_dof_vals
        if val != 0.0
            col = K_full[:, dof]
            rows = rowvals(col)
            vals = nonzeros(col)
            for k in eachindex(rows)
                row = rows[k]
                if !(row in bnd_dof_set)
                    rhs[row] -= vals[k] * val
                end
            end
        end
    end

    # Filter out entries involving boundary DOFs, then add identity for boundary DOFs
    I_final = Int[]
    J_final = Int[]
    V_final = Float64[]
    for k in eachindex(I_idx)
        i = I_idx[k]
        j = J_idx[k]
        if !(i in bnd_dof_set) && !(j in bnd_dof_set)
            push!(I_final, i)
            push!(J_final, j)
            push!(V_final, V_val[k])
        end
    end
    # Add identity for boundary DOFs with prescribed values
    for (dof, val) in bnd_dof_vals
        push!(I_final, dof)
        push!(J_final, dof)
        push!(V_final, 1.0)
        rhs[dof] = val
    end

    # Assemble sparse matrix
    K = sparse(I_final, J_final, V_final, ndof, ndof)

    return (K, rhs)
end

"""
    solve_vem_elasticity(mesh, youngs_modulus, poisson_ratio, pressure_change; biot_coefficient=1.0, boundary_displacement=nothing)

Solve the VEM linear elasticity problem on an UnstructuredMesh with zero
displacement at the boundaries (or prescribed nonzero boundary displacement).

# Arguments
- `mesh::UnstructuredMesh`: The mesh to solve on.
- `youngs_modulus::AbstractVector`: Young's modulus per cell.
- `poisson_ratio::AbstractVector`: Poisson's ratio per cell.
- `pressure_change::AbstractVector`: Pressure change per cell.
- `biot_coefficient`: Biot coefficient (scalar or per-cell vector), default 1.0.
- `boundary_displacement`: If `nothing` (default), zero displacement on all boundary nodes.
  If a vector of length `D * num_nodes`, the values at boundary DOFs are used as prescribed displacement.

# Returns
A named tuple `(displacement, setup, K, rhs)` where:
- `displacement`: nodal displacement matrix of size `D × num_nodes` (e.g. `3 × N` in 3D).
- `setup`: the `VEMElasticitySetup` for reuse.
- `K`: the assembled sparse stiffness matrix.
- `rhs`: the right-hand side vector.

# Example
```julia
mesh = UnstructuredMesh(CartesianMesh((3, 3, 3), (1.0, 1.0, 1.0)))
nc = number_of_cells(mesh)
result = solve_vem_elasticity(
    mesh,
    fill(1e9, nc),    # Young's modulus
    fill(0.3, nc),    # Poisson's ratio
    fill(1e6, nc)     # Pressure change
)
u = result.displacement  # 3 × num_nodes matrix
```
"""
function solve_vem_elasticity(
    mesh::UnstructuredMesh,
    youngs_modulus::AbstractVector,
    poisson_ratio::AbstractVector,
    pressure_change::AbstractVector;
    biot_coefficient::Union{Real, AbstractVector} = 1.0,
    boundary_displacement::Union{Nothing, AbstractVector} = nothing
)
    setup = VEMElasticitySetup(mesh)
    return solve_vem_elasticity(setup, youngs_modulus, poisson_ratio, pressure_change;
        biot_coefficient = biot_coefficient, boundary_displacement = boundary_displacement)
end

"""
    solve_vem_elasticity(setup, youngs_modulus, poisson_ratio, pressure_change; biot_coefficient=1.0, boundary_displacement=nothing)

Solve with a precomputed `VEMElasticitySetup` (useful for repeated solves).
"""
function solve_vem_elasticity(
    setup::VEMElasticitySetup{D},
    youngs_modulus::AbstractVector,
    poisson_ratio::AbstractVector,
    pressure_change::AbstractVector;
    biot_coefficient::Union{Real, AbstractVector} = 1.0,
    boundary_displacement::Union{Nothing, AbstractVector} = nothing
) where {D}
    K, rhs = assemble_vem_elasticity(setup, youngs_modulus, poisson_ratio, pressure_change;
        biot_coefficient = biot_coefficient, boundary_displacement = boundary_displacement)
    u_vec = K \ rhs
    displacement = reshape(u_vec, D, setup.num_nodes)
    return (displacement = displacement, setup = setup, K = K, rhs = rhs)
end

# ============================================================================
# Internal helper functions
# ============================================================================

"""
Precompute cell data for a single cell by traversing its faces (interior + boundary).
"""
function _precompute_cell_data(mesh::UnstructuredMesh, cell::Int, D::Int)
    pts = mesh.node_points

    # Collect all unique global node indices for this cell
    global_nodes = Set{Int}()

    # Collect face-to-node connectivity (global indices)
    face_nodes_global = Vector{Vector{Int}}()

    # Interior faces
    for face in mesh.faces.cells_to_faces[cell]
        nodes = collect(mesh.faces.faces_to_nodes[face])
        for n in nodes
            push!(global_nodes, n)
        end
        push!(face_nodes_global, nodes)
    end

    # Boundary faces
    for bf in mesh.boundary_faces.cells_to_faces[cell]
        nodes = collect(mesh.boundary_faces.faces_to_nodes[bf])
        for n in nodes
            push!(global_nodes, n)
        end
        push!(face_nodes_global, nodes)
    end

    # Sort global node indices and create global-to-local map
    sorted_nodes = sort(collect(global_nodes))
    num_nodes = length(sorted_nodes)
    g2l = Dict{Int, Int}()
    for (li, gi) in enumerate(sorted_nodes)
        g2l[gi] = li
    end

    # Local coordinates matrix (D × num_nodes)
    coords = zeros(Float64, D, num_nodes)
    for (li, gi) in enumerate(sorted_nodes)
        for d in 1:D
            coords[d, li] = pts[gi][d]
        end
    end

    # Convert face-to-node connectivity to local indices
    face_nodes_local = Vector{Vector{Int}}(undef, length(face_nodes_global))
    for (fi, fnodes) in enumerate(face_nodes_global)
        face_nodes_local[fi] = [g2l[n] for n in fnodes]
    end

    # Compute outward normals for each face
    nfaces = length(face_nodes_local)
    outward_normals = zeros(Float64, D, nfaces)

    if D == 3
        _compute_outward_normals_3d!(outward_normals, face_nodes_local, coords, mesh, cell)
    else
        _compute_outward_normals_2d!(outward_normals, face_nodes_local, coords, mesh, cell)
    end

    # Compute cell centroid and volume
    centroid, volume = _compute_cell_centroid_and_volume(coords, face_nodes_local, outward_normals, D)

    return VEMCellData(sorted_nodes, coords, face_nodes_local, outward_normals, volume, centroid)
end

"""
Compute outward normals for 3D faces. The normals are unit normals pointing
outward from the cell.
"""
function _compute_outward_normals_3d!(normals, face_nodes, coords, mesh, cell)
    nfaces = length(face_nodes)
    # Compute face centroids and normals
    face_centroids = zeros(Float64, 3, nfaces)
    for fi in 1:nfaces
        fn = face_nodes[fi]
        nfn = length(fn)
        # Face centroid (simple average of nodes)
        fc = zeros(3)
        for ni in fn
            fc .+= coords[:, ni]
        end
        fc ./= nfn
        face_centroids[:, fi] .= fc

        # Compute normal via weighted cross-product sum
        normal = zeros(3)
        for i in 1:nfn
            prev_i = i == 1 ? nfn : i - 1
            next_i = i == nfn ? 1 : i + 1
            a = coords[:, fn[prev_i]] .- coords[:, fn[i]]
            c = coords[:, fn[next_i]] .- coords[:, fn[i]]
            normal .+= cross(c, a)
        end
        nn = norm(normal)
        if nn > 0
            normal ./= nn
        end
        normals[:, fi] .= normal
    end

    # Check orientation per face: each normal should point outward from the cell
    # Compute cell center (average of all node coordinates)
    num_nodes = size(coords, 2)
    cell_center = zeros(3)
    for i in 1:num_nodes
        cell_center .+= coords[:, i]
    end
    cell_center ./= num_nodes

    # Flip individual normals that point inward
    for fi in 1:nfaces
        v = face_centroids[:, fi] .- cell_center
        if dot(v, normals[:, fi]) < 0
            normals[:, fi] .*= -1
        end
    end
end

"""
Compute outward normals for 2D faces (edges).
"""
function _compute_outward_normals_2d!(normals, face_nodes, coords, mesh, cell)
    nfaces = length(face_nodes)
    face_centroids = zeros(Float64, 2, nfaces)
    for fi in 1:nfaces
        fn = face_nodes[fi]
        @assert length(fn) == 2 "2D faces must have exactly 2 nodes"
        p1 = coords[:, fn[1]]
        p2 = coords[:, fn[2]]
        # Edge midpoint
        fc = (p1 + p2) / 2
        face_centroids[:, fi] .= fc
        # Normal is perpendicular to edge
        edge = p2 - p1
        normal = [edge[2], -edge[1]]
        nn = norm(normal)
        if nn > 0
            normal ./= nn
        end
        normals[:, fi] .= normal
    end

    # Check orientation per face
    num_nodes = size(coords, 2)
    cell_center = zeros(2)
    for i in 1:num_nodes
        cell_center .+= coords[:, i]
    end
    cell_center ./= num_nodes

    for fi in 1:nfaces
        v = face_centroids[:, fi] .- cell_center
        if dot(v, normals[:, fi]) < 0
            normals[:, fi] .*= -1
        end
    end
end

"""
Compute cell centroid and volume from local face-node data and outward normals.
Uses the divergence theorem: V = (1/D) ∫_∂E x · n dA
"""
function _compute_cell_centroid_and_volume(coords, face_nodes, outward_normals, D)
    if D == 3
        return _compute_cell_geometry_3d(coords, face_nodes, outward_normals)
    else
        return _compute_cell_geometry_2d(coords, face_nodes, outward_normals)
    end
end

function _compute_cell_geometry_3d(coords, face_nodes, normals)
    # Use the average of all node points as an "inside point"
    num_nodes = size(coords, 2)
    inside_pt = zeros(3)
    for i in 1:num_nodes
        inside_pt .+= coords[:, i]
    end
    inside_pt ./= num_nodes

    volume = 0.0
    centroid = zeros(3)

    for fi in eachindex(face_nodes)
        fn = face_nodes[fi]
        nfn = length(fn)
        # Compute face centroid
        fc = zeros(3)
        for ni in fn
            fc .+= coords[:, ni]
        end
        fc ./= nfn

        # Tessellate face into triangles using face centroid
        for i in 1:nfn
            next_i = i == nfn ? 1 : i + 1
            # Triangle: fc, fn[i], fn[next_i]
            p1 = coords[:, fn[i]]
            p2 = coords[:, fn[next_i]]

            # Tetrahedron volume: V = |det([p1-p4, p2-p4, p3-p4])| / 6
            v1 = fc .- inside_pt
            v2 = p1 .- inside_pt
            v3 = p2 .- inside_pt
            tet_vol = abs(dot(v1, cross(v2, v3))) / 6.0
            tet_centroid = (fc .+ p1 .+ p2 .+ inside_pt) ./ 4.0

            volume += tet_vol
            centroid .+= tet_vol .* tet_centroid
        end
    end

    if volume > 0
        centroid ./= volume
    end

    return (centroid, volume)
end

function _compute_cell_geometry_2d(coords, face_nodes, normals)
    num_nodes = size(coords, 2)
    inside_pt = zeros(2)
    for i in 1:num_nodes
        inside_pt .+= coords[:, i]
    end
    inside_pt ./= num_nodes

    area = 0.0
    centroid = zeros(2)

    for fi in eachindex(face_nodes)
        fn = face_nodes[fi]
        @assert length(fn) == 2
        p1 = coords[:, fn[1]]
        p2 = coords[:, fn[2]]

        # Triangle: inside_pt, p1, p2
        A = p1 .- inside_pt
        B = p2 .- inside_pt
        tri_area = abs(A[1] * B[2] - A[2] * B[1]) / 2.0
        tri_centroid = (inside_pt .+ p1 .+ p2) ./ 3.0

        area += tri_area
        centroid .+= tri_area .* tri_centroid
    end

    if area > 0
        centroid ./= area
    end

    return (centroid, area)
end

"""
Find all nodes on the boundary of the mesh.
"""
function _find_boundary_nodes(mesh::UnstructuredMesh)
    boundary_nodes = Set{Int}()
    nbf = number_of_boundary_faces(mesh)
    for bf in 1:nbf
        for node in mesh.boundary_faces.faces_to_nodes[bf]
            push!(boundary_nodes, node)
        end
    end
    return sort(collect(boundary_nodes))
end

"""
    boundary_nodes(setup::VEMElasticitySetup)

Return a sorted vector of global node indices that lie on the boundary of the mesh.

# Example
```julia
setup = VEMElasticitySetup(mesh)
bnodes = boundary_nodes(setup)
```
"""
boundary_nodes(setup::VEMElasticitySetup) = setup.boundary_nodes

"""
    boundary_nodes_on_side(setup::VEMElasticitySetup, direction::Symbol; tol=1e-10)

Return a sorted vector of boundary node indices that lie on a specific side of
the bounding box of the mesh. The `direction` symbol must be one of:
`:xmin`, `:xmax`, `:ymin`, `:ymax` (2D/3D), `:zmin`, `:zmax` (3D only).

An optional tolerance `tol` is used for the coordinate comparison.

# Example
```julia
setup = VEMElasticitySetup(mesh)
# Get boundary nodes on the x=0 face
left_nodes = boundary_nodes_on_side(setup, :xmin)
# Get boundary nodes on the y=ymax face
top_nodes = boundary_nodes_on_side(setup, :ymax)
```
"""
function boundary_nodes_on_side(setup::VEMElasticitySetup{D}, direction::Symbol; tol::Real = 1e-10) where {D}
    dir_map = Dict(
        :xmin => (1, minimum),
        :xmax => (1, maximum),
        :ymin => (2, minimum),
        :ymax => (2, maximum),
        :zmin => (3, minimum),
        :zmax => (3, maximum)
    )

    @assert haskey(dir_map, direction) "Unknown direction $direction. Use one of: :xmin, :xmax, :ymin, :ymax" * (D == 3 ? ", :zmin, :zmax" : "")
    dim_idx, extremum_fn = dir_map[direction]
    @assert dim_idx <= D "Direction $direction not valid for $(D)D mesh"

    pts = setup.mesh.node_points
    bnd = setup.boundary_nodes

    # Find the extreme value among boundary nodes in the given dimension
    ext_val = extremum_fn(pts[n][dim_idx] for n in bnd)

    # Filter boundary nodes that are at this extreme value
    return sort([n for n in bnd if abs(pts[n][dim_idx] - ext_val) <= tol])
end

# ============================================================================
# VEM stiffness matrix assembly per cell
# ============================================================================

"""
Compute the elasticity tensor D in modified Voigt notation (with factor 2 on shear terms).
"""
function _compute_D(young::Float64, poisson::Float64, D_dim::Int)
    fac = young / (1 + poisson) / (1 - 2 * poisson)
    if D_dim == 2
        # 3x3 matrix (plane strain, modified Voigt notation)
        D = zeros(3, 3)
        D[1, 1] = (1 - poisson) * fac
        D[1, 2] = poisson * fac
        D[2, 1] = poisson * fac
        D[2, 2] = (1 - poisson) * fac
        D[3, 3] = 2 * (1 - 2 * poisson) * fac
    else
        # 6x6 matrix (3D, modified Voigt notation)
        D = zeros(6, 6)
        for i in 1:3
            D[i, i] = (1 - poisson) * fac
        end
        D[1, 2] = poisson * fac; D[2, 1] = poisson * fac
        D[1, 3] = poisson * fac; D[3, 1] = poisson * fac
        D[2, 3] = poisson * fac; D[3, 2] = poisson * fac
        D[4, 4] = 2 * (1 - 2 * poisson) * fac
        D[5, 5] = 2 * (1 - 2 * poisson) * fac
        D[6, 6] = 2 * (1 - 2 * poisson) * fac
    end
    return D
end

"""
Compute q-values for 2D (related to VBF integrals over edges).
q_i = 1/(2A) ∫_∂E φ_i n dA

For 2D: q is a (num_nodes × 2) matrix.
For each edge, ∫_edge φ_i n dA = (edge_length/2) * n for each of the 2 endpoint nodes.
"""
function _compute_q_2d(coords, face_nodes, outward_normals, volume)
    num_nodes = size(coords, 2)
    q = zeros(num_nodes, 2)
    fac = 1.0 / (2.0 * volume)

    for fi in eachindex(face_nodes)
        fn = face_nodes[fi]
        @assert length(fn) == 2
        i = fn[1]
        inext = fn[2]
        # Edge length
        edge_len = norm(coords[:, inext] .- coords[:, i])
        # Area-weighted normal = edge_length * unit_outward_normal
        # Each endpoint gets half the integral
        for d in 1:2
            contrib = fac * (edge_len / 2.0) * outward_normals[d, fi]
            q[i, d] += contrib
            q[inext, d] += contrib
        end
    end
    return q
end

"""
Compute q-values for 3D (related to VBF integrals over faces).
q_i = 1/(2V) ∫_∂E φ_i n dA
"""
function _compute_q_3d(coords, face_nodes, outward_normals, volume)
    num_nodes = size(coords, 2)
    q = zeros(num_nodes, 3)
    fac = 1.0 / (2.0 * volume)

    for fi in eachindex(face_nodes)
        fn = face_nodes[fi]
        nfn = length(fn)
        normal = outward_normals[:, fi]

        # Compute face integration: each basis function φ_i integrated over the face
        # Using tessellation into triangles from face centroid
        fc = zeros(3)
        for ni in fn
            fc .+= coords[:, ni]
        end
        fc ./= nfn

        for e in 1:nfn
            enext = e == nfn ? 1 : e + 1
            # Compute integrals of φ for two sub-triangles associated with edge (e, enext)
            # Sub-triangle 1: fc, coords[fn[e]], midpoint
            # Sub-triangle 2: fc, midpoint, coords[fn[enext]]
            p1 = coords[:, fn[e]]
            p2 = coords[:, fn[enext]]
            mid = (p1 .+ p2) ./ 2.0

            # Triangle areas using cross product
            # Triangle 1: fc, p1, mid
            v1 = p1 .- fc
            v2 = mid .- fc
            area1 = norm(cross(v1, v2)) / 2.0

            # Triangle 2: fc, mid, p2
            v1 = mid .- fc
            v2 = p2 .- fc
            area2 = norm(cross(v1, v2)) / 2.0

            # φ_e is 1 at node e, 0 at others. Under tessellation, the integral
            # of φ_e over its associated sub-triangles gives the area.
            for d in 1:3
                q[fn[e], d] += fac * area1 * normal[d]
                q[fn[enext], d] += fac * area2 * normal[d]
            end
        end
    end
    return q
end

"""
Create a 2D matentry (2 × 3 sub-matrix for node i).
Returns the entries in row-major order: [e1 0 e2; 0 e3 e4]
"""
function _matentry_2d(e1, e2, e3, e4)
    return [e1 0.0 e2; 0.0 e3 e4]
end

"""
Create a 3D matentry (3 × 6 sub-matrix for node i).
Returns [e1 0 0 e2 0 e3; 0 e4 0 e5 e6 0; 0 0 e7 0 e8 e9]
"""
function _matentry_3d(e1, e2, e3, e4, e5, e6, e7, e8, e9)
    return [e1 0.0 0.0 e2 0.0 e3;
            0.0 e4 0.0 e5 e6 0.0;
            0.0 0.0 e7 0.0 e8 e9]
end

"""
Compute Nr matrix (rigid body modes) for 2D.
Nr is (2*num_nodes) × 3.
"""
function _compute_Nr_2d(coords)
    num_nodes = size(coords, 2)
    midpoint = vec(mean(coords, dims = 2))
    Nr = zeros(2 * num_nodes, 3)
    for i in 1:num_nodes
        dx = coords[1, i] - midpoint[1]
        dy = coords[2, i] - midpoint[2]
        rows = (2 * (i - 1) + 1):(2 * i)
        Nr[rows, :] .= _matentry_2d(1.0, dy, 1.0, -dx)
    end
    return Nr
end

"""
Compute Nr matrix (rigid body modes) for 3D.
Nr is (3*num_nodes) × 6.
"""
function _compute_Nr_3d(coords)
    num_nodes = size(coords, 2)
    midpoint = vec(mean(coords, dims = 2))
    Nr = zeros(3 * num_nodes, 6)
    for i in 1:num_nodes
        dx = coords[1, i] - midpoint[1]
        dy = coords[2, i] - midpoint[2]
        dz = coords[3, i] - midpoint[3]
        rows = (3 * (i - 1) + 1):(3 * i)
        Nr[rows, :] .= _matentry_3d(1.0, dy, -dz, 1.0, -dx, dz, 1.0, -dy, dx)
    end
    return Nr
end

"""
Compute Nc matrix (linear deformation modes) for 2D.
Nc is (2*num_nodes) × 3.
"""
function _compute_Nc_2d(coords)
    num_nodes = size(coords, 2)
    midpoint = vec(mean(coords, dims = 2))
    Nc = zeros(2 * num_nodes, 3)
    for i in 1:num_nodes
        dx = coords[1, i] - midpoint[1]
        dy = coords[2, i] - midpoint[2]
        rows = (2 * (i - 1) + 1):(2 * i)
        Nc[rows, :] .= _matentry_2d(dx, dy, dy, dx)
    end
    return Nc
end

"""
Compute Nc matrix (linear deformation modes) for 3D.
Nc is (3*num_nodes) × 6.
"""
function _compute_Nc_3d(coords)
    num_nodes = size(coords, 2)
    midpoint = vec(mean(coords, dims = 2))
    Nc = zeros(3 * num_nodes, 6)
    for i in 1:num_nodes
        dx = coords[1, i] - midpoint[1]
        dy = coords[2, i] - midpoint[2]
        dz = coords[3, i] - midpoint[3]
        rows = (3 * (i - 1) + 1):(3 * i)
        Nc[rows, :] .= _matentry_3d(dx, dy, dz, dy, dx, dz, dz, dy, dx)
    end
    return Nc
end

"""
Compute Wr matrix for 2D (from q-values + rigid body rotation).
Wr is (2*num_nodes) × 3.
"""
function _compute_Wr_2d(q)
    num_nodes = size(q, 1)
    ncinv = 1.0 / num_nodes
    Wr = zeros(2 * num_nodes, 3)
    for i in 1:num_nodes
        rows = (2 * (i - 1) + 1):(2 * i)
        Wr[rows, :] .= _matentry_2d(ncinv, q[i, 2], ncinv, -q[i, 1])
    end
    return Wr
end

"""
Compute Wr matrix for 3D.
"""
function _compute_Wr_3d(q)
    num_nodes = size(q, 1)
    ncinv = 1.0 / num_nodes
    Wr = zeros(3 * num_nodes, 6)
    for i in 1:num_nodes
        rows = (3 * (i - 1) + 1):(3 * i)
        Wr[rows, :] .= _matentry_3d(ncinv, q[i, 2], -q[i, 3], ncinv, -q[i, 1], q[i, 3], ncinv, -q[i, 2], q[i, 1])
    end
    return Wr
end

"""
Compute Wc matrix for 2D.
"""
function _compute_Wc_2d(q)
    num_nodes = size(q, 1)
    Wc = zeros(2 * num_nodes, 3)
    for i in 1:num_nodes
        rows = (2 * (i - 1) + 1):(2 * i)
        Wc[rows, :] .= _matentry_2d(2 * q[i, 1], q[i, 2], 2 * q[i, 2], q[i, 1])
    end
    return Wc
end

"""
Compute Wc matrix for 3D.
"""
function _compute_Wc_3d(q)
    num_nodes = size(q, 1)
    Wc = zeros(3 * num_nodes, 6)
    for i in 1:num_nodes
        rows = (3 * (i - 1) + 1):(3 * i)
        Wc[rows, :] .= _matentry_3d(
            2 * q[i, 1], q[i, 2], q[i, 3],
            2 * q[i, 2], q[i, 1], q[i, 3],
            2 * q[i, 3], q[i, 2], q[i, 1]
        )
    end
    return Wc
end

"""
Compute the VEM stability term S.
Based on Gain et al. 2014, DOI:10.1016/j.cma.2014.05.005
"""
function _compute_S(Nc, D_mat, num_nodes, volume, D_dim)
    lsdim = D_dim == 2 ? 3 : 6
    r = D_dim * num_nodes
    NtN = Nc' * Nc
    trNtN = tr(NtN)
    if trNtN < eps(Float64)
        # Fallback for degenerate geometry
        alpha = volume * tr(D_mat)
    else
        alpha = volume * tr(D_mat) / trNtN
    end
    return alpha * I(r)
end

"""
Compute I - P where P = Nr * Wr' + Nc * Wc'.
"""
function _compute_ImP(Nr, Nc, Wr, Wc)
    r = size(Nr, 1)
    Pr = Nr * Wr'
    Pc = Nc * Wc'
    return Matrix{Float64}(I, r, r) - Pr - Pc
end

"""
Assemble the local VEM stiffness matrix for a single cell.
"""
function _assemble_local_stiffness(cd::VEMCellData, young::Float64, poisson::Float64, D_dim::Int)
    coords = cd.coordinates
    face_nodes = cd.face_nodes
    outward_normals = cd.outward_normals
    volume = cd.volume
    num_nodes = length(cd.node_indices)

    # Compute elasticity tensor
    D_mat = _compute_D(young, poisson, D_dim)

    # Compute q-values
    if D_dim == 2
        q = _compute_q_2d(coords, face_nodes, outward_normals, volume)
    else
        q = _compute_q_3d(coords, face_nodes, outward_normals, volume)
    end

    # Compute matrices
    if D_dim == 2
        Nr = _compute_Nr_2d(coords)
        Nc = _compute_Nc_2d(coords)
        Wr = _compute_Wr_2d(q)
        Wc = _compute_Wc_2d(q)
    else
        Nr = _compute_Nr_3d(coords)
        Nc = _compute_Nc_3d(coords)
        Wr = _compute_Wr_3d(q)
        Wc = _compute_Wc_3d(q)
    end

    ImP = _compute_ImP(Nr, Nc, Wr, Wc)
    S = _compute_S(Nc, D_mat, num_nodes, volume, D_dim)

    # Final assembly: K = V * Wc * D * Wc' + (I-P)' * S * (I-P)
    lsdim = D_dim == 2 ? 3 : 6
    totdim = D_dim * num_nodes

    DWct = D_mat * Wc'
    EWcDWct = volume * (Wc * DWct)

    SImP = S * ImP
    ImPtSImP = ImP' * SImP

    K_local = EWcDWct + ImPtSImP

    return K_local
end

"""
Assemble the pressure contribution to the RHS using face-based approach.

For piecewise-constant pressure, the weak form contribution is:
  rhs_i = −∑_f α·(p_L − p_R)·∫_f φ_i n dA     (interior faces)
        − ∑_f α·p·∫_f φ_i n dA                   (boundary faces)

This correctly gives zero RHS when pressure is uniform, because p_L == p_R
on every interior face cancels exactly, and boundary face contributions 
cancel by the divergence theorem (∑_bnd_faces ∫_f n dA = 0 for each node's
"share" when pressure is constant).

Actually for the VEM formulation with piecewise-constant pressure in cells,
the body force from pressure in a cell is:
  f = -α * ∇p * V ≈ -α * ∑_faces (p_face * n * A_face)
where the sum is over cell faces with outward normals.

The face-based assembly distributes -α*p*∫_f φ_i n dA for each face,
where the sign accounts for which cell "owns" the normal direction.
For interior faces shared by cells L and R:
  - Cell L sees outward normal n_f, contributes -α_L * p_L * ∫_f φ_i n dA
  - Cell R sees outward normal -n_f, contributes +α_R * p_R * ∫_f φ_i n dA
  => net = -∫_f φ_i n dA * (α_L * p_L - α_R * p_R)
When p_L == p_R and α_L == α_R, this is zero.

For boundary faces, only one cell contributes.
"""
function _assemble_pressure_rhs!(rhs, setup::VEMElasticitySetup{D}, pressure_change, biot, D_dim) where {D}
    mesh = setup.mesh
    pts = mesh.node_points

    # Process interior faces
    nf = number_of_faces(mesh)
    for face in 1:nf
        L, R = mesh.faces.neighbors[face]
        biot_dp_L = biot[L] * pressure_change[L]
        biot_dp_R = biot[R] * pressure_change[R]
        dp_jump = biot_dp_L - biot_dp_R
        if dp_jump == 0.0
            continue
        end

        # Get face nodes (global indices)
        face_node_indices = collect(mesh.faces.faces_to_nodes[face])

        # Compute face normal (oriented from L to R by convention in Jutul)
        face_normal_vec = _compute_face_normal_from_nodes(face_node_indices, pts, D_dim)

        # Orient: Jutul stores normals pointing from L to R (first to second neighbor)
        # We need outward normal from L, which is face_normal_vec
        # Verify orientation using cell centroids
        cL = setup.cell_data[L].centroid
        cR = setup.cell_data[R].centroid
        LR = zeros(D_dim)
        for d in 1:D_dim
            LR[d] = cR[d] - cL[d]
        end
        if dot(face_normal_vec, LR) < 0
            face_normal_vec = -face_normal_vec
        end

        # Compute ∫_f φ_i n dA for each face node
        # For the contribution: rhs -= dp_jump * ∫_f φ_i n dA
        _add_face_pressure_contribution!(rhs, face_node_indices, pts, face_normal_vec, dp_jump, D_dim)
    end

    # Process boundary faces
    nbf = number_of_boundary_faces(mesh)
    for bf in 1:nbf
        cell = mesh.boundary_faces.neighbors[bf]
        biot_dp = biot[cell] * pressure_change[cell]
        if biot_dp == 0.0
            continue
        end

        face_node_indices = collect(mesh.boundary_faces.faces_to_nodes[bf])
        face_normal_vec = _compute_face_normal_from_nodes(face_node_indices, pts, D_dim)

        # Orient outward from cell
        cc = setup.cell_data[cell].centroid
        fc = zeros(D_dim)
        for ni in face_node_indices
            for d in 1:D_dim
                fc[d] += pts[ni][d]
            end
        end
        fc ./= length(face_node_indices)
        outward = fc .- cc
        if dot(face_normal_vec, outward) < 0
            face_normal_vec = -face_normal_vec
        end

        # Boundary face: only one cell contributes
        # rhs -= biot_dp * ∫_f φ_i n dA
        _add_face_pressure_contribution!(rhs, face_node_indices, pts, face_normal_vec, biot_dp, D_dim)
    end
end

"""
Compute face normal vector (unnormalized, area-weighted) from global node indices.
"""
function _compute_face_normal_from_nodes(node_indices, pts, D_dim)
    if D_dim == 2
        @assert length(node_indices) == 2
        p1 = pts[node_indices[1]]
        p2 = pts[node_indices[2]]
        edge = [p2[1] - p1[1], p2[2] - p1[2]]
        return [edge[2], -edge[1]]  # unnormalized normal (length = edge length)
    else
        # 3D: compute normal via cross-product sum over polygon
        nfn = length(node_indices)
        fc = zeros(3)
        for ni in node_indices
            for d in 1:3
                fc[d] += pts[ni][d]
            end
        end
        fc ./= nfn

        normal = zeros(3)
        for i in 1:nfn
            prev_i = i == 1 ? nfn : i - 1
            next_i = i == nfn ? 1 : i + 1
            pp = [pts[node_indices[prev_i]][d] for d in 1:3]
            pc = [pts[node_indices[i]][d] for d in 1:3]
            pn = [pts[node_indices[next_i]][d] for d in 1:3]
            a = pp .- pc
            c = pn .- pc
            normal .+= cross(c, a)
        end
        # This gives 2 * area * unit_normal
        # We want area-weighted normal, so divide by 2
        return normal ./ 2.0
    end
end

"""
Add the face pressure contribution −dp_jump * ∫_f φ_i n_d dA to the RHS
for each node on the face and each direction d.

For 2D faces (edges): ∫_f φ_i n dA = (edge_length / 2) * unit_normal for each of the 2 nodes
  = (1/2) * area_normal_vec (since area_normal_vec = edge_length * unit_normal)

For 3D faces: we tessellate into triangles from face centroid and distribute
to nodes proportional to their sub-triangle areas.
"""
function _add_face_pressure_contribution!(rhs, node_indices, pts, area_normal_vec, dp_jump, D_dim)
    nfn = length(node_indices)

    if D_dim == 2
        # 2D edge: equal distribution to 2 nodes, each gets half the face integral
        for ni in node_indices
            for d in 1:D_dim
                rhs[D_dim * (ni - 1) + d] -= dp_jump * 0.5 * area_normal_vec[d]
            end
        end
    else
        # 3D face: tessellate into triangles from face centroid
        # Each node's share is proportional to the sub-triangle areas adjacent to it
        fc = zeros(3)
        for ni in node_indices
            for d in 1:3
                fc[d] += pts[ni][d]
            end
        end
        fc ./= nfn

        # Unit normal direction
        nn = norm(area_normal_vec)
        if nn < eps(Float64)
            return
        end
        unit_normal = area_normal_vec ./ nn

        # Compute sub-triangle areas for each node
        node_areas = zeros(nfn)
        total_area = 0.0
        for e in 1:nfn
            enext = e == nfn ? 1 : e + 1
            p1 = [pts[node_indices[e]][d] for d in 1:3]
            p2 = [pts[node_indices[enext]][d] for d in 1:3]
            # Triangle: fc, p1, p2
            v1 = p1 .- fc
            v2 = p2 .- fc
            tri_area = norm(cross(v1, v2)) / 2.0
            # Split between the two edge nodes
            node_areas[e] += tri_area / 2.0
            node_areas[enext] += tri_area / 2.0
            total_area += tri_area
        end

        # Each node gets its share of the face area times the unit normal
        for (li, ni) in enumerate(node_indices)
            weight = node_areas[li]
            for d in 1:D_dim
                rhs[D_dim * (ni - 1) + d] -= dp_jump * weight * unit_normal[d]
            end
        end
    end
end


