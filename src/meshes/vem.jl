export solve_vem_elasticity, VEMElasticitySetup, assemble_vem_elasticity

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
struct VEMElasticitySetup{D}
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

    return VEMElasticitySetup{D}(D, nn, nc, cell_data, boundary_nodes)
end

"""
    assemble_vem_elasticity(setup, youngs_modulus, poisson_ratio, pressure_change; biot_coefficient=1.0)

Assemble the VEM linear elasticity system (stiffness matrix and RHS) for given
per-cell material properties and pressure change.

Returns `(K, rhs)` where `K` is the sparse stiffness matrix and `rhs` is the
right-hand side vector, both with boundary conditions applied.
"""
function assemble_vem_elasticity(
    setup::VEMElasticitySetup{D},
    youngs_modulus::AbstractVector,
    poisson_ratio::AbstractVector,
    pressure_change::AbstractVector;
    biot_coefficient::Union{Real, AbstractVector} = 1.0
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

        # Compute body force from pressure change and scatter to global RHS
        # Body force = -biot * grad(p) ≈ distributed via VEM face integration
        _add_pressure_bodyforce!(rhs, cd, biot[cell] * pressure_change[cell], D)
    end

    # Determine boundary DOFs
    bnd_dof_set = Set{Int}()
    for node in setup.boundary_nodes
        for d in 1:D
            push!(bnd_dof_set, D * (node - 1) + d)
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
    # Add identity for boundary DOFs
    for dof in bnd_dof_set
        push!(I_final, dof)
        push!(J_final, dof)
        push!(V_final, 1.0)
        rhs[dof] = 0.0
    end

    # Assemble sparse matrix
    K = sparse(I_final, J_final, V_final, ndof, ndof)

    return (K, rhs)
end

"""
    solve_vem_elasticity(mesh, youngs_modulus, poisson_ratio, pressure_change; biot_coefficient=1.0)

Solve the VEM linear elasticity problem on an UnstructuredMesh with zero
displacement at the boundaries.

# Arguments
- `mesh::UnstructuredMesh`: The mesh to solve on.
- `youngs_modulus::AbstractVector`: Young's modulus per cell.
- `poisson_ratio::AbstractVector`: Poisson's ratio per cell.
- `pressure_change::AbstractVector`: Pressure change per cell.
- `biot_coefficient`: Biot coefficient (scalar or per-cell vector), default 1.0.

# Returns
A named tuple `(displacement, setup, K, rhs)` where:
- `displacement`: nodal displacement vector of length `D * num_nodes`.
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
u = result.displacement
```
"""
function solve_vem_elasticity(
    mesh::UnstructuredMesh,
    youngs_modulus::AbstractVector,
    poisson_ratio::AbstractVector,
    pressure_change::AbstractVector;
    biot_coefficient::Union{Real, AbstractVector} = 1.0
)
    setup = VEMElasticitySetup(mesh)
    return solve_vem_elasticity(setup, youngs_modulus, poisson_ratio, pressure_change; biot_coefficient = biot_coefficient)
end

"""
    solve_vem_elasticity(setup, youngs_modulus, poisson_ratio, pressure_change; biot_coefficient=1.0)

Solve with a precomputed `VEMElasticitySetup` (useful for repeated solves).
"""
function solve_vem_elasticity(
    setup::VEMElasticitySetup,
    youngs_modulus::AbstractVector,
    poisson_ratio::AbstractVector,
    pressure_change::AbstractVector;
    biot_coefficient::Union{Real, AbstractVector} = 1.0
)
    K, rhs = assemble_vem_elasticity(setup, youngs_modulus, poisson_ratio, pressure_change; biot_coefficient = biot_coefficient)
    displacement = K \ rhs
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

    # Check orientation: normals should point outward
    # Compute cell center (average of all face centroids)
    cell_center = zeros(3)
    for fi in 1:nfaces
        cell_center .+= face_centroids[:, fi]
    end
    cell_center ./= nfaces

    # Check if normals point outward by checking dot product with vector from
    # cell center to face centroid
    dot_sum = 0.0
    for fi in 1:nfaces
        v = face_centroids[:, fi] .- cell_center
        dot_sum += dot(v, normals[:, fi])
    end

    # If the sum is negative, flip all normals
    if dot_sum < 0
        normals .*= -1
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

    # Check orientation
    cell_center = zeros(2)
    for fi in 1:nfaces
        cell_center .+= face_centroids[:, fi]
    end
    cell_center ./= nfaces

    dot_sum = 0.0
    for fi in 1:nfaces
        v = face_centroids[:, fi] .- cell_center
        dot_sum += dot(v, normals[:, fi])
    end
    if dot_sum < 0
        normals .*= -1
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

For 2D: q is a (num_nodes × 2) matrix stored as flat vector.
"""
function _compute_q_2d(coords, face_nodes, volume)
    num_nodes = size(coords, 2)
    q = zeros(num_nodes, 2)
    fac = 1.0 / (4.0 * volume)  # 1/(2*2*area) since we distribute to two nodes

    for fi in eachindex(face_nodes)
        fn = face_nodes[fi]
        @assert length(fn) == 2
        i = fn[1]
        inext = fn[2]
        # Scaled normal (not unit normal, but edge-length scaled)
        scaled_normal = [coords[2, inext] - coords[2, i],
                        -(coords[1, inext] - coords[1, i])]
        for d in 1:2
            q[i, d] += fac * scaled_normal[d]
            q[inext, d] += fac * scaled_normal[d]
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
        q = _compute_q_2d(coords, face_nodes, volume)
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
Distribute the body force from pressure change to the global RHS vector.
For a cell with pressure change dp and Biot coefficient α:
  body_force = -α * ∇p → distributed to nodes via VEM face integration

The contribution from a cell to node i in direction d is:
  f_d = α * dp * (volume of region associated with node i in direction d)

Using VEM: this is equivalent to -α * dp * 2 * q_i[d] * volume
(since q_i = 1/(2V) ∫ φ_i n dA, the integral ∫ φ_i dV in direction d
relates to the face integrals).

More precisely, the pressure gradient contribution is:
  rhs_id = -biot * dp * ∫_∂E φ_i n_d dA = -biot * dp * 2V * q_id
"""
function _add_pressure_bodyforce!(rhs, cd::VEMCellData, biot_dp, D_dim)
    if biot_dp == 0.0
        return
    end

    coords = cd.coordinates
    face_nodes = cd.face_nodes
    outward_normals = cd.outward_normals
    volume = cd.volume
    num_nodes = length(cd.node_indices)

    # Compute the face integral ∫_∂E φ_i n dA for each node and direction
    # This equals 2V * q_i (since q_i = 1/(2V) * ∫ φ_i n dA)
    # The body force from uniform pressure is distributed as:
    # f_id = -biot * dp * ∫_∂E φ_i n_d dA
    #
    # We compute z_id = ∫_∂E φ_i n_d dA directly (which is 2V * q_id)
    # and set rhs contribution to -biot * dp * z_id

    if D_dim == 2
        z = _compute_z_2d(coords, face_nodes)
    else
        z = _compute_z_3d(coords, face_nodes, outward_normals)
    end

    # Add to global RHS: note the NEGATIVE sign because pressure acts as compression
    for i in 1:num_nodes
        gi = cd.node_indices[i]
        for d in 1:D_dim
            rhs[D_dim * (gi - 1) + d] -= biot_dp * z[i, d]
        end
    end
end

"""
Compute z_id = ∫_∂E φ_i n_d dA for 2D (the unnormalized q values).
"""
function _compute_z_2d(coords, face_nodes)
    num_nodes = size(coords, 2)
    z = zeros(num_nodes, 2)

    for fi in eachindex(face_nodes)
        fn = face_nodes[fi]
        @assert length(fn) == 2
        i = fn[1]
        inext = fn[2]
        # Scaled normal (edge-length × outward normal direction)
        # This is the same as the outer normal weighted by edge length / 2
        scaled_normal = [coords[2, inext] - coords[2, i],
                        -(coords[1, inext] - coords[1, i])]
        for d in 1:2
            z[i, d] += 0.5 * scaled_normal[d]
            z[inext, d] += 0.5 * scaled_normal[d]
        end
    end
    return z
end

"""
Compute z_id = ∫_∂E φ_i n_d dA for 3D.
"""
function _compute_z_3d(coords, face_nodes, outward_normals)
    num_nodes = size(coords, 2)
    z = zeros(num_nodes, 3)

    for fi in eachindex(face_nodes)
        fn = face_nodes[fi]
        nfn = length(fn)
        normal = outward_normals[:, fi]

        fc = zeros(3)
        for ni in fn
            fc .+= coords[:, ni]
        end
        fc ./= nfn

        for e in 1:nfn
            enext = e == nfn ? 1 : e + 1
            p1 = coords[:, fn[e]]
            p2 = coords[:, fn[enext]]
            mid = (p1 .+ p2) ./ 2.0

            v1 = p1 .- fc
            v2 = mid .- fc
            area1 = norm(cross(v1, v2)) / 2.0

            v1 = mid .- fc
            v2 = p2 .- fc
            area2 = norm(cross(v1, v2)) / 2.0

            for d in 1:3
                z[fn[e], d] += area1 * normal[d]
                z[fn[enext], d] += area2 * normal[d]
            end
        end
    end
    return z
end


