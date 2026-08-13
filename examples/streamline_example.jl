"""
Example: Streamline Tracing in a 2D Cartesian Mesh

This example demonstrates how to use the streamline tracing functionality
to visualize flow paths through a simple 2D mesh with a uniform flow field.
"""

using Jutul
using StaticArrays

# Create a simple 2D Cartesian mesh
nx, ny = 10, 10
dims = (nx, ny)
g = CartesianMesh(dims)
mesh = UnstructuredMesh(g)

println("Created mesh with $(number_of_cells(mesh)) cells and $(number_of_faces(mesh)) faces")

# Create a simple uniform flow field (left to right)
nf = number_of_faces(mesh)
fluxes = ones(nf)

# For a more realistic example, you could set fluxes based on face orientation:
# geo = tpfv_geometry(mesh)
# for i in 1:nf
#     normal = geo.normals[:, i]
#     # Flow in x-direction: flux proportional to x-component of normal
#     fluxes[i] = normal[1]
# end

println("\nSetup phase: Creating streamline tracer...")
tracer = setup_streamline_tracer(mesh, fluxes, max_depth = 6)

println("Created $(length(tracer.subcells)) sub-cells from tesselation")
println("Octree depth: $(tracer.octree.max_depth)")

# Define starting points along the left edge
n_streamlines = 5
geo = tpfv_geometry(mesh)

# Get cells on the left edge and create starting points
start_points = SVector{2, Float64}[]
for j in 1:n_streamlines
    # Pick cells along left edge
    cell = (j - 1) * nx + 1
    if cell <= number_of_cells(mesh)
        pt = SVector{2, Float64}(geo.cell_centroids[:, cell])
        push!(start_points, pt)
    end
end

println("\nTracing phase: Computing $(length(start_points)) streamlines...")
streamlines = trace_streamlines(
    tracer, 
    start_points,
    max_steps = 200,
    step_size = 0.05,
    forward = true,
    backward = false
)

# Print statistics about the streamlines
println("\nStreamline statistics:")
for (i, sl) in enumerate(streamlines)
    println("  Streamline $i: $(length(sl)) points")
end

# Example: Compute total path length for each streamline
println("\nPath lengths:")
for (i, sl) in enumerate(streamlines)
    total_length = 0.0
    for j in 2:length(sl)
        total_length += norm(sl[j] - sl[j-1])
    end
    println("  Streamline $i: $(round(total_length, digits=3))")
end

println("\nStreamline tracing complete!")
println("\nNote: To visualize streamlines, you can use Makie or another plotting package.")
println("Each streamline is a Vector{SVector{2, Float64}} that can be plotted as a line.")
