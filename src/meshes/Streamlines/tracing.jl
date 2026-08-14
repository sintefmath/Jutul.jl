"""
Main streamline tracing functionality
"""

"""
    setup_streamline_tracer(mesh::UnstructuredMesh; geometry = tpfv_geometry(mesh), max_depth = 8)

Phase 1 setup for streamline tracing: Build mesh tesselation and octree.
This phase is independent of velocities and only needs to be done once per mesh.

# Arguments
- `mesh`: The unstructured mesh
- `geometry`: Optional pre-computed geometry (defaults to tpfv_geometry(mesh))
- `max_depth`: Maximum octree depth for spatial indexing

# Returns
A `StreamlineTracer` object with geometry but no velocities yet.
Use `update_velocities!` to add flux information.
"""
function setup_streamline_tracer(mesh::UnstructuredMesh{D}; 
                                geometry = Jutul.tpfv_geometry(mesh), 
                                max_depth::Int = 8) where D
    T = Jutul.float_type(mesh)
    nc = Jutul.number_of_cells(mesh)
    
    # Phase 1: Create sub-cells from centroid tesselation (without velocities)
    subcells = SubCell{D, T}[]
    
    for cell in 1:nc
        cell_subcells = tesselate_cell(mesh, cell, geometry)
        append!(subcells, cell_subcells)
    end
    
    # Build octree for fast spatial queries
    bbox_min, bbox_max = compute_global_bbox(mesh)
    octree = build_octree(subcells, bbox_min, bbox_max, max_depth)
    
    return StreamlineTracer(mesh, subcells, octree, geometry)
end

"""
    setup_streamline_tracer(mesh::UnstructuredMesh, fluxes::AbstractVector; geometry = tpfv_geometry(mesh), max_depth = 8, boundary_fluxes = nothing)

Complete setup for streamline tracing (both phases in one call).
This is a convenience function that calls both setup_streamline_tracer and update_velocities!.

# Arguments
- `mesh`: The unstructured mesh
- `fluxes`: Vector of face fluxes (one value per interior face)
- `geometry`: Optional pre-computed geometry (defaults to tpfv_geometry(mesh))
- `max_depth`: Maximum octree depth for spatial indexing
- `boundary_fluxes`: Optional vector of boundary face fluxes (defaults to zero flux)

# Returns
A `StreamlineTracer` object ready for tracing.
"""
function setup_streamline_tracer(mesh::UnstructuredMesh{D}, fluxes::AbstractVector;
                                geometry = Jutul.tpfv_geometry(mesh),
                                max_depth::Int = 8,
                                boundary_fluxes = nothing) where D
    # Phase 1: Setup mesh and octree
    tracer = setup_streamline_tracer(mesh; geometry=geometry, max_depth=max_depth)
    
    # Phase 2: Update velocities
    update_velocities!(tracer, fluxes; boundary_fluxes=boundary_fluxes)
    
    return tracer
end

"""
    update_velocities!(tracer::StreamlineTracer, fluxes::AbstractVector; boundary_fluxes = nothing)

Phase 2 setup: Update velocities in an existing tracer based on new face fluxes.
This allows you to update velocities without rebuilding the mesh tesselation and octree.

# Arguments
- `tracer`: Existing StreamlineTracer from phase 1 setup
- `fluxes`: Vector of face fluxes (one value per interior face)
- `boundary_fluxes`: Optional vector of boundary face fluxes (defaults to zero flux)

# Returns
The updated tracer (modified in-place).
"""
function update_velocities!(tracer::StreamlineTracer{D, T}, fluxes::AbstractVector; boundary_fluxes = nothing) where {D, T}
    update_subcell_velocities!(tracer.subcells, tracer.mesh, fluxes, tracer.geometry; boundary_fluxes=boundary_fluxes)
    return tracer
end

"""
    trace_streamlines(tracer::StreamlineTracer, start_points; 
                     max_steps = 1000, step_size = 0.1, 
                     forward = true, backward = false,
                     integrator = EulerIntegrator())

Tracing phase: Compute streamlines from given starting points.

# Arguments
- `tracer`: Pre-computed StreamlineTracer from setup phase
- `start_points`: Vector of starting points (as SVector or Matrix)
- `max_steps`: Maximum number of integration steps per streamline
- `step_size`: Integration step size
- `forward`: Trace in forward direction (along velocity)
- `backward`: Trace in backward direction (against velocity)
- `integrator`: Integration method (EulerIntegrator(), RK2Integrator(), or RK4Integrator())

# Returns
A vector of streamlines, where each streamline is a vector of points.
"""
function trace_streamlines(tracer::StreamlineTracer{D, T}, 
                          start_points::AbstractVector{SVector{D, T}};
                          max_steps::Int = 1000,
                          step_size::T = T(0.1),
                          forward::Bool = true,
                          backward::Bool = false,
                          integrator::StreamlineIntegrator = EulerIntegrator()) where {D, T}
    streamlines = Vector{Vector{SVector{D, T}}}()
    
    @showprogress for start_point in start_points
        streamline = SVector{D, T}[]
        
        if forward
            line = trace_single_streamline(tracer, start_point, step_size, max_steps, T(1), integrator)
            append!(streamline, line)
        end
        
        if backward
            line = trace_single_streamline(tracer, start_point, step_size, max_steps, T(-1), integrator)
            backward_line = reverse(line)
            if forward
                streamline = vcat(backward_line, streamline[2:end])
            else
                streamline = backward_line
            end
        end
        
        push!(streamlines, streamline)
    end
    
    return streamlines
end

# Overload for Matrix input
function trace_streamlines(tracer::StreamlineTracer{D, T}, 
                          start_points::AbstractMatrix;
                          kwargs...) where {D, T}
    # Convert matrix to vector of SVectors
    @assert size(start_points, 1) == D "Start points matrix must have $D rows"
    points = [SVector{D, T}(start_points[:, i]) for i in 1:size(start_points, 2)]
    return trace_streamlines(tracer, points; kwargs...)
end

"""
    trace_single_streamline(tracer, start_point, step_size, max_steps, direction, integrator)

Trace a single streamline using the specified integration method.
"""
function trace_single_streamline(tracer::StreamlineTracer{D, T}, 
                                start_point::SVector{D, T},
                                step_size::T,
                                max_steps::Int,
                                direction::T,
                                integrator::StreamlineIntegrator) where {D, T}
    points = SVector{D, T}[start_point]
    current_point = start_point
    
    for step in 1:max_steps
        # Use integrator to compute next point
        next_point = integrate_step(integrator, tracer, current_point, step_size, direction)
        
        if isnothing(next_point)
            # Left the domain or hit stagnation point
            break
        end
        
        push!(points, next_point)
        
        # Check for very small movement (potential cycle or stagnation)
        if step > 1 && norm(next_point - current_point, 2) < 1e-10
            break
        end
        
        current_point = next_point
    end
    
    return points
end
