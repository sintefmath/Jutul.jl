"""
Main streamline tracing functionality
"""

"""
    setup_streamline_tracer(mesh::UnstructuredMesh, fluxes::AbstractVector; geometry = tpfv_geometry(mesh), max_depth = 8)

Setup phase for streamline tracing. This function:
1. Subdivides each cell into centroid-based tesselations
2. Reconstructs velocities from face fluxes for each sub-cell
3. Builds an octree spatial index for fast point location

# Arguments
- `mesh`: The unstructured mesh
- `fluxes`: Vector of face fluxes (one value per interior face)
- `geometry`: Optional pre-computed geometry (defaults to tpfv_geometry(mesh))
- `max_depth`: Maximum octree depth for spatial indexing

# Returns
A `StreamlineTracer` object that can be used with `trace_streamlines`.
"""
function setup_streamline_tracer(mesh::UnstructuredMesh{D}, fluxes::AbstractVector; 
                                 geometry = Jutul.tpfv_geometry(mesh), 
                                 max_depth::Int = 8) where D
    T = Jutul.float_type(mesh)
    nc = Jutul.number_of_cells(mesh)
    
    # Create sub-cells from centroid tesselation and reconstruct velocities
    subcells = SubCell{D, T}[]
    
    for cell in 1:nc
        cell_subcells = tesselate_cell_and_reconstruct_velocity(mesh, cell, fluxes, geometry)
        append!(subcells, cell_subcells)
    end
    
    # Build octree for fast spatial queries
    bbox_min, bbox_max = compute_global_bbox(mesh)
    octree = build_octree(subcells, bbox_min, bbox_max, max_depth)
    
    return StreamlineTracer(mesh, subcells, octree, geometry)
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
    
    for start_point in start_points
        streamline = SVector{D, T}[]
        
        if forward
            line = trace_single_streamline(tracer, start_point, step_size, max_steps, T(1), integrator)
            append!(streamline, line)
        end
        
        if backward
            line = trace_single_streamline(tracer, start_point, step_size, max_steps, T(-1), integrator)
            # Reverse and append (excluding start point to avoid duplication)
            append!(streamline, reverse(line[2:end]))
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

# Helper to convert various start point formats
function convert_start_points(start_points::AbstractMatrix{T}, ::Val{D}) where {D, T}
    @assert size(start_points, 1) == D
    return [SVector{D, T}(start_points[:, i]) for i in 1:size(start_points, 2)]
end

function convert_start_points(start_points::Vector{SVector{D, T}}, ::Val{D}) where {D, T}
    return start_points
end
