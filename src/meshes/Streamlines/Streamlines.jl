"""
    Streamlines

Module for tracing streamlines through unstructured meshes.

Provides functionality for:
- Cell tesselation into tetrahedral/triangular sub-cells
- Velocity reconstruction from face fluxes
- Octree spatial indexing for fast point location
- Multiple integration methods (Euler, RK2, RK4)
- Forward and backward streamline tracing

The setup is split into two phases:
1. Mesh/octree setup: `setup_streamline_tracer(mesh; geometry, max_depth)`
2. Velocity update: `update_velocities!(tracer, fluxes)`

Or use the convenience function: `setup_streamline_tracer(mesh, fluxes; geometry, max_depth)`
"""
module Streamlines

using StaticArrays
using LinearAlgebra
using ProgressMeter
using ..Jutul

export StreamlineTracer, setup_streamline_tracer, trace_streamlines, update_velocities!
export EulerIntegrator, RK2Integrator, RK4Integrator

# Core types
include("types.jl")

# Integration methods
include("integrators.jl")

# Octree implementation
include("octree.jl")

# Tesselation and velocity reconstruction
include("tesselation.jl")

# Main tracing functionality
include("tracing.jl")

end # module
