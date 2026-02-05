"""
    EmbeddedMeshes

A module for handling embedded mesh structures in Jutul, designed for e.g.,
discrete fracture matrix (DFM) simulations.

The EmbeddedMeshes module provides functionality to work with meshes where
lower-dimensional features (like fractures or faults) are embedded within
higher-dimensional domains. These embedded features are represented as
interconnected faces that can intersect with each other, creating complex
network topologies.
"""
module EmbeddedMeshes

    using Jutul, LinearAlgebra, StaticArrays
    include("types.jl")
    include("embedded.jl")
    include("geometry.jl")
    include("finite-volume.jl")
    include("plotting.jl")

end
