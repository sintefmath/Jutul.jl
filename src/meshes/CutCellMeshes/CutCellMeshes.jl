module CutCellMeshes
using Jutul
using StaticArrays
using LinearAlgebra

include("types.jl")
include("frames.jl")
include("planar.jl")
include("geometry.jl")
include("segments.jl")
include("moments.jl")
include("sections.jl")
include("spatial.jl")
include("extrusion.jl")
include("cutting.jl")
include("merge_faces.jl")
include("gluing.jl")
include("embedding.jl")

export cut_mesh, glue_mesh, cut_and_displace_mesh
export embed_mesh
export PlanarFrame, plane_basis, polygon_area_centroid, polygon_parts_area_centroid
export convex_parts, clip_polygon, cell_section, cell_average_distance
export CellSpatialIndex, query_cells, polygon_prism
end
