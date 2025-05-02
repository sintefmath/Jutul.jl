"""
    spiral_mesh_tags(rmesh, spacing = missing)

Get the tags for a spiral mesh. If spacing is provided, it will also return the
spacing and winding tags.
"""
function spiral_mesh_tags(rmesh, spacing = missing)
    I = map(i -> cell_ijk(rmesh, i)[1], 1:number_of_cells(rmesh))
    J = map(i -> cell_ijk(rmesh, i)[2], 1:number_of_cells(rmesh))
    K = map(i -> cell_ijk(rmesh, i)[3], 1:number_of_cells(rmesh))
    out = Dict(:radial => I, :angular => J, :depth => K)
    if !ismissing(spacing)
        spacing = spiral_spacing(spacing)
        spacing_width = length(spacing) - 1
        out[:spacing] = mod.(I .- 1, spacing_width) .+ 1
        out[:winding] = div.(I .- 1, spacing_width) .+ 1
    end
    return out
end
