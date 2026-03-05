function fix_nonpositive_cell_volumes!(msh, bad_cells; print::Bool = true)
    if length(bad_cells) > 0
        # We always print this since the issue cannot be fixed
        jutul_print("Mesh fixing", "Mesh has cells with non-positive or non-finite volume. Automatic fixing is not implemented for this issue.", color = :yellow)
    end
    return msh
end
