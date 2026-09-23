# NOTE: This extension was coded using AI agents (GitHub Copilot).
export export_mesh_vtu

"""
    export_mesh_vtu(mesh, filename; point_data = NamedTuple(), cell_data = NamedTuple())

Export a Jutul `UnstructuredMesh` to VTK unstructured format (`.vtu`).
Requires the WriteVTK.jl package to be loaded.

# Examples
```julia
using WriteVTK
export_mesh_vtu(mesh, "output.vtu")
export_mesh_vtu(mesh, "output.vtu"; cell_data = (pressure = p,))
```
"""
function export_mesh_vtu

end
