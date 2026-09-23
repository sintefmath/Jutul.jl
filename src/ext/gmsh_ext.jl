export mesh_from_gmsh

"""
    G = mesh_from_gmsh(pth)
    G = mesh_from_gmsh()
    G = mesh_from_gmsh(pth; verbose = true)

Parse a Gmsh file and return a Jutul `UnstructuredMesh` (in 3D only). Requires
the Gmsh.jl package to be loaded. If no path is provided in `pth` it is assumed
that you are managing the Gmsh state manually and it will use the current
selected mesh inside Gmsh. Please note that Gmsh is GPL licensed unless you have
obtained another type of license from the authors.
"""
function mesh_from_gmsh

end
