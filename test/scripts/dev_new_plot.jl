using Terv
casename = "pico"
casename = "1cell3d"
G, mrst_data = get_minimal_tpfa_grid_from_mrst(casename, extraout = true)
##
m = MRSTWrapMesh(mrst_data["G"])
G = m.data

pts, tri = triangulate_outer_surface(m)
using GLMakie
n = size(pts, 1)
mesh(pts, tri, color = rand(n))
