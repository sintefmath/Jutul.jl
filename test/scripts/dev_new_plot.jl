using Terv
casename = "pico"
# casename = "1cell3d"
casename = "norne"
# casename = "medium_2d"
# casename = "1cell2d"
G, mrst_data = get_minimal_tpfa_grid_from_mrst(casename, extraout = true)
#
m = MRSTWrapMesh(mrst_data["G"])
G = m.data

pts, tri, mapper = triangulate_outer_surface(m)
using GLMakie
n = size(pts, 1)

data = G.cells.centroids[:, 1]
f = mesh(pts, tri, color = mapper.Cells(data))
display(f)
##
