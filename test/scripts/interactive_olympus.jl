using Terv
using GLMakie
using MAT
casename = "olympus"
##
G, mrst_data = get_minimal_tpfa_grid_from_mrst(casename, extraout = true)
##
fn = string(dirname(pathof(Terv)), "/../data/testgrids/olympus_rocks.mat")
rocks = MAT.matread(fn)["rocks"]

##
function convert_sym(r)
    t = Vector{Float64}
    d = Dict{Symbol, t}()
    for key in keys(r)
        d[Symbol(key)] = t(vec(r[key]))
    end
    return d
end
converted_rocks = map(convert_sym, rocks)
##
#
m = MRSTWrapMesh(mrst_data["G"])
G = m.data
##
f, ax = plot_interactive(m, converted_rocks, colormap = :roma)
##
w_raw = mrst_data["W"]
for w in w_raw
    if w["sign"] > 0
        c = :midnightblue
    else
        c = :firebrick
    end
    plot_well!(ax, m, w, color = c)
end
##