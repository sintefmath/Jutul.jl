#=
Compare the simulation of electrolyte in Julia and Matlab
=#

using Terv

function load_elyte()
    name="model1d"
    fn = string(dirname(pathof(Terv)), "/../data/models/", name, ".mat")
    exported = MAT.matread(fn)
    exported = exported_all["model"]["NegativeElectrode"]["CurrentCollector"];
end