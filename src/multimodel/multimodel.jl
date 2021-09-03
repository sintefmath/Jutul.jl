export MultiModel, get_domain_intersection, update_cross_term!
export InjectiveCrossTerm, MultiModelCoupling
import Base: show

include("types.jl")
include("model.jl")
include("crossterm.jl")
include("interface.jl")
include("utils.jl")
