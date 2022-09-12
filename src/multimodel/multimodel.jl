export MultiModel
export InjectiveCrossTerm, AdditiveCrossTerm
import Base: show

include("types.jl")
include("model.jl")
include("crossterm.jl")
include("interface.jl")
include("utils.jl")
include("gradients.jl")
include("vectorization.jl")
