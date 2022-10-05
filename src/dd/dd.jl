abstract type AbstractVariableSet end

"The global set of variables"
struct GlobalSet <: AbstractVariableSet end

"Set of a variable where variables are defined"
struct VariableSet <: AbstractVariableSet end

"Set of a variable where equations are defined"
struct EquationSet <: AbstractVariableSet end

include("subdomains.jl")
include("submodels.jl")
include("substate.jl")
include("trivial_map.jl")
include("finite_volume_map.jl")
