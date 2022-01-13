include("physical_constants.jl")
include("battery_types.jl")
include("tensor_tools.jl")
include("physics.jl")
include("battery_utils.jl")
include("test_setup.jl")

include("models/elchem_component.jl")
include("models/elyte.jl")
include("models/current_collector.jl")
include("models/current_collector_temp.jl")
include("models/activematerial.jl")
include("models/ocd.jl")
include("models/simple_elyte.jl")

include("models/battery_cross_terms.jl") # Works now
