
include("porousmedia_grids.jl")
include("flux.jl")
# Definitions for multiphase flow
include("multiphase.jl")
include("multiphase_secondary_variables.jl")
# Wells etc.
include("facility/types.jl")

include("facility/flux.jl")
include("facility/wells.jl")
include("facility/facility.jl")
include("porousmedia.jl")
# MRST inputs and test cases that use MRST input
include("mrst_input.jl")
# Various input tricks
include("io.jl")
include("cpr.jl")
