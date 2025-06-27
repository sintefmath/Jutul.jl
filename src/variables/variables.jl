# Primary variables
export degrees_of_freedom_per_entity
export absolute_increment_limit, relative_increment_limit, maximum_value, minimum_value, update_primary_variable!, default_value, initialize_variable_value!, number_of_entities
include("utils.jl")
# Turning variables into vectors for optimzation
include("vectorization.jl")
# Turning variables into scalars for serialization
include("scalarization.jl")
