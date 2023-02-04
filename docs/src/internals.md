
```@meta
CurrentModule = Jutul
```
# Internal docstrings
These functions are mostly relevant for implementing simulators in the Jutul framework.

## Entities and variables
### Variables
Variable types:
```@docs
JutulVariables
ScalarVariable
VectorVariables
```
Variables API:
```@docs
degrees_of_freedom_per_entity
values_per_entity
maximum_value
minimum_value
variable_scale
absolute_increment_limit
relative_increment_limit
associated_entity
```
Updating variables
```@docs
@jutul_secondary
Jutul.get_dependencies
```
### Entities
```@docs
JutulEntity
Cells
Faces
Nodes
```
Entities API
```@docs
number_of_partials_per_entity
number_of_entities
```

## Equations
```@docs
JutulEquation
```
Equations API
```@docs
Jutul.number_of_equations
Jutul.number_of_equations_per_entity
```

## Automatic differentiation
```@docs
Jutul.value
Jutul.as_value
Jutul.local_ad
Jutul.JutulAutoDiffCache
Jutul.CompactAutoDiffCache
Jutul.allocate_array_ad
Jutul.get_ad_entity_scalar
Jutul.get_entries
```

## Matrix
```@docs
Jutul.JutulMatrixLayout
Jutul.EntityMajorLayout
Jutul.EquationMajorLayout
Jutul.BlockMajorLayout
```

## Various
```@docs
convergence_criterion
Jutul.partition
Jutul.load_balanced_endpoint
```

