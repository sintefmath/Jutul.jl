# Units

Jutul assumes that all physical quantities are represented in SI units
internally (or another consistent system). To facilitate conversion between
different unit systems, Jutul provides utility functions to convert values to and
from SI units.

There are also Julia packages like
[Unitful.jl](https://github.com/JuliaPhysics/Unitful.jl) and
[DynamicQuantities.jl](https://github.com/JuliaPhysics/DynamicQuantities.jl)
that can be used to embed units in types, but these are not directly integrated
with Jutul at this time. Performing calculations with your unit system of choice
before striping units and passing values in SI to Jutul is the recommended
approach.

```@docs
convert_to_si
convert_from_si
si_unit
@si_str
```
