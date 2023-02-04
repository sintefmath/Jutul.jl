
# High-level docstrings
This section lists the main interfaces for interacting with codes that use Jutul as their foundation.

## Simulator interfaces
Used to run simulations once a model has been set up.

```@docs
simulate
simulate!
```

## Sensitivities, adjoints and optimization

```@docs
solve_adjoint_sensitivities
solve_adjoint_sensitivities!
solve_numerical_sensitivities
setup_parameter_optimization
```

## Linear solvers
```@docs
AMGPreconditioner
ILUZeroPreconditioner
JacobiPreconditioner
LUPreconditioner
GroupWisePreconditioner
```

