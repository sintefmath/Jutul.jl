# Sensitivities, adjoints and optimization

Jutul.jl is build from the ground up to support gradient-based optimization problems. This includes both data assimilation/parameter calibration and control problems.

An example application from `JutulDarcy.jl` demonstrates many of these functions: [A fully differentiable geothermal doublet: History matching and control optimization](https://sintefmath.github.io/JutulDarcy.jl/dev/examples/workflow/fully_differentiable_geothermal)

## Objective functions

There are two main types of objective functions supported in Jutul: Those that evaluated globally over all time-steps, and those who are evaluated locally (typically as a sum over all time-steps).

```@docs
Jutul.AbstractJutulObjective
```

For either type of objective, you must implement the correct interface. This can either be done by passing a function directly, with Jutul guessing from the number of arguments what kind of objective it is, or by explicitly making a subtype that is [a Julia callable struct](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects).

These functions take in the `step_info` `Dict`, which is worth having a look at if you plan on writing objective functions:

```@docs
Jutul.optimization_step_info
```

### Global objectives

These are objectives that are defined over the entire simulation time. This means that you define the objective function as a single function that takes in the solution for all time-steps together with forces, time-step information, initial state and input data used to set up the model (if any).

```@docs
Jutul.AbstractGlobalObjective
Jutul.WrappedGlobalObjective
```

### Local/sum objectives

```@docs
Jutul.AbstractSumObjective
Jutul.WrappedSumObjective
```

## Generic optimization interface

The generic optimization interface is very general, handling gradients with respect to any parameter used in a function that sets up a complete simulation case from a `AbstractDict`. This makes use of [AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl) and the default configuration assumes that your setup function can be differentiated with the `ForwardDiff` backend. In practice, this means that you must take care when initializing arrays and other types so that they can fit the AD type (e.g. avoid use of `zeros` without a type). In addition, there may be a large number of calls to the setup function, which can sometimes be slow. Alternatively, the numerical parameter optimization interface can be used, which only differentiates with respect to numerical parameters inside the model. A hybrid approach is also supported for the generic optimization interface by setting the `deps` and `deps_ad` arguments to `optimize`, which can be much faster, but assumes that the optimization variables only affect the numerical parameters/variables of the model (values stored in the Dicts from `setup_parameters` and `setup_state0`) and not any values that exist e.g. inside the model itself.

### Defining the parameter object

```@docs
DictParameters
```

### Defining constraints and free parameters

```@docs
free_optimization_parameter!
freeze_optimization_parameter!
set_optimization_parameter!
add_optimization_multiplier!
```

### Optimizing and computing gradients

```@docs
optimize
parameters_gradient
```

## Numerical parameter optimization interface

```@docs
solve_adjoint_sensitivities
solve_adjoint_sensitivities!
Jutul.solve_numerical_sensitivities
setup_parameter_optimization
```
