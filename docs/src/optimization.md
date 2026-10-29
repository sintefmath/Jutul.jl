# Sensitivities, adjoints and optimization

Jutul.jl is build from the ground up to support gradient-based optimization problems. This includes both data assimilation/parameter calibration and control problems.

An example application from `JutulDarcy.jl` demonstrates many of these functions: [A fully differentiable geothermal doublet: History matching and control optimization](https://sintefmath.github.io/JutulDarcy.jl/dev/examples/workflow/fully_differentiable_geothermal)

## Objective functions

There are two main types of objective functions supported in Jutul: Those that evaluated globally over all time-steps in one go, and those who are evaluated locally at each step (typically as a sum over all time-steps). The former is very general, but can be costly to evaluate during adjoint solves, and the latter is efficient, but constrains the format a bit.

```@docs
Jutul.AbstractJutulObjective
```

For either type of objective, you must implement the correct interface. This can either be done by passing a function directly, with Jutul guessing from the number of arguments what kind of objective it is, or by explicitly making a subtype that is [a Julia callable struct](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects).

These functions take in the `step_info` `Dict`, which is worth having a look at if you plan on writing objective functions:

```@docs
Jutul.optimization_step_info
```

### Sum objectives (sum over all states, or dependence on only a few states)

The sum objective is the recommended way to pose your objective, as it is by far the fastest to evaluate. The sum objective consists of a sum of contributions over each step. From the perspective of the user, you write a function that is given the state at a specific step and returns the contribution of that step to the objective.

```@docs
Jutul.AbstractSumObjective
Jutul.WrappedSumObjective
```

### Global objectives (objective of all states simultaneously)

These are objectives that make use of all the states simultaneously. This means that you define the objective function as a single function that takes in the solution for all time-steps together with forces, time-step information, initial state and input data used to set up the model (if any). This is very general, and provides a lot of flexibility, but it can be fairly slow to evaluate. This is required for objectives where you e.g. want to normalize by values of the end state or take the cumulative difference between two responses.

```@docs
Jutul.AbstractGlobalObjective
Jutul.WrappedGlobalObjective
```

!!! note "Slow performance"
    Global objective functions are expensive to evaluate inside the adjoint solve. We **highly** recommend using the sum objective if it is possible to frame your objective as a sum over all time-steps.

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

### Utilities for debugging and coupling to other optimizers

```@docs
Jutul.optimization_problem
```

A few unexported useful utilities (API may change without breaking release):

```@docs
Jutul.DictOptimization.evaluate
Jutul.DictOptimization.finite_difference_gradient_entry
```

## Numerical parameter optimization interface

```@docs
solve_adjoint_sensitivities
solve_adjoint_sensitivities!
Jutul.solve_numerical_sensitivities
setup_parameter_optimization
```
