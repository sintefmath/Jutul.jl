
# High-level docstrings

This section lists the main interfaces for interacting with codes that use Jutul as their foundation.

## Setting up models

```@docs
SimulationModel
MultiModel
```

### Model API

#### Getters

```@docs
Jutul.get_secondary_variables
Jutul.get_primary_variables
Jutul.get_parameters
Jutul.get_variables
```

Setters

```@docs
Jutul.set_secondary_variables!
Jutul.set_primary_variables!
Jutul.set_parameters!
```

Various

```@docs
Jutul.number_of_degrees_of_freedom
Jutul.number_of_values
```

## Systems and domains

```@docs
JutulSystem
JutulContext
JutulDomain
DataDomain
DiscretizedDomain
Jutul.JutulDiscretization
```

## Set up of system

```@docs
setup_state
setup_parameters
setup_state_and_parameters
setup_forces
```

## Simulator interfaces

Used to run simulations once a model has been set up.

```@docs
simulate
simulate!
Jutul.solve_timestep!
```

Configure a simulator:

```@docs
Simulator
simulator_config
JutulConfig
add_option!
```

## Linear solvers

```@docs
GenericKrylov
LUSolver
```

### Preconditioners

```@docs
AMGPreconditioner
ILUZeroPreconditioner
JacobiPreconditioner
SPAI0Preconditioner
LUPreconditioner
GroupWisePreconditioner
```

## Execution contexts

```@docs
Jutul.DefaultContext
Jutul.ParallelCSRContext
```
