# Jutul.jl
[![DOI](https://zenodo.org/badge/358506421.svg)](https://zenodo.org/badge/latestdoi/358506421)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sintefmath.github.io/Jutul.jl/dev/)
[![Build Status](https://github.com/sintefmath/Jutul.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/sintefmath/Jutul.jl/actions/workflows/CI.yml?query=branch%3Amain)


![Jutul Darcy logo](docs/src/assets/jutul_small.png)

Experimental Julia framework for fully differentiable multiphysics simulators based on implicit finite-volume methods with automatic differentiation.

# Applications
Jutul is used for several applications. The primary package serves as a common infrastructure for several simulation projects. A examples of simple PDE solvers are included in this repo for testing and inspiration. Implementations are found [here](https://github.com/sintefmath/Jutul.jl/tree/main/src/applications/test_systems) and tests for these systems that demonstrate usage are found [here](https://github.com/sintefmath/Jutul.jl/tree/main/test/test_systems).
## Reservoir simulation
[![Jutul Darcy logo](docs/src/assets/darcy_wide.png)](https://github.com/sintefmath/JutulDarcy.jl)
[JutulDarcy.jl](https://github.com/sintefmath/JutulDarcy.jl) is a high performance Darcy flow simulator and the main demonstrator application for Jutul. See also [JutulDarcyRules.jl](https://github.com/slimgroup/JutulDarcyRules.jl) for use in differentiable workflows involving CO2 storage.
## Battery simulation
[![BattMo logo](docs/src/assets/battmologo_text.png)](https://github.com/BattMoTeam/BattMo.jl)
[BattMo.jl](https://github.com/BattMoTeam/BattMo.jl) is a battery simulator that implements a subset of the MATLAB-based [BattMo](https://github.com/BattMoTeam/BattMo) toolbox in Julia for improved performance.

## Carbon capture
Jutul.jl powers a simulator that implements vacuum swing adsorption and direct air capture processes for the capture of CO2. This application is currently not public.

# Contact information
You can use [GitHub discussions](https://github.com/sintefmath/Jutul.jl/discussions), [GitHub issues](https://github.com/sintefmath/Jutul.jl/issues) or send an e-mail to [Olav Møyner](mailto:olav.moyner@sintef.no).
