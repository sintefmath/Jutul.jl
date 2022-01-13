# Jutul.jl

Experimental module for solution of flow and transport in porous media.

```Julia
# Hit ] to enter package mode
# prompt changes to: (...) pkg>
activate .
instantiate # First time dependency setup
# Hit backspace to leave package mode
# prompt changes to julia>
using Revise
using Jutul
# Simple gravity segregation test
include("test/scripts/two_phase_gravity_segregation.jl")
# Uses the dumped pico grid (3 x 3 x 1) from MRST:
include("test/scripts/two_phase_with_plotting.jl")
```

You can dump more test grids by using MRST together with the `writeMRSTData.m` function that dumps a grid and rock to a .mat file. Once it is stored as `data/testgrids/mycase.mat` where the file contains the fields `G` and `rock` you can run 
```Julia
perform_test("mycase")
```
to run a simple two-phase simulation and 
```perform_test("mycase", true)```
to launch with plotting, provided that the model has a cartesian grid.

A next step might be to run all the tests to verify that everything is ok and see what functionality exists:
```Julia
include("test/runtests.jl")
```
