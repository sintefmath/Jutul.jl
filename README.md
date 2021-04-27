# Terv.jl

Experimental module for solution of flow and transport in porous media.

```
# Hit ] to enter package mode
# prompt changes to: (...) pkg>
activate .
instantiate # First time dependency setup
# Hit backspace to leave package mode
# prompt changes to julia>
using Revise
using Terv
# Uses the dumped pico grid (3 x 3 x 1) from MRST:
include("dev/dev_pressure_eq.jl") 
```

You can dump more test grids by using MRST together with the `writeMRSTData.m` function that dumps a grid and rock to a .mat file. Once it is stored as `data/testgrids/mycase.mat` where the file contains the fields `G` and `rock` you can run `perform_test("mycase")` to test a simple pressure solve with AMG.
