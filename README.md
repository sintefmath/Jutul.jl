# Terv.jl

Experimental module for solution of flow and transport in porous media.

```
]activate .
]instantiate . # First time dependency setup
using Revise
using Terv
include("dev\\dev_pressure_eq.jl") # Uses the dumped pico grid (3 x 3 x 1) from MRST
```

You can dump more test grids by using MRST together with the `writeMRSTData.m` function that dumps a grid and rock to a .mat file. Once it is stored as `data/testgrids/mycase.mat` where the file contains the fields `G` and `rock` you can run `perform_test("mycase")` to test a simple pressure solve with AMG.