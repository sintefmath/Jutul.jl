export ILUZeroPreconditioner, SPAI0Preconditioner, LUPreconditioner, GroupWisePreconditioner, TrivialPreconditioner, JacobiPreconditioner, AMGPreconditioner, JutulPreconditioner, apply!

abstract type JutulPreconditioner end
abstract type DiagonalPreconditioner <: JutulPreconditioner end

include("utils.jl")
include("amg.jl")
include("diagonal.jl")
include("ilu.jl")
include("spai.jl")
include("jacobi.jl")
include("various.jl")

