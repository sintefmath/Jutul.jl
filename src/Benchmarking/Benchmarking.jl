module Benchmarking
    using Jutul, TimerOutputs, LinearAlgebra
    include("variables.jl")
    include("linalg.jl")
    include("assembly.jl")
end
