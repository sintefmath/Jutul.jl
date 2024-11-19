module LBFGS
    export unit_box_bfgs
    using Printf, SparseArrays, Polynomials, LinearAlgebra
    include("types.jl")
    include("limited_memory_hessian.jl")
    include("constrained_optimizer.jl")
end
