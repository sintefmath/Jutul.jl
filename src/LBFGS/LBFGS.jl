module LBFGS
    export unit_box_bfgs, optimize_bound_constrained
    using Printf, SparseArrays, Polynomials, LinearAlgebra, Jutul
    include("types.jl")
    include("limited_memory_hessian.jl")
    include("limited_memory_hessian_legacy.jl")
    include("constrained_optimizer.jl")
    include("bound_constrained_optimizer.jl")
    include("inexact_line_search.jl")
end
