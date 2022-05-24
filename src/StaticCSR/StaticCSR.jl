module StaticCSR
    using SparseArrays, Polyester, LinearAlgebra

    export StaticSparsityMatrixCSR, colvals, static_sparsity_sparse

    include("mat.jl")
end
