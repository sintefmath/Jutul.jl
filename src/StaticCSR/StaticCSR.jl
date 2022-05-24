module StaticCSR
    using SparseArrays, Polyester, LinearAlgebra

    export StaticSparsityMatrixCSR, colvals, static_sparsity_sparse

    include("mat.jl")
    export AbstractILUFactorization, ilu0_csr, ilu0_csr!
    include("ilu0.jl")
end
