module StaticCSR
    using SparseArrays, Polyester, LinearAlgebra

    export StaticSparsityMatrixCSR, colvals, static_sparsity_sparse
    export nthreads, minbatch

    include("mat.jl")
    export AbstractILUFactorization, ilu0_csr, ilu0_csr!, coarse_product!
    include("ilu0.jl")
    include("par_ilu0.jl")
end
