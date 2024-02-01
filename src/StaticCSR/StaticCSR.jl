module StaticCSR
    using SparseArrays, StaticArrays, Polyester, LinearAlgebra

    export StaticSparsityMatrixCSR, colvals, static_sparsity_sparse
    export nthreads, minbatch

    include("mat.jl")
    export AbstractILUFactorization, ilu0_csr, ilu0_csr!, in_place_mat_mat_mul!
    include("ilu0.jl")
    include("par_ilu0.jl")
end
