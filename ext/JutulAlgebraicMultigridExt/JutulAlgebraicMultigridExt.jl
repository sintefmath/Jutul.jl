module JutulAlgebraicMultigridExt
    using Jutul
    using AlgebraicMultigrid
    using SparseArrays
    using LinearAlgebra
    using Polyester

    import LinearAlgebra: ldiv!, lu, mul!
    import Jutul: AMGPreconditioner,
        update_preconditioner!,
        partial_update_preconditioner!,
        operator_nrows,
        StaticSparsityMatrixCSR,
        SPAI0Preconditioner,
        ILUZeroPreconditioner,
        JacobiPreconditioner,
        default_executor,
        get_factorization,
        nthreads,
        minbatch,
        colvals,
        @tic

    function Jutul.check_algebraicmultigrid_availability_impl()
        return true
    end

    function Jutul.amg_default_cycle_impl()
        return AlgebraicMultigrid.V()
    end

    include(joinpath(@__DIR__, "..", "..", "src", "linsolve", "precond", "amg.jl"))
end
