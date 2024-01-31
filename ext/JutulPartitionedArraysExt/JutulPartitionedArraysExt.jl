module JutulPartitionedArraysExt
    using Jutul, TimerOutputs
    import Jutul: @tic

    timeit_debug_enabled() = false

    # Specific dependencies
    using PartitionedArrays, MPI
    # Already in Jutul
    using SparseArrays, Krylov, LinearAlgebra, LinearOperators, JLD2, SymRCM

    import Jutul: PArraySimulator, MPISimulator, PArrayExecutor
    import Jutul: DebugPArrayBackend, JuliaPArrayBackend, MPI_PArrayBackend

    include("interface.jl")
    include("linalg.jl")
    include("krylov.jl")
    include("overloads.jl")
    include("utils.jl")
    include("io.jl")
end
