abstract type HYPREPreconditioner <: Jutul.JutulPreconditioner end

struct BoomerAMGPreconditioner{T} <: HYPREPreconditioner
    prec::T
    data::Dict{Symbol, Any}
end

function setup_hypre_precond

end

export BoomerAMGPreconditioner

function BoomerAMGPreconditioner(;
    CoarsenType = 10,      # HMIS
    StrongThreshold = 0.5, # For 3D
    AggNumLevels = 3,      # Aggressive coarsening for first levels
    AggTruncFactor = 0.3,  # Remove weak connections
    InterpType = 6,        # ext+i
    NumPaths = 2,
    kwarg...
    )
    # Default settings inspired by
    # https://mooseframework.inl.gov/releases/moose/2021-05-18/application_development/hypre.html
    prec = missing
    try
        prec = setup_hypre_precond(:boomeramg;
            CoarsenType = CoarsenType,
            StrongThreshold = StrongThreshold,
            AggNumLevels = AggNumLevels,
            AggTruncFactor = AggTruncFactor,
            InterpType = InterpType,
            NumPaths = NumPaths,
            kwarg...
            )
    catch e
        @error "Unable to initialize HYPRE preconditioner. Is HYPRE.jl loaded and Julia at least 1.9?"
        rethrow(e)
    end
    return BoomerAMGPreconditioner(prec, Dict{Symbol, Any}())
end

function generate_hypre_assembly_helper

end

function local_hypre_copy!

end
