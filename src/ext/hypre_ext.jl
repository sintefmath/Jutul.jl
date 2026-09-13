abstract type HYPREPreconditioner <: Jutul.JutulPreconditioner end

struct BoomerAMGPreconditioner{T} <: HYPREPreconditioner
    prec::T
    data::Dict{Symbol, Any}
end

function setup_hypre_precond

end

function set_hypre_threads

end

export BoomerAMGPreconditioner

function BoomerAMGPreconditioner(;
    CoarsenType = 10,      # HMIS
    StrongThreshold = 0.5, # For 3D
    AggNumLevels = 1,      # Aggressive coarsening for first levels
    AggTruncFactor = 0.3,  # Remove weak connections
    InterpType = 6,        # ext+i
    PrintLevel = 0,
    Tol = 0.0,
    MaxIter = 1,
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
            Tol = Tol,
            MaxIter = MaxIter,
            PrintLevel = PrintLevel,
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

function check_hypre_availability(; throw = true)
    ok = true
    try
        ok = check_hypre_availability_impl()
    catch e
        if throw
            if e isa MethodError
                error("HYPRE is not available. To fix: using Pkg; Pkg.add(\"HYPRE\") and then call using HYPRE to enable HYPRE.")
            else
                rethrow(e)
            end
        else
            ok = false
        end
    end
    return ok
end

function check_hypre_availability_impl

end
