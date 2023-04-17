abstract type HYPREPreconditioner <: Jutul.JutulPreconditioner end

struct BoomerAMGPreconditioner{T} <: HYPREPreconditioner
    prec::T
    data::Dict{Symbol, Any}
end

function setup_hypre_precond

end

export BoomerAMGPreconditioner

function BoomerAMGPreconditioner(; kwarg...)
    prec = missing
    try
        prec = setup_hypre_precond(:boomeramg; kwarg...)
    catch e
        @error "Unable to initialize HYPRE preconditioner. Is HYPRE.jl loaded and Julia at least 1.9?"
        rethrow(e)
    end
    return BoomerAMGPreconditioner(prec, Dict{Symbol, Any}())
end
