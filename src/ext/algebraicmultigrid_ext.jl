mutable struct AMGPreconditioner{T} <: JutulPreconditioner
    method_kwarg
    cycle
    factor
    dim
    hierarchy
    smoothers
    smoother_type::Symbol
    npre::Int
    npost::Int
end

function AMGPreconditioner(method::Symbol; smoother_type = :default, cycle = nothing, npre = 1, npost = npre, kwarg...)
    check_algebraicmultigrid_availability()
    @assert method == :smoothed_aggregation || method == :ruge_stuben || method == :aggregation
    if isnothing(cycle)
        cycle = amg_default_cycle_impl()
    end
    return AMGPreconditioner{method}(kwarg, cycle, nothing, nothing, nothing, nothing, smoother_type, npre, npost)
end

"""
    check_algebraicmultigrid_availability(; throw = true)

Check if AlgebraicMultigrid extension is available. If `throw=true` this will be
an error, otherwise a Boolean indicating if the extension is available will be
returned.
"""
function check_algebraicmultigrid_availability(; throw = true)
    ok = true
    try
        ok = check_algebraicmultigrid_availability_impl()
    catch e
        if throw
            if e isa MethodError
                error("AlgebraicMultigrid is not available. To fix: using Pkg; Pkg.add(\"AlgebraicMultigrid\") and then call import AlgebraicMultigrid to enable AlgebraicMultigrid.")
            else
                rethrow(e)
            end
        else
            ok = false
        end
    end
    return ok
end

function amg_default_cycle_impl

end

function check_algebraicmultigrid_availability_impl

end
