mutable struct AMGCLPreconditioner{I}
    wrapper
    type::Symbol
    param::String
    rowptr::Vector{I}
    colval::Vector{I}
end

function AMGCLPreconditioner(type = :amg; param = Dict{Symbol, Any}(), index_type = Int, kwarg...)
    check_amgcl_availability()
    row = Vector{index_type}()
    col = Vector{index_type}()
    for (k, v) in pairs(kwarg)
        param[k] = v
    end
    s = amgcl_parse_parameters_impl(param)
    return AMGCLPreconditioner{index_type}(missing, type, s, row, col)
end

"""
    check_amgcl_availability(; throw = true)

Check if AMGCLWrap extension is available. If `throw=true` this wil be an error,
otherwise a Boolean indicating if the extension is available will be returned.
"""
function check_amgcl_availability(; throw = true)
    ok = true
    try
        ok = check_amgcl_availability_impl()
    catch e
        if throw
            if e isa MethodError
                error("AMGCLWrap is not available. To fix: using Pkg; Pkg.add(\"AMGCLWrap\") and then call import AMGCLWrap to enable AMGCLWrap.")
            else
                rethrow(e)
            end
        else
            ok = false
        end
    end
    return ok
end

function check_amgcl_availability_impl

end

function amgcl_parse_parameters_impl

end