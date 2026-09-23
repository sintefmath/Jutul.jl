"""
    check_lbfgsb_availability(; throw = true)

Check if LBFGSB.jl extension is available. If `throw=true` this wil be an error,
otherwise a Boolean indicating if the extension is available will be returned.
"""
function check_lbfgsb_availability(; throw = true)
    ok = true
    try
        ok = check_lbfgsb_availability_impl()
    catch e
        if throw
            if e isa MethodError
                error("LBFGSB is not available. To fix: using Pkg; Pkg.add(\"LBFGSB\") and then call import LBFGSB to enable LBFGSB.")
            else
                rethrow(e)
            end
        else
            ok = false
        end
    end
    return ok
end

function check_lbfgsb_availability_impl

end
