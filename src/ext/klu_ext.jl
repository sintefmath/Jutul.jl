export KLUSolver

"""
    KLUSolver(; reuse_memory = true, check = true, max_size = 50000)

Direct solver using KLU for a `SparseMatrixCSC` Jacobian. Suitable for small
systems with a low fill ratio `nnz(L + U) / nnz(J)`. Requires `using KLU`.
"""
mutable struct KLUSolver
    F
    reuse_memory::Bool
    check::Bool
    max_size
    function KLUSolver(; reuse_memory = true, check = true, max_size = 50000)
        check_klu_availability()
        return new(nothing, reuse_memory, check, max_size)
    end
end

"""
    check_klu_availability(; throw = true)

Check if the KLU.jl extension is available. If `throw=true` this will be an error,
otherwise a Boolean indicating if the extension is available will be returned.
"""
function check_klu_availability(; throw = true)
    ok = true
    try
        ok = check_klu_availability_impl()
    catch e
        if throw
            if e isa MethodError
                error("KLU is not available. To fix: using Pkg; Pkg.add(\"KLU\") and then call using KLU to enable KLUSolver.")
            else
                rethrow(e)
            end
        else
            ok = false
        end
    end
    return ok
end

function check_klu_availability_impl

end
