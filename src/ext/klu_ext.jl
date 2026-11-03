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
                error("KLU is not available. To fix: using Pkg; Pkg.add(\"KLU\") and then call using KLU to enable the KLU backend of LUSolver.")
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

function factorize_jacobian(::Val{:klu}, J)
    check_klu_availability()
    error("The KLU backend requires a SparseMatrixCSC Jacobian, got $(typeof(J)).")
end

"""
    lu_fill_ratio(F, J)

Number of nonzeros in the factors of `F` relative to the number of nonzeros in
the factorized matrix `J`. Defined by the KLU extension.
"""
function lu_fill_ratio

end
