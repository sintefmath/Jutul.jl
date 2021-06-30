struct BlockDQGMRES end

function solve!(sys::LinearizedSystem, solver::BlockDQGMRES)
    # Simple block solver for testing. Not efficiently implemented.
    n = length(sys.r_buffer)

    r = reshape(sys.r_buffer, n)
    jac = sys.jac

    Vt = eltype(sys.r)
    Mt = eltype(jac)

    as_svec = (x) -> reinterpret(Vt, x)
    # as_smat = (x) -> reinterpret(Mt, x)
    as_float = (x) -> reinterpret(Float64, x)

    function op_mult!(res, x, α, β::T) where T
        if β == zero(T)
            res .= as_float(α.*jac*as_svec(x))
        else
            res .= as_float(α.*jac*as_svec(x) + β.*res)
        end
    end

    opA = LinearOperator(Float64, n, n, false, false, op_mult!)
    (x, stats) = dqgmres(opA, r)

    sys.dx_buffer .= -reshape(x, size(sys.dx_buffer))
end
