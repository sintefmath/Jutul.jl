module JutulAMGCLWrapExt
    using Jutul, TimerOutputs, LinearAlgebra
    import AMGCLWrap

    timeit_debug_enabled() = Jutul.timeit_debug_enabled()

    function Jutul.check_amgcl_availability_impl()
        return true
    end

    function Jutul.amgcl_parse_parameters_impl(param)
        return AMGCLWrap.tojson(param)
    end

    function get_amgcl_wrapper_constructor(p::Jutul.AMGCLPreconditioner)
        if p.type == :amg
            F = AMGCLWrap.AMGPrecon
        elseif p.type == :amg_solver
            F = AMGCLWrap.AMGSolver
        elseif p.type == :relaxation_solver
            F = AMGCLWrap.RLXSolver
        else
            @assert p.type == :relaxation
            F = RLXPrecon
        end
        return F
    end

    function replace_values_sparse_array!(x, y)
        n = length(y)
        resize!(x, n)
        @inbounds for i in 1:n
            x[i] = y[i] - 1
        end
        return x
    end

    function Jutul.update_preconditioner!(p::Jutul.AMGCLPreconditioner, A::Jutul.StaticCSR.StaticSparsityMatrixCSR, b, ctx, executor)
        n, m = size(A)
        @assert n == m
        rowptr = replace_values_sparse_array!(p.rowptr, A.At.colptr)
        colval = replace_values_sparse_array!(p.colval, A.At.rowval)
        nzval = A.At.nzval
        F = get_amgcl_wrapper_constructor(p)
        op = F(m, rowptr, colval, nzval, 1, p.param)
        e = AMGCLWrap.error_state(op)
        if e != 0
            error("""Unable to create operator\nerror_state: $e\nparam: "$s"\n""")
        end
        p.wrapper = op
        return p
    end

    function Jutul.update_preconditioner!(p::Jutul.AMGCLPreconditioner, A, b, ctx, executor)
        # TODO: This is a bit slow, CSR interface is obviously faster since we can
        # reuse memory.
        F = get_amgcl_wrapper_constructor(p)
        p.wrapper = F(A, param = p.param)
        return p
    end

    function Jutul.apply!(x, p::Jutul.AMGCLPreconditioner, y, arg...)
        ldiv!(x, p.wrapper, y)
    end
end