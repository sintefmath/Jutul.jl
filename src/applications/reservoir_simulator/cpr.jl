export CPRPreconditioner
"""
Constrained pressure residual
"""
mutable struct CPRPreconditioner <: TervPreconditioner
    A_p
    r_p
    w_p
    pressure_precond
    system_precond
    strategy
    block_size
    function CPRPreconditioner(p = LUPreconditioner(), s = ILUZeroPreconditioner(); strategy = :quasi_impes)
        new(nothing, nothing, nothing, p, s, strategy, nothing)
    end
end


function update!(cpr::CPRPreconditioner, lsys, model, storage)
    update_pressure_system!(cpr, lsys, model, storage)
    update!(cpr.system_precond, lsys, model, storage)
end

function update_pressure_system!(cpr::CPRPreconditioner, lsys, model, storage)
    s = storage.Reservoir
    J = reservoir_jacobian(lsys)
    # r = reservoir_residual(lsys)
    if isnothing(cpr.A_p)
        n = J.n
        m = J.m
        cpr.block_size = size(eltype(J), 1)
        @assert n == m == length(s.state.Pressure)
        nzval = zeros(length(J.nzval))

        cpr.A_p = SparseMatrixCSC(n, n, J.colptr, J.rowval, nzval)
        cpr.r_p = zeros(n)
    end
    bz = cpr.block_size
    update_weights!(cpr, storage, J)
    A_p = cpr.A_p

    cp = A_p.colptr
    nz = A_p.nzval
    nz_s = J.nzval
    rv = A_p.rowval
    w_p = cpr.w_p
    for i in 1:n
        for j in cp[i]:cp[i+1]-1
            row = rv[j]
            Ji = nz_s[j]
            tmp = 0
            for b = 1:bz
                tmp += Ji[b, 1]*w_p[b, row]
            end
            nz[j] = tmp
        end
    end
end

function reservoir_residual(lsys)
    return lsys[1, 1].r
end

function reservoir_jacobian(lsys)
    return lsys[1, 1].jac
end

function update_weights!(cpr, storage, J)
    n = size(cpr.A_p, 1)
    bz = cpr.block_size
    if isnothing(cpr.w_p)
        cpr.w_p = zeros(bz, n)
    end
    w = cpr.w_p
    r = zeros(bz)
    r[1] = 1
    r_p = SVector{bz}(r)
    if cpr.strategy == :true_impes
        error("Not implemented")
    elseif cpr.strategy == :quasi_impes
        @threads for block = 1:n
            J_b = J[block, block]'
            tmp = J_b\r_p
            for i = 1:bz
                w[i, block] = tmp[i]
            end
        end
    else
        error("Unsupported strategy $(cpr.strategy)")
    end
end
