export CPRPreconditioner
"""
Constrained pressure residual
"""
mutable struct CPRPreconditioner <: TervPreconditioner
    A_p  # pressure system
    r_p  # pressure residual
    p    # last pressure approximation
    buf  # buffer of size equal to full system rhs
    A_ps # full system
    w_p  # pressure weights
    pressure_precond
    system_precond
    strategy
    block_size
    function CPRPreconditioner(p = LUPreconditioner(), s = ILUZeroPreconditioner(); strategy = :quasi_impes)
        new(nothing, nothing, nothing, nothing, nothing, nothing, p, s, strategy, nothing)
    end
end


function update!(cpr::CPRPreconditioner, lsys, model, storage)
    update_cpr_internals!(cpr, lsys, model, storage)
    update!(cpr.system_precond, lsys, model, storage)
    update!(cpr.pressure_precond, cpr.A_p, cpr.r_p)
end

function initialize_storage!(cpr, J, s)
    if isnothing(cpr.A_p)
        m = J.m
        n = J.n
        cpr.block_size = bz = size(eltype(J), 1)
        @assert n == m == length(s.state.Pressure)
        nzval = zeros(length(J.nzval))

        cpr.A_p = SparseMatrixCSC(n, n, J.colptr, J.rowval, nzval)
        cpr.r_p = zeros(n)
        cpr.buf = zeros(n*bz)
        cpr.p = zeros(n)
        cpr.w_p = zeros(bz, n)
    end
end

function update_cpr_internals!(cpr::CPRPreconditioner, lsys, model, storage)
    s = storage.Reservoir
    A = reservoir_jacobian(lsys)
    cpr.A_ps = linear_operator(lsys)
    initialize_storage!(cpr, A, s)
    w_p = update_weights!(cpr, storage, A)
    update_pressure_system!(cpr.A_p, A, w_p, cpr.block_size)
end

function update_pressure_system!(A_p, A, w_p, bz)
    cp = A_p.colptr
    nz = A_p.nzval
    nz_s = A.nzval
    rv = A_p.rowval
    n = A.n
    # Update the pressure system with the same pattern in-place
    @threads for i in 1:n
        @inbounds for j in cp[i]:cp[i+1]-1
            row = rv[j]
            Ji = nz_s[j]
            tmp = 0
            @inbounds for b = 1:bz
                tmp += Ji[b, 1]*w_p[b, row]
            end
            nz[j] = tmp
        end
    end
end

function operator_nrows(cpr::CPRPreconditioner)
    return length(cpr.r_p)*cpr.block_size
end

function apply!(x, cpr::CPRPreconditioner, y, arg...)
    r_p, w_p, bz, Δp = cpr.r_p, cpr.w_p, cpr.block_size, cpr.p
    # Construct right hand side by the weights
    update_p_rhs!(r_p, y, bz, w_p)
    # Apply preconditioner to pressure part
    apply!(Δp, cpr.pressure_precond, r_p)
    correct_residual_for_dp!(y, x, Δp, bz, cpr.buf, cpr.A_ps)
    apply!(x, cpr.system_precond, y)
    for i in eachindex(Δp)
        x[(i-1)*bz + 1] += Δp[i]
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
        cpr.w_p = ones(bz, n)
    end
    w = cpr.w_p
    r = zeros(bz)
    r[1] = 1
    
    if cpr.strategy == :true_impes
        eq = storage.Reservoir.equations[:mass_conservation]
        acc = eq.accumulation.entries
        true_impes!(w, acc, r, n, bz)
    elseif cpr.strategy == :quasi_impes
        quasi_impes!(w, J, r, n, bz)
    elseif cpr.strategy == :none
        # Do nothing. Already set to one.
    else
        error("Unsupported strategy $(cpr.strategy)")
    end
    return w
end

function true_impes!(w, acc, r, n, bz)
    r_p = SVector{bz}(r)
    A = MMatrix{bz, bz, eltype(r)}(zeros(bz, bz))
    for cell in 1:n
        @inbounds for i = 1:bz
            v = acc[i, cell]
            @inbounds for j = 1:bz
                A[j, i] = v.partials[j]
            end
        end
        invert_w!(w, A, r_p, cell, bz)
    end
end

function quasi_impes!(w, J, r, n, bz)
    r_p = SVector{bz}(r)
    @threads for cell = 1:n
        J_b = J[cell, cell]'
        invert_w!(w, J_b, r_p, cell, bz)
    end
end

@inline function invert_w!(w, J, r, cell, bz)
    tmp = J\r
    @inbounds for i = 1:bz
        w[i, cell] = tmp[i]
    end
end

function update_p_rhs!(r_p, y, bz, w_p)
    @threads for i in eachindex(r_p)
        v = 0
        @inbounds for b = 1:bz
            v += y[(i-1)*bz + b]*w_p[b, i]
        end
        @inbounds r_p[i] = v
    end
end

function correct_residual_for_dp!(y, x, Δp, bz, buf, A)
    # x = x' + p
    # A (x' + p) = y
    # A x' = y'
    # y' = y - A*x
    # x = A \ y + p
    for i in eachindex(Δp)
        x[(i-1)*bz + 1] = Δp[i]
        for j = 2:bz
            x[(i-1)*bz + j] = 0
        end
    end
    mul!(buf, A, x)
    @. y -= buf
end
