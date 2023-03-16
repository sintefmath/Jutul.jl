import Base.getindex

function Base.getindex(ls::MultiLinearizedSystem, i, j = i)
    return ls.subsystems[i, j]
end

function Base.getindex(ls::LinearizedSystem, i, j = i)
    @assert i == j == 1
    return ls
end

number_of_subsystems(ls::LinearizedSystem) = 1
number_of_subsystems(ls::MultiLinearizedSystem) = size(ls.subsystems, 1)

do_schur(sys) = sys.reduction == :schur_apply

function prepare_linear_solve!(sys::MultiLinearizedSystem)
    if do_schur(sys)
        _, C, _, E, a, b = get_schur_blocks!(sys, true, update = true)
        b_buf, = sys.schur_buffer[2]
        # The following is the in-place version of a -= C*(E\b)
        n = length(E)
        @batch for i in 1:n
            b_buf, = sys.schur_buffer[i+1]
            ldiv!(b_buf, E[i], b[i])
        end
        for i in 1:n
            b_buf, = sys.schur_buffer[i+1]
            mul!(a, C[i], b_buf, -1.0, 1.0)
        end
    end
end


function get_schur_blocks!(sys, include_r = true; update = false, keep_ix = 1, elim_ix = 2:size(sys.subsystems, 1), factorized = true)
    keep = sys[keep_ix, keep_ix]
    elim = map(i -> sys[i, i], elim_ix)

    B = keep.jac
    C = map(i -> sys[keep_ix, i].jac, elim_ix)
    D = map(i -> sys[i, keep_ix].jac, elim_ix)
    E = map(x -> x.jac, elim)

    F = sys.factor

    if update
        E_lu = update!(F, lu, lu!, E)
    else
        E_lu = F.factor
    end
    ϵ = factorized ? E_lu : E
    if include_r
        a = vector_residual(keep)
        b = vector_residual(elim)
        return (B, C, D, ϵ, a, b)
    else
        return (B, C, D, ϵ)
    end
end

function vector_residual(sys::MultiLinearizedSystem)
    if do_schur(sys)
        r = vector_residual(sys[1, 1])
    else
        r = sys.r
    end
    return r
end

function linear_operator(sys::MultiLinearizedSystem; skip_red = false)
    if do_schur(sys) && !skip_red
        B, C, D, E = get_schur_blocks!(sys, false)
        # A = B - CE\D
        n = size(first(C), 1)
        T = eltype(sys[1, 1].r)
        apply! = get_schur_apply(sys.schur_buffer, Val(T), B, C, D, E)
        op = LinearOperator(Float64, n, n, false, false, apply!)
    else
        S = sys.subsystems
        if true
            @assert size(S) == (2, 2) "Only supported for 2x2 blocks"
            ops = map(linear_operator, S)
            op = vcat(hcat(ops[1, 1], ops[1, 2]), hcat(ops[2, 1], ops[2, 2]))
        else
            d = size(S, 2)
            ops = map(linear_operator, vec(S))
            op = hvcat(size(S), ops...)
        end
    end
    return op
end

function get_schur_apply(schur_buffers, Tv, B, C, D, E)
    return (res, x, α, β) -> schur_mul!(res, schur_buffers, Tv, B, C, D, E, x, α, β)
end

function update_dx_from_vector!(sys::MultiLinearizedSystem, dx_from_solver; dx = sys.dx)
    if do_schur(sys)
        Δx = dx_from_solver
        _, C, D, E, _, b = get_schur_blocks!(sys)
        n = length(dx_from_solver)
        m = length(dx)

        x = view(dx, 1:n)
        y = view(dx, (n+1):m)

        schur_dx_update!(x, y, C, D, E, b, sys, dx_from_solver, Δx, sys.schur_buffer)
    else
        @tullio dx[i] = -dx_from_solver[i]
    end
end

function schur_dx_update!(x, y, C, D, E, b, sys, dx, Δx, buffers)
    @tullio x[i] = -dx[i]
    # We want to do (in-place):
    # dy = B = -E\(b - D*Δx) = E\(D*Δx - b)
    offsets = cumsum(length(x) for x in b)
    n = length(D)
    @inbounds for i in 1:n
        if i == 1
            offset = 0
        else
            offset = offsets[i-1]
        end
        b_i = b[i]
        n = length(b_i)
        buf_b, = buffers[i+1]
        mul!(buf_b, D[i], Δx)
        # now buf_b = D*Δx
        @batch minbatch=1000 for j in 1:n
            @inbounds buf_b[j] -= b_i[j]
        end
        y_i = view(y, (offset+1):(offset+n))
        ldiv!(y_i, E[i], buf_b)
    end
end

@inline function schur_mul_internal!(res, res_v, schur_buffers, B, C, D, E, x, x_v, α, β::T) where T
    @tic "spmv (schur)" begin
        # This function does:
        # res ← β*res + α*(B*x - C*(E\(D*x)))
        n = length(D)
        mul!(res_v, B, x_v, α, β)
        @batch for i = 1:n
            @inbounds b_buf_1, b_buf_2 = schur_buffers[i+1]
            @inbounds D_i = D[i]
            @inbounds E_i = E[i]
            @inbounds C_i = C[i]
            mul!(b_buf_2, D_i, x)
            ldiv!(b_buf_1, E_i, b_buf_2)
            mul!(res, C_i, b_buf_1, -α, true)
        end
    end
    return res
end

@inline function schur_mul!(res, schur_buffers, ::Val{r_type}, B, C, D, E, x, α, β) where r_type
    n = size(B, 2)
    res_v = unsafe_reinterpret(r_type, res, n)
    x_v = unsafe_reinterpret(r_type, x, n)
    schur_mul_internal!(res, res_v, schur_buffers, B, C, D, E, x, x_v, α, β)
end

function jacobian(sys::MultiLinearizedSystem)
    if do_schur(sys)
        J = sys[1, 1].jac
    else
        error()
    end
    return J
end

function residual(sys::MultiLinearizedSystem)
    if do_schur(sys)
        r = sys[1, 1].r
    else
        error()
    end
    return r
end

function linear_system_context(model, sys::MultiLinearizedSystem)
    if do_schur(sys)
        ctx = first(model.models).context
    else
        error()
    end
    return ctx
end