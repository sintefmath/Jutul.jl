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

function equation_major_to_block_major_view(a, block_size)
    @views x = reshape(reshape(vec(a), :, block_size)', :)
    return x
end

function block_major_to_equation_major_view(a, block_size)
    @views x = reshape(reshape(vec(a), block_size, :)', :)
    return x
end

@inline major_to_minor(n::I, m::I, i::I) where I = (n*((i - 1) % m) + 1 + ((i - 1) ÷ m))::I
@inline from_block_index(bz, nc, i) = major_to_minor(bz, nc, i)
@inline from_entity_index(bz, nc, i) = major_to_minor(nc, bz, i)


@inline from_block_urange(bz, n, b) = (b-1)*n+1:b*n
@inline from_entity_urange(bz, n, b) = b:bz:((n-1)*bz+b)


function prepare_solve!(sys::MultiLinearizedSystem)
    if do_schur(sys)
        B, C, D, E, a, b = get_schur_blocks!(sys, true, update = true)
        e = eltype(B)
        is_float = e <: Real

        a_buf = sys.schur_buffer[1]
        b_buf = sys.schur_buffer[2]
        # The following is the in-place version of Δa = C*(E\b)
        ldiv!(b_buf, E, b)
        mul!(a_buf, C, b_buf)
        Δa = a_buf
        if !is_float
            bz = size(e, 1)
            Δa = equation_major_to_block_major_view(Δa, bz)
        end
        @. a -= Δa
    end
end


function get_schur_blocks!(sys, include_r = true; update = false, keep_ix = 1, elim_ix = 2, factorized = true)
    keep = sys[keep_ix, keep_ix]
    elim = sys[elim_ix, elim_ix]

    B = keep.jac
    C = sys[keep_ix, elim_ix].jac
    D = sys[elim_ix, keep_ix].jac
    E = elim.jac

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
        n = size(C, 1)
        T = eltype(sys[1, 1].r)
        a_buf = sys.schur_buffer[1]
        b_buf = sys.schur_buffer[2]
        apply! = (res, x, α, β) -> schur_mul!(res, a_buf, b_buf, T, B, C, D, E, x, α, β)
        op = LinearOperator(Float64, n, n, false, false, apply!)
    else
        S = sys.subsystems
        d = size(S, 2)
        ops = map(linear_operator, permutedims(S))
        op = hvcat(d, ops...)
    end
    return op
end

function update_dx_from_vector!(sys::MultiLinearizedSystem, dx)
    if do_schur(sys)
        bz = block_size(sys[1, 1])
        if bz == 1
            Δx = dx
        else
            Δx = block_major_to_equation_major_view(dx, bz)
        end
        buf_a = sys.schur_buffer[1]
        buf_b = sys.schur_buffer[2]
        _, C, D, E, a, b = get_schur_blocks!(sys)
        n = length(dx)
        m = length(sys.dx)

        A = view(sys.dx, 1:n)
        B = view(sys.dx, (n+1):m)

        schur_dx_update!(A, B, C, D, E, a, b, sys, dx, Δx, buf_a, buf_b)
    else
        @tullio sys.dx[i] = -dx[i]
    end
end

function schur_dx_update!(A, B, C, D, E, a, b, sys, dx, Δx, buf_a, buf_b)

    @tullio A[i] = -dx[i]
    # We want to do (in-place):
    # dy = B = -E\(b - D*Δx) = E\(D*Δx - b)
    @inbounds for i in eachindex(Δx)
        buf_a[i] = Δx[i]
    end
    mul!(buf_b, D, buf_a)
    # now buf_b = D*Δx
    @inbounds for i in eachindex(b)
        buf_b[i] -= b[i]
    end
    ldiv!(B, E, buf_b)
end

function schur_mul3!(res, r_type::Float64, B, C, D, E, x, α, β::T) where T
    if β == zero(T)
        tmp = B*x - C*(E\(D*x))
        res .= tmp
        if α != one(T)
            lmul!(α, res)
        end
    else
        error("Not implemented yet.")
    end
end

function schur_mul_float!(res, B, C, D, E, x, α, β::T) where T
    @assert β == zero(T) "Only β == 0 implemented."
    # compute B*x
    mul!(res, B, x)
    drs = C*(E\(D*x))
    @tullio res[i] = res[i] - drs[i]
    if α != one(T)
        lmul!(α, res)
    end
end

function schur_mul_block!(res, res_v, a_buf, b_buf, block_size, B, C, D, E, x, x_v, α, β::T) where T
    @assert β == zero(T) "Only β == 0 implemented."
    N = length(res)
    n = N ÷ block_size
    # compute B*x
    mul!(res_v, B, x_v)
    for i = 1:n
        for b = 1:block_size
            ix = (i-1)*block_size + b
            jx = (b-1)*n + i
            @inbounds a_buf[jx] = x[ix]
        end
    end
    mul!(b_buf, D, a_buf)
    ldiv!(E, b_buf)
    mul!(a_buf, C, b_buf)
    # Convert back to block major and subtract
    @batch minbatch = 1000 for i = 1:n
        for b = 1:block_size
            ix = (i-1)*block_size + b
            jx = (b-1)*n + i
            @inbounds res[ix] -= a_buf[jx]
        end
    end
    # Simple version:
    # res .= B*x - C*(E\(D*x))
    if α != one(T)
        lmul!(α, res)
    end
end

function schur_mul!(res, a_buf, b_buf, r_type, B, C, D, E, x, α, β)
    if r_type == eltype(res)
        schur_mul_float!(res, B, C, D, E, x, α, β)
    else
        res_v = reinterpret(r_type, res)
        x_v = reinterpret(r_type, x)
        block_size = length(r_type)
        schur_mul_block!(res, res_v, a_buf, b_buf, block_size, B, C, D, E, x, x_v, α, β)
    end
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
