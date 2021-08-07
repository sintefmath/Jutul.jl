import Base.getindex

function Base.getindex(ls::MultiLinearizedSystem, i, j = i)
    return ls.subsystems[i, j]
end

do_schur(sys) = sys.reduction == :schur_apply

function equation_major_to_block_major_view(a, block_size)
    @views x = reshape(reshape(vec(a), :, block_size)', :)
    return x
end

function block_major_to_equation_major_view(a, block_size)
    @views x = reshape(reshape(vec(a), block_size, :)', :)
    return x
end

@inline major_to_minor(n, m, i) = n*((i - 1) % m) + 1 + ((i -1) ÷ m)
@inline from_block_index(bz, nc, i) = major_to_minor(bz, nc, i)
@inline from_unit_index(bz, nc, i) = major_to_minor(nc, bz, i)



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

function linear_operator(sys::MultiLinearizedSystem)
    if do_schur(sys)
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
        B, C, D, E, a, b = get_schur_blocks!(sys)
        bz = block_size(sys[1, 1])
        if bz == 1
            Δx = dx
        else
            Δx = block_major_to_equation_major_view(dx, bz)
        end

        n = length(dx)
        m = length(sys.dx)

        A = view(sys.dx, 1:n)
        B = view(sys.dx, (n+1):m)

        @. A = -dx
        buf_a = sys.schur_buffer[1]
        buf_b = sys.schur_buffer[2]
        # We want to do (in-place):
        # dy = B = -E\(b - D*Δx) = E\(D*Δx - b)
        buf_a .= Δx
        mul!(buf_b, D, buf_a)
        # now buf_b = D*Δx
        @. buf_b -= b
        ldiv!(B, E, buf_b)
    else
        sys.dx .= -dx
    end
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
    # compute B*x
    mul!(res_v, B, x_v)
    n = length(res) ÷ block_size
    # Convert to cell major view
    @turbo for i in 1:(n*block_size)
        a_buf[i] = x[from_block_index(block_size, n, i)]
    end
    mul!(b_buf, D, a_buf)
    ldiv!(E, b_buf)
    mul!(a_buf, C, b_buf)
    # Convert back to block major and subtract
    @turbo for i in 1:(n*block_size)
        res[i] -= a_buf[from_unit_index(block_size, n, i)]
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
