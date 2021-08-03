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

function prepare_solve!(sys::MultiLinearizedSystem)
    if do_schur(sys)
        B, C, D, E, a, b = get_schur_blocks!(sys, true, update = true)
        e = eltype(B)
        is_float = e == Float64
        Δa = C*(E\b)
        
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
        m = size(E, 1)
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

function schur_mul!(res, a_buf, b_buf, r_type, B, C, D, E, x, α, β::T) where T
    # display(mean(B.nzval))
    # display(mean(C.nzval))
    # display(mean(E.nzval))
    # display(mean(D.nzval))
    # display(Matrix(E))
    if false
        tmp = B*x - C*(E\(D*x))
        res .= tmp
        if α != one(T)
            lmul!(α, res)
        end

        return 
    end
    is_float = r_type == eltype(res)
    if is_float
        as_svec = (x) -> x
    else
        as_svec = (x) -> reinterpret(r_type, x)
    end
    res_v = as_svec(res)
    x_v = as_svec(x)
    if β == zero(T)
        # compute B*x
        mul!(res_v, B, x_v)
        if is_float
            drs = C*(E\(D*x))
        else
            block_size = length(r_type)
            M = (x) -> block_major_to_equation_major_view(x, block_size)
            M⁻¹ = (x) -> equation_major_to_block_major_view(x, block_size)
            
            a_buf .= M(x)
            mul!(b_buf, D, a_buf)
            ldiv!(E, b_buf)
            mul!(a_buf, C, b_buf)
            drs = M⁻¹(a_buf)
            # @time drs = M⁻¹(C*(E\(D*M(x))))
        end

        @. res -= drs
        # Simple version:
        # res .= B*x - C*(E\(D*x))
        if α != one(T)
            lmul!(α, res)
        end
    else
        error("Not implemented yet.")
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
