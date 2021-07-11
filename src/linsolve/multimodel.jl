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
    @views x = reshape(reshape(vec(a), block_size,:)', :)
    return x
end

function prepare_solve!(sys::MultiLinearizedSystem)
    if do_schur(sys)
        B, C, D, E, a, b = get_schur_blocks!(sys, true, update = true)
        tmp = C*(E\b)
        e = eltype(B)
        if e == Float64
           da = tmp
        else
            bz = size(e, 1)
            da = equation_major_to_block_major_view(tmp, bz)
        end
        @. a -= da
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
        apply! = (res, x, α, β) -> schur_mul!(res, T, B, C, D, E, x, α, β)
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
        dy = E\(b - D*dx)
        n = length(dx)
        m = length(sys.dx)
        sys.dx[1:n] = -dx
        sys.dx[(n+1):m] = -dy
    else
        sys.dx .= -dx
    end
end

function schur_mul!(res, r_type::Float64, B, C, D, E, x, α, β::T) where T
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

function schur_mul!(res, r_type, B, C, D, E, x, α, β::T) where T
    as_svec = (x) -> reinterpret(r_type, x)
    res_v = as_svec(res)
    x_v = as_svec(x)
    if β == zero(T)
        # compute B*x
        mul!(res_v, B, x_v)
        tmp = C*(E\(D*x))
        if r_type == Float64
            drs = tmp
        else
            block_size = length(r_type)
            drs = equation_major_to_block_major_view(tmp, block_size)
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
