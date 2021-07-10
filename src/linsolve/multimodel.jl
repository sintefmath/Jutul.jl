import Base.getindex

function Base.getindex(ls::MultiLinearizedSystem, i, j = i)
    return ls.subsystems[i, j]
end

do_schur(sys) = sys.reduction == :schur_apply

function prepare_solve!(sys::MultiLinearizedSystem)
    if do_schur(sys)
        B, C, D, E, a, b = get_schur_blocks!(sys, true, update = true)
        a -= C*(E\b)
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
        a = keep.r
        b = elim.r
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
        n = size(B, 1)
        function schur_mul!(res, x, α, β::T) where T
            if β == zero(T)
                tmp = B*x - C*(E\(D*x))
                res .= tmp
                # mul!(res, jac, x)
                if α != one(T)
                    lmul!(α, res)
                end
            else
                error("Not implemented yet.")
            end
        end
        op = LinearOperator(Float64, n, n, false, false, schur_mul!)
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
