"""
Limited-memory BFGS (L-BFGS) Hessian approximation (legacy implementation).
NOTE: This interfaces the *inverse* of the Hessian approximation, i.e., 
    H * v = (∇²f)⁻¹ * v.
"""


function update(H::LimitedMemoryHessianLegacy, s, y)
    # store new vectors
    H.it_count += 1
    pix = (:)
    if H.it_count > H.m
        pix = 2:H.m
    end
    if H.it_count == 1
        H.S = reshape(s, :, 1)
        H.Y = reshape(y, :, 1)
    else
        H.S = hcat(H.S[:, pix], reshape(s, :, 1))
        H.Y = hcat(H.Y[:, pix], reshape(y, :, 1))
    end
    if H.it_count == 1
        # limit m to number of vars
        if H.m > length(s)
            H.m = length(s)
            Jutul.jutul_message("LBFGS", "Resetting 'm' to number of parameters: m = $(length(s))", color = :yellow)
        end
    end
    return H
end

function reset(H::LimitedMemoryHessianLegacy)
    H.S = []
    H.Y = []
    H.it_count = 0
    H.nullspace = []
    return H
end

import Base.*
function *(H::LimitedMemoryHessianLegacy, v::Vector)
    # apply mulitplication H*v
    if isa(v, LimitedMemoryHessianLegacy)
        error("Only right-multiplciation is supported")
    end
    if H.it_count == 0
        r = (H.sign * H.init_scale) .* v
        if !isempty(H.nullspace)
            if isa(H.nullspace, Vector{Bool})
                r = r .* (.!H.nullspace)
            else
                r = r - H.nullspace * (H.nullspace' * r)
            end
        end
    else
        @assert size(v) == (size(H.S, 1),) "Dimension mismatch"
        if isempty(H.nullspace)
            # do standard L-BFGS
            nVec = size(H.S, 2)
            rho = fill(NaN, (1, nVec))
            alpha = fill(NaN, (1, nVec))
            for k in nVec:-1:1
                rho[k] = 1 / (H.S[:, k]' * H.Y[:, k])
                alpha[k] = rho[k] * (H.S[:, k]' * v)
                v = v - alpha[k] * H.Y[:, k]
            end
            r = apply_initial!(H, v)
            for k in 1:nVec
                beta = rho[k] * (H.Y[:, k]' * r)
                r = r + (alpha[k] - beta) * H.S[:, k]
            end
        else
            # do subspace version
            th = apply_initial!(H, 1.0)
            r = subspace_product(H.S, H.Y, th, H.nullspace, v)
        end
    end
    return r
end

import Base.-
function -(H::LimitedMemoryHessianLegacy)
    H.sign = -H.sign
    return H
end

function apply_initial!(H::LimitedMemoryHessianLegacy, v)
    if H.init_strategy == :static
        r = H.sign * H.init_scale * v
    elseif H.init_strategy == :dynamic
        s = H.S[:, end]
        y = H.Y[:, end]
        r = ((s' * y) / (y' * y)) * v
    else
        error("Unknown strategy: $H.init_strategy")
    end
    return r
end

function set_nullspace!(H::LimitedMemoryHessianLegacy, Q = nothing)
    if isnothing(Q)
        H.nullspace = []
    else
        if H.it_count > 0 && !isa(Q, Vector{Bool})
            @assert size(Q, 1) == size(H.S, 1) "Dimension mismatch"
            @assert size(Q, 1) >= size(Q, 2) "Number of columns in nullspace matrix exceeds number of rows"
        end
        H.nullspace = Q
    end
    return H
end

"""
Perform product restricted to nullspace of Q-colunms, i.e.,
r = Z * (Z'*Hessian*Z)^-1 * Z'*v
with Z = null(Q')
Formula from:
Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995).
A limited memory algorithm for bound constrained optimization.
SIAM Journal on scientific computing, 16(5), 1190-1208.

# Arguments

- `S`: control diffs
- `Y`: gradient diffs
- t`h`: inverse Hessian scaling
- `Q`: nullspace of active subspace
- `v`: vector
"""
function subspace_product(S, Y, th, Q, v)
    if size(Q, 1) <= size(Q, 2)
        # % null(Q')=[], return zero
        r = 0 * v
    else
        n = size(S, 2)
        W = hcat(Y, S / th)
        tmp = S' * Y
        L = tril(tmp, -1)
        D = diagm(diag(tmp))
        M = vcat(hcat(-D, L'), hcat(L, (S' * S) / th))
        # projection onto active subspace (u -> Z*Z'*u)
        if isa(Q, Vector{Bool})
            projSub = (u) -> u .* .!Q
        else
            projSub = (u) -> u - Q * (Q' * u)
        end
        r = M \ (W' * projSub(v))
        r = (sparse(1:(2 * n), 1:(2 * n), 1) - th * (M \ (W' * projSub(W)))) \ r
        r = projSub(th * v + th^2 * (W * r))
    end
    return r
end
