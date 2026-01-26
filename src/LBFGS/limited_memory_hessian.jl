"""
Limited Memory Hessian Approximation

This file implements L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) 
operations for quasi-Newton optimization. It provides efficient multiplication 
with Hessian and inverse Hessian approximations and the corresponding operations
restricted to given subspaces.

H::LimitedMemoryHessian represents the L-BFGS Hessian approximation, i.e., H ≈ ∇²f(x)
"""

function update!(H::LimitedMemoryHessian, s, y)
    # store new vectors
    @assert length(s) == length(y) "Dimension mismatch between s and y"
    @assert length(s) == size(H.S, 1) || H.it_count == 0 "Dimension mismatch with stored vectors"
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

function reset!(H::LimitedMemoryHessian)
    # reset to scaled identity (keep initial scaling)
    H.S = []
    H.Y = []
    H.it_count = 0
    return H
end

"""
Handle multiplication with inverse Hessian. Eqivalent to:
1. Un-reduced: w = H^-1 * v
2. Reduced: 
    if nullspace is provided as vector of bools b:
        w(b) = 0
        w(!b) = H(!b, !b)^-1 * v(!b)
    if nullspace is provided as matrix Q with orthonormal columns:
        w = Z *(Z'*H*Z)^-1 * Z'*v 
    where Z = (I - Q*Q')
"""

# left-multiplication with inverse of un-reduced Hessian
import Base.\
function \(H::LimitedMemoryHessian, v::Vector)
    if H.it_count == 0
        r = (H.sign / H.init_scale) .* v
        return r
    end
    @assert size(v) == (size(H.S, 1),) "Dimension mismatch of vector"
    # we use subspace version with empty nullspace
    th = apply_initial(H, 1.0)
    null_space = falses(size(v, 1))
    return subspace_product_inverse(H.S, H.Y, th, null_space, v, true)
    # below is for reference the standard recursion for L-BFGS, but this is slightly less 
    # stable than the above matrix version for large vectors and large m
    #
    #    nVec = size(H.S, 2)
    #    rho = fill(NaN, (1, nVec))
    #    alpha = fill(NaN, (1, nVec))
    #    for k in nVec:-1:1
    #        rho[k] = 1 / (H.S[:, k]' * H.Y[:, k])
    #        alpha[k] = rho[k] * (H.S[:, k]' * v)
    #        v = v - alpha[k] * H.Y[:, k]
    #    end
    #    r = apply_initial!(H, v)
    #    for k in 1:nVec
    #        beta = rho[k] * (H.Y[:, k]' * r)
    #        r = r + (alpha[k] - beta) * H.S[:, k]
    #    end
end

# left-multiplication with inverse of reduced Hessian applied to corresponding subspace
# First version: nullspace defined by a vector of booleans
function apply_reduced_hessian_inverse(H::LimitedMemoryHessian, v::Vector, null_space::Union{BitVector, Vector{Bool}})
    if H.it_count == 0
        r = (H.sign / H.init_scale) .* v
        r[null_space] .= 0.0
    else
        @assert size(v) == (size(H.S, 1),) "Dimension mismatch"
        th = apply_initial(H, 1.0)
        r = subspace_product_inverse(H.S, H.Y, th, null_space, v, true)
    end
    return r
end
# Second version: nullspace defined by a matrix with orthonormal columns
function apply_reduced_hessian_inverse(H::LimitedMemoryHessian, v::Vector, null_space)
    if isempty(null_space)
        # empty nullspace, do standard un-reduced version
        return H \ v
    end
    @assert length(v) == size(null_space, 1) "Null-space dimension mismatch"
    if H.it_count == 0
        r = (H.sign / H.init_scale) .* v
        r = r - null_space * (null_space' * r)
    else
        @assert length(v) == size(H.S, 1) "Dimension mismatch"
        th = apply_initial(H, 1.0)
        r = subspace_product_inverse(H.S, H.Y, th, null_space, v, false)
    end
    return r
end

"""
Handle multiplication with Hessian: 
1. Un-reduced: w = H * v
2. Reduced: w(!b) = H(!b, !b) * v(!b) / w = Z * (Z'*H*Z) * Z'*v 
"""

import Base.*
function *(H::LimitedMemoryHessian, v::Vector)
    # apply multiplication with un-reduced Hessian 
    if H.it_count == 0
        r = (H.sign * H.init_scale) .* v
        return r
    end
    @assert length(v) == size(H.S, 1) "Dimension mismatch"
     # use subspace version with empty nullspace
    th = apply_initial(H, 1.0)
    null_space = falses(size(v, 1))
    return subspace_product(H.S, H.Y, th, null_space, v, true)
end

# left-multiplication with reduced Hessian applied to corresponding subspace
# First version: nullspace defined by a vector of booleans
function apply_reduced_hessian(H::LimitedMemoryHessian, v::Vector, null_space::Union{BitVector, Vector{Bool}})
    @assert length(v) == length(null_space) "Null-space dimension mismatch"
    if H.it_count == 0
        r = (H.sign * H.init_scale) .* v
        r[null_space] .= 0.0
    else
        @assert length(v) == size(H.S, 1) "Dimension mismatch"
        th = apply_initial(H, 1.0)
        r = subspace_product(H.S, H.Y, th, null_space, v, true)
    end
    return r
end

# Second version: nullspace defined by matrix with orthonormal columns
function apply_reduced_hessian(H::LimitedMemoryHessian, v::Vector, null_space)
    if !isempty(null_space)
        @assert length(v) == size(null_space, 1) "Null-space dimension mismatch"
    end
    if H.it_count == 0
        r = (H.sign * H.init_scale) .* v
        r = r - null_space * (null_space' * r)
    else
        @assert length(v) == size(H.S, 1) "Dimension mismatch"
        th = apply_initial(H, 1.0)
        r = subspace_product(H.S, H.Y, th, null_space, v, false)
    end
    return r
end

import Base.-
function -(H::LimitedMemoryHessian)
    Hm = deepcopy(H)
    Hm.sign = -H.sign
    return Hm
end

function apply_initial(H::LimitedMemoryHessian, v)
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
function subspace_product_inverse(S, Y, th, null_space, v, is_bool_type)
    if null_space_has_full_rank(null_space, is_bool_type)
        # nothing to do, return zero vector
        r = 0 * v
    else
        n = size(S, 2)
        W = hcat(Y, S / th)
        tmp = S' * Y
        L = tril(tmp, -1)
        D = diagm(diag(tmp))
        M = vcat(hcat(-D, L'), hcat(L, (S' * S) / th))
        # projection onto active subspace (u -> Z*Z'*u)
        proj_sub = get_projection_operator(null_space, is_bool_type)
        r = M \ (W' * proj_sub(v))
        #r = (sparse(1:(2 * n), 1:(2 * n), 1) - th * (M \ (W' * proj_sub(W)))) \ r
        r = (I - th * (M \ (W' * proj_sub(W)))) \ r
        r = proj_sub(th * v + th^2 * (W * r))
    end
    return r
end

function subspace_product(S, Y, th, null_space, v, is_bool_type)
    # perform multiplication with reduced Hessian
    # use matrix form from e.g., above reference eq. 3.2
    if null_space_has_full_rank(null_space, is_bool_type)
        # nothing to do, return zero vector
        r = 0 * v
    else
        n = size(S, 2)
        W = hcat(Y, S / th)
        tmp = S' * Y
        L = tril(tmp, -1)
        D = diagm(diag(tmp))
        M = vcat(hcat(-D, L'), hcat(L, (S' * S) / th))
        # projection onto active subspace (u -> Z*Z'*u)
        proj_sub = get_projection_operator(null_space, is_bool_type)
        r = proj_sub(v)
        r = proj_sub(r/th - W * (M \ (W' * r)))
    end
    return r
end

function null_space_has_full_rank(null_space, is_bool_type)
if is_bool_type
        return all(null_space)
    else
        return size(null_space, 1) <= size(null_space, 2)
    end
end

function get_projection_operator(null_space, is_bool_type)
    if null_space_is_empty(null_space, is_bool_type)
        return identity
    end
    if is_bool_type
        return (u) -> (u[null_space, :] .= 0.0; return u)
    else
        return (u) -> u - null_space * (null_space' * u)
    end
end

function null_space_is_empty(null_space, is_bool_type)
if is_bool_type
        return !any(null_space)
    else
        return isempty(null_space) || size(null_space, 2) == 0 
    end
end

"""
For debugging and testing purposes
"""
function full_matrix(H::LimitedMemoryHessian)
    n = size(H.S, 1)
    Id = Matrix{Float64}(I, n, n)
    Hfull = zeros(n, n)
    for i in 1:n
        e = Id[:, i]
        Hfull[:, i] = H * e
    end
    return Hfull
end

function full_inverse_matrix(H::LimitedMemoryHessian)
    n = size(H.S, 1)
    Id = Matrix{Float64}(I, n, n)
    Hfull = zeros(n, n)
    for i in 1:n
        e = Id[:, i]
        Hfull[:, i] = H \ e
    end
    return Hfull
end
