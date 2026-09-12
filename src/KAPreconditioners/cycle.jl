function restrict!(bc, Pt::TransposeMap, P::Prolongation, r, backend, block_size)
    k! = restrict_kernel!(backend, block_size)
    k!(bc, Pt.offsets, Pt.fine_rows, Pt.p_indices, P.nzval, r, P.ncol;
       ndrange=P.ncol)
    bc
end

function restrict!(bc::Vector, Pt::TransposeMap, P::Prolongation, r::Vector,
                    ::KernelAbstractions.CPU, block_size)
    foreach_cpu_row(P.ncol) do I
        value = zero(eltype(bc))
        @inbounds @simd for k in Pt.offsets[I]:(Pt.offsets[I+1]-1)
            value += conj(P.nzval[Pt.p_indices[k]]) * r[Pt.fine_rows[k]]
        end
        @inbounds bc[I] = value
    end
    bc
end

function prolong!(x, P::Prolongation, xc, backend, block_size)
    k! = prolong_kernel!(backend, block_size)
    k!(x, P.rowptr, P.colval, P.nzval, xc, P.nrow; ndrange=P.nrow)
    x
end

function prolong!(x::Vector, P::Prolongation, xc::Vector,
                   ::KernelAbstractions.CPU, block_size)
    foreach_cpu_row(P.nrow) do i
        value = zero(eltype(x))
        @inbounds @simd for k in P.rowptr[i]:(P.rowptr[i+1]-1)
            value += P.nzval[k] * xc[P.colval[k]]
        end
        @inbounds x[i] += value
    end
    x
end

function prolong_to!(dst, src, P::Prolongation, xc, backend, block_size)
    k! = prolong_to_kernel!(backend, block_size)
    k!(dst, src, P.rowptr, P.colval, P.nzval, xc, P.nrow; ndrange=P.nrow)
    dst
end

function prolong_to!(dst::Vector, src::Vector, P::Prolongation, xc::Vector,
                      ::KernelAbstractions.CPU, block_size)
    foreach_cpu_row(P.nrow) do i
        value = zero(eltype(dst))
        @inbounds @simd for k in P.rowptr[i]:(P.rowptr[i+1]-1)
            value += P.nzval[k] * xc[P.colval[k]]
        end
        @inbounds dst[i] = src[i] + value
    end
    dst
end

function vcycle!(x, b, H::AMGHierarchy, l::Int;
                  residual=nothing, zero_initial::Bool=false)
    level = H.levels[l]
    if l == length(H.levels)
        if isnothing(level.coarse_solver)
            smooth_level!(x, level.A, b, level.smoother, H.options.coarse_steps;
                     residual=residual, zero_initial=zero_initial)
        else
            coarse_solve!(x, b, level.coarse_solver, H.backend, H.block_size)
        end
        return x
    end
    steps = H.options.smoother.steps
    if steps == 1 && zero_initial && !isnothing(residual) &&
       level.smoother isa SPAI0State
        zero_smooth_residual!(x, level.residual, level.A, residual, level.smoother)
    else
        smooth_level!(x, level.A, b, level.smoother, steps;
                 residual=residual, zero_initial=zero_initial)
        residual!(level.residual, level.A, x, b)
    end
    restrict!(level.rhs, level.Pt, level.P, level.residual, H.backend, H.block_size)
    # The recursive zero-initial path overwrites its complete correction on the
    # first smoothing step (or in the direct coarse solve), so clearing this
    # persistent buffer first only adds memory traffic and a device launch.
    coarse_correction = vcycle!(level.correction, level.rhs, H, l+1;
                                 residual=level.rhs, zero_initial=true)
    if H.options.cycle == :W
        coarse_correction = vcycle!(coarse_correction, level.rhs, H, l+1)
    end
    if H.options.cycle == :V && steps == 1 && level.smoother isa SPAI0State
        # Put the prolongated iterate in the smoother scratch and the Jacobi
        # result back in x. At every level this replaces prolong + Jacobi +
        # scratch copy with two kernels and keeps the destination stable.
        prolong_to!(level.smoother.temporary, x, level.P, coarse_correction,
                     H.backend, H.block_size)
        smooth_once_to!(x, level.smoother.temporary, level.A, b, level.smoother)
        return x
    end
    prolong!(x, level.P, coarse_correction, H.backend, H.block_size)
    if H.options.cycle == :V
        # ILU-family smoothers fuse residual formation and correction here.
        smooth_result!(x, level.A, b, level.smoother, steps)
    else
        smooth_level!(x, level.A, b, level.smoother, steps)
    end
end

"""Apply one cycle to the current iterate without clearing it."""
function cycle!(x::AbstractVector, H::AMGHierarchy, b::AbstractVector)
    result = vcycle!(x, b, H, 1)
    result === x || copyto!(x, result)
    x
end
cycle!(x::AbstractVector, H::AMGHierarchy, ::Any, b::AbstractVector) = cycle!(x, H, b)

"""Apply the hierarchy as a linear preconditioner (`x` is cleared first)."""
function apply!(x::AbstractVector, H::AMGHierarchy, b::AbstractVector)
    length(x) == size(H, 2) || throw(DimensionMismatch())
    length(b) == size(H, 1) || throw(DimensionMismatch())
    # The zero-initial V-cycle writes every entry of x before reading it. This
    # saves both the finest SpMV and a separate clear of every level buffer.
    result = vcycle!(x, b, H, 1; residual=b, zero_initial=true)
    result === x || copyto!(x, result)
    x
end

function apply!(x::AbstractVector, H::AMGHierarchy, A, b::AbstractVector)
    size(A) == size(H) || throw(DimensionMismatch("operator and hierarchy sizes differ"))
    apply!(x, H, b)
end

LinearAlgebra.ldiv!(x::AbstractVector, H::AMGHierarchy, b::AbstractVector) = apply!(x, H, b)
LinearAlgebra.mul!(x::AbstractVector, H::AMGHierarchy, b::AbstractVector) = apply!(x, H, b)

function Base.:*(H::AMGHierarchy{Tv}, b::AbstractVector) where Tv
    x = similar(b, Tv, size(H, 2))
    apply!(x, H, b)
end

"""
    solve!(x, H, b; rtol=1e-8, atol=0, maxiter=100)

Solve with repeated AMG cycles. Returns `(x, iterations)` and records the final
relative residual in `H.last_residual`.
"""
function solve!(x::AbstractVector, H::AMGHierarchy, b::AbstractVector;
                rtol::Real=1e-8, atol::Real=0, maxiter::Integer=100)
    maxiter >= 0 || throw(ArgumentError("maxiter must be non-negative"))
    bnorm = norm(b)
    threshold = max(Float64(atol), Float64(rtol) * Float64(bnorm))
    level = H.levels[1]
    for iteration in 0:Int(maxiter)
        residual!(level.residual, level.A, x, b)
        residual = Float64(norm(level.residual))
        H.last_iterations = iteration
        H.last_residual = if iszero(bnorm)
            residual
        else
            residual / Float64(bnorm)
        end
        residual <= threshold && return x, iteration
        iteration == maxiter && break
        # Reuse the convergence-check residual as the first Jacobi update.
        # Calling cycle! here would immediately calculate the same A*x again.
        result = vcycle!(x, b, H, 1; residual=level.residual)
        result === x || copyto!(x, result)
    end
    x, Int(maxiter)
end

solve!(x::AbstractVector, H::AMGHierarchy, A, b::AbstractVector; kwargs...) =
    (size(A) == size(H) || throw(DimensionMismatch()); solve!(x, H, b; kwargs...))
