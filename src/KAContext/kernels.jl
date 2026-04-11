# KernelAbstractions kernels for Jutul equation evaluation,
# Jacobian assembly, variable updates, and convergence checks.

using ..Jutul: SimpleHeatEquation, SimpleHeatSystem
using ..Jutul: CartesianMesh, physical_representation, cell_ijk, cell_dims, cell_index

import GPUArrays

# ──────────────────────────────────────────────────────────────
# 1. fill_equation_entries! – extract AD values/partials into
#    the Jacobian nzval and residual vectors.
# ──────────────────────────────────────────────────────────────

@kernel function ka_fill_entries_generic_kernel!(
        nz, r, @Const(entries), @Const(jac_pos),
        @Const(vpos), @Const(dpos),
        nu::Int, ne_val::Int, np_val::Int
    )
    i = @index(Global)
    @inbounds diag_index = dpos[i]
    @inbounds for jno_off in vpos[i]:(vpos[i+1]-1)
        fill_residual = (jno_off == diag_index)
        for e in 1:ne_val
            @inbounds a = entries[e, jno_off]
            if fill_residual
                @inbounds r[e, i] = ForwardDiff.value(a)
            end
            for d in 1:np_val
                pos_idx = (e-1)*np_val + d
                @inbounds pos = jac_pos[pos_idx, jno_off]
                @inbounds nz[pos] = a.partials[d]
            end
        end
    end
end

function Jutul.fill_equation_entries!(nz, r, model::SimulationModel{<:Any, <:Any, <:Any, <:KernelAbstractionsContext}, cache::GenericAutoDiffCache_device)
    backend = model.context.backend
    nu = number_of_entities(cache)
    ne = equations_per_entity(cache)
    np = number_of_partials(cache)
    entries = cache.entries
    jac_pos = cache.jacobian_positions
    vpos = cache.vpos
    dpos = cache.diagonal_positions
    if isnothing(dpos)
        error("GenericAutoDiffCache_device without diagonal_positions not supported on KA backend.")
    end
    kernel = ka_fill_entries_generic_kernel!(backend)
    kernel(nz, r, entries, jac_pos, vpos, dpos, nu, ne, np; ndrange = nu)
end

@kernel function ka_fill_entries_compact_kernel!(
        nz, r, @Const(entries), @Const(jac_pos),
        nu::Int, ne_val::Int, np_val::Int
    )
    i = @index(Global)
    @inbounds for e in 1:ne_val
        @inbounds a = entries[e, i]
        @inbounds r[i + nu*(e-1)] = ForwardDiff.value(a)
        for d in 1:np_val
            pos_idx = (e-1)*np_val + d
            @inbounds pos = jac_pos[pos_idx, i]
            @inbounds nz[pos] = a.partials[d]
        end
    end
end

function Jutul.fill_equation_entries!(nz, r, model::SimulationModel{<:Any, <:Any, <:Any, <:KernelAbstractionsContext}, cache::CompactAutoDiffCache_device)
    backend = model.context.backend
    nu = number_of_entities(cache)
    ne = equations_per_entity(cache)
    np = number_of_partials(cache)
    entries = cache.entries
    jac_pos = cache.jacobian_positions
    kernel = ka_fill_entries_compact_kernel!(backend)
    kernel(nz, r, entries, jac_pos, nu, ne, np; ndrange = nu)
end

# ──────────────────────────────────────────────────────────────
# 2. Equation evaluation – CPU fallback for GenericAutoDiffCache
#    on device. Copies entries to CPU, evaluates equations with
#    AD, then copies back to device.
# ──────────────────────────────────────────────────────────────

function Jutul.update_equation_for_entity!(cache::GenericAutoDiffCache_device, eq::Jutul.JutulEquation, state, state0, model::SimulationModel{<:Any, <:Any, <:Any, <:KernelAbstractionsContext}, dt)
    # Copy device arrays to CPU for equation evaluation
    entries_cpu = Array(cache.entries)
    vpos_cpu = Array(cache.vpos)
    vars_cpu = Array(cache.variables)

    ne = number_of_entities(cache)
    T_ad = eltype(entries_cpu)

    # Convert device state to CPU
    state_cpu = _collect_state_to_cpu(state)
    state0_cpu = _collect_state_to_cpu(state0)

    # Create local AD wrappers
    local_state = Jutul.local_ad(state_cpu, 1, T_ad)
    local_state0 = Jutul.local_ad(state0_cpu, 1, T_ad)

    for i in 1:ne
        ldisc = local_discretization(eq, i)
        for j in vpos_cpu[i]:(vpos_cpu[i+1]-1)
            v_i = @views entries_cpu[:, j]
            var = vars_cpu[j]
            state_i = Jutul.new_entity_index(local_state, var)
            state0_i = Jutul.new_entity_index(local_state0, var)
            update_equation_in_entity!(v_i, i, state_i, state0_i, eq, model, dt, ldisc)
        end
    end
    # Copy back to device
    copyto!(cache.entries, entries_cpu)
end

function _collect_state_to_cpu(state)
    if state isa NamedTuple
        pairs_list = Pair{Symbol, Any}[]
        for k in keys(state)
            v = state[k]
            if v isa GPUArrays.AbstractGPUArray
                push!(pairs_list, k => Array(v))
            else
                push!(pairs_list, k => v)
            end
        end
        return NamedTuple(pairs_list)
    else
        return state
    end
end

# ──────────────────────────────────────────────────────────────
# 3. CSR matrix-vector multiply kernel
# ──────────────────────────────────────────────────────────────

@kernel function ka_csr_spmv_kernel!(
        y, @Const(nzval), @Const(colval), @Const(rowptr), @Const(x), α
    )
    row = @index(Global)
    @inbounds rp_start = rowptr[row]
    @inbounds rp_end   = rowptr[row+1] - 1
    v = zero(eltype(y))
    @inbounds for nz in rp_start:rp_end
        col = colval[nz]
        v += nzval[nz] * x[col]
    end
    @inbounds y[row] = α * v
end

@kernel function ka_csr_spmv_add_kernel!(
        y, @Const(nzval), @Const(colval), @Const(rowptr), @Const(x), α
    )
    row = @index(Global)
    @inbounds rp_start = rowptr[row]
    @inbounds rp_end   = rowptr[row+1] - 1
    v = zero(eltype(y))
    @inbounds for nz in rp_start:rp_end
        col = colval[nz]
        v += nzval[nz] * x[col]
    end
    @inbounds y[row] += α * v
end

"""
    ka_mul!(y, A::StaticSparsityMatrixCSR, x, α, β, backend)

KA-based CSR matrix-vector multiply: y = α*A*x + β*y
"""
function ka_mul!(y, A::StaticSparsityMatrixCSR, x, α, β, backend)
    n = size(A, 1)
    nzval = nonzeros(A)
    cv = colvals(A)
    rp = to_device(backend, A.rowptr)
    if β == 0
        kernel = ka_csr_spmv_kernel!(backend)
        kernel(y, nzval, cv, rp, x, α; ndrange = n)
    else
        if β != 1
            rmul!(y, β)
        end
        kernel = ka_csr_spmv_add_kernel!(backend)
        kernel(y, nzval, cv, rp, x, α; ndrange = n)
    end
    return y
end

# Override mul! for device-backed CSR matrices
function LinearAlgebra.mul!(y::AbstractVector, A::StaticSparsityMatrixCSR{Tv, Ti, V, I}, x::AbstractVector, α::Number, β::Number) where {Tv, Ti, V<:GPUArrays.AbstractGPUVector, I}
    backend = get_backend(nonzeros(A))
    ka_mul!(y, A, x, α, β, backend)
    return y
end

# ──────────────────────────────────────────────────────────────
# 4. Variable update kernel
# ──────────────────────────────────────────────────────────────

@kernel function ka_update_variable_kernel!(v, @Const(dx), w, minval_f, maxval_f, has_min::Bool, has_max::Bool)
    i = @index(Global)
    @inbounds old = v[i]
    updated = ForwardDiff.value(old) + w * dx[i]
    if has_min
        updated = max(updated, minval_f)
    end
    if has_max
        updated = min(updated, maxval_f)
    end
    @inbounds v[i] = old - ForwardDiff.value(old) + updated
end

@kernel function ka_update_values_ad_kernel!(v, @Const(next))
    i = @index(Global)
    @inbounds old = v[i]
    @inbounds new_val = next[i]
    @inbounds v[i] = old - ForwardDiff.value(old) + ForwardDiff.value(new_val)
end

@kernel function ka_update_values_kernel!(v, @Const(next))
    i = @index(Global)
    @inbounds v[i] = next[i]
end

# ──────────────────────────────────────────────────────────────
# 5. Convergence check helpers
# ──────────────────────────────────────────────────────────────

@kernel function ka_abs_kernel!(out, @Const(r))
    i = @index(Global)
    @inbounds out[i] = abs(r[i])
end

# ──────────────────────────────────────────────────────────────
# 6. Linear solve for KA context
#    Converts device CSR to CPU sparse, solves, copies dx back
# ──────────────────────────────────────────────────────────────

function Jutul.linear_solve!(sys::LinearizedSystem{<:Any, <:StaticSparsityMatrixCSR{Tv, Ti, V, I}}, ::Nothing, context::KernelAbstractionsContext, arg...; dx = sys.dx, r = sys.r_buffer, atol = nothing, rtol = nothing, executor = Jutul.default_executor()) where {Tv, Ti, V<:GPUArrays.AbstractGPUVector, I}
    # Copy CSR data to CPU sparse matrix and solve
    nzval_cpu = Array(nonzeros(sys.jac))
    colval_cpu = Array(colvals(sys.jac))
    rowptr_cpu = sys.jac.rowptr
    m, n = size(sys.jac)
    r_cpu = Array(r)

    # Reconstruct CSC from CSR data: CSR stores A as At in CSC format
    # So At_csc has colptr=rowptr, rowval=colval, nzval=nzval
    At_csc = SparseMatrixCSC(n, m, rowptr_cpu, colval_cpu, nzval_cpu)

    # Solve A*x = r using A = At_csc'
    dx_cpu = -(At_csc' \ r_cpu)
    @assert all(isfinite, dx_cpu) "Linear solve resulted in non-finite values."
    copyto!(dx, dx_cpu)
    return Jutul.linear_solve_return()
end

# ──────────────────────────────────────────────────────────────
# 7. GPU-compatible update_values! for device arrays
#    Avoids scalar indexing by using KA kernels
# ──────────────────────────────────────────────────────────────

@kernel function ka_update_values_ad_to_real_kernel!(v, @Const(next))
    i = @index(Global)
    @inbounds v[i] = ForwardDiff.value(next[i])
end

# Dual{Tag} array <- Real array: preserve partials, update value
function Jutul.update_values!(v::GPUArrays.AbstractGPUArray{<:ForwardDiff.Dual}, next::GPUArrays.AbstractGPUArray{<:Real})
    backend = get_backend(v)
    kernel = ka_update_values_ad_kernel!(backend)
    kernel(v, next; ndrange = length(v))
    return v
end

# Real array <- Dual{Tag} array: extract value
function Jutul.update_values!(v::GPUArrays.AbstractGPUArray{<:AbstractFloat}, next::GPUArrays.AbstractGPUArray{<:ForwardDiff.Dual})
    backend = get_backend(v)
    kernel = ka_update_values_ad_to_real_kernel!(backend)
    kernel(v, next; ndrange = length(v))
    return v
end

# Dual{Tag} array <- Dual{Tag} array: direct copy
function Jutul.update_values!(v::GPUArrays.AbstractGPUArray{T}, next::GPUArrays.AbstractGPUArray{T}) where {T<:ForwardDiff.Dual}
    copyto!(v, next)
    return v
end

# Any GPU array <- any GPU array: direct broadcast
function Jutul.update_values!(v::GPUArrays.AbstractGPUArray, next::GPUArrays.AbstractGPUArray)
    v .= next
    return v
end

# ──────────────────────────────────────────────────────────────
# 8. GPU-compatible variable update (avoids scalar indexing)
# ──────────────────────────────────────────────────────────────

function Jutul.update_jutul_variable_internal!(v::GPUArrays.AbstractGPUVector, active, p, dx, w)
    minval = Jutul.minimum_value(p)
    maxval = Jutul.maximum_value(p)
    has_min = !isnothing(minval)
    has_max = !isnothing(maxval)
    minval_f = has_min ? Float64(minval) : 0.0
    maxval_f = has_max ? Float64(maxval) : 0.0
    active_dev = _ensure_device(v, active)
    # Flatten dx to a vector on device
    dx_flat = _ensure_device_vec(v, dx)
    backend = get_backend(v)
    kernel = ka_update_primary_variable_kernel!(backend)
    kernel(v, active_dev, dx_flat, Float64(w), minval_f, maxval_f, has_min, has_max; ndrange = length(active))
    return v
end

@kernel function ka_update_primary_variable_kernel!(v, @Const(active), @Const(dx), w, minval_f, maxval_f, has_min::Bool, has_max::Bool)
    idx = @index(Global)
    @inbounds a_i = active[idx]
    @inbounds old = v[a_i]
    dv = w * dx[idx]
    updated = ForwardDiff.value(old) + dv
    if has_min
        updated = max(updated, minval_f)
    end
    if has_max
        updated = min(updated, maxval_f)
    end
    @inbounds v[a_i] = old - ForwardDiff.value(old) + updated
end

function _ensure_device(ref::GPUArrays.AbstractGPUArray, arr::AbstractArray)
    if arr isa GPUArrays.AbstractGPUArray
        return arr
    else
        backend = get_backend(ref)
        return to_device(backend, Array(arr))
    end
end

function _ensure_device(ref::GPUArrays.AbstractGPUArray, arr)
    return arr
end

function _ensure_device_vec(ref::GPUArrays.AbstractGPUArray, dx)
    # Convert dx (potentially Adjoint or reshaped view) to a flat GPU vector
    backend = get_backend(ref)
    dx_cpu = vec(Array(collect(dx)))
    return to_device(backend, dx_cpu)
end

function _ensure_device_vec(ref::GPUArrays.AbstractGPUArray, dx::GPUArrays.AbstractGPUVector)
    return dx
end

# ──────────────────────────────────────────────────────────────
# 9. GPU-compatible increment_norm (avoids scalar indexing)
# ──────────────────────────────────────────────────────────────

function Jutul.increment_norm(dX::Union{GPUArrays.AbstractGPUArray, LinearAlgebra.Adjoint{<:Any, <:GPUArrays.AbstractGPUArray}}, state, model, X, pvar)
    T = eltype(dX)
    scale = @something Jutul.variable_scale(pvar) one(T)
    # Use GPU-compatible reductions
    dX_flat = vec(Array(dX))
    max_v = maximum(abs, dX_flat)
    sum_v = sum(abs, dX_flat)
    return (sum = scale*sum_v, max = scale*max_v)
end

# ──────────────────────────────────────────────────────────────
# 10. get_diagonal_entries for device caches
# ──────────────────────────────────────────────────────────────

@inline function Jutul.get_diagonal_entries(eq::Jutul.JutulEquation, eq_s::CompactAutoDiffCache_device)
    return get_entries(eq_s)
end

# ──────────────────────────────────────────────────────────────
# 11. apply_forces_to_equation! for PoissonSource on device
#     The diagonal entries `d` may be a SubArray of a GPU array
# ──────────────────────────────────────────────────────────────

function _is_gpu_backed(x::GPUArrays.AbstractGPUArray)
    return true
end

function _is_gpu_backed(x::SubArray)
    return _is_gpu_backed(parent(x))
end

function _is_gpu_backed(x)
    return false
end

function Jutul.apply_forces_to_equation!(d, storage, model::SimulationModel{<:Any, <:Any, <:Any, <:KernelAbstractionsContext}, eq::Jutul.AbstractPoissonEquation, eq_s, force::Vector{<:Jutul.PoissonSource}, time)
    # PoissonSource forces are small, apply via CPU round-trip
    d_cpu = Array(d)
    for f in force
        c = f.cell
        d_cpu[c] += f.value
    end
    copyto!(d, d_cpu)
end

# ──────────────────────────────────────────────────────────────
# 12. GPU-compatible convergence_criterion
#     Replaces @tullio max reduction with GPU-compatible ops
# ──────────────────────────────────────────────────────────────

function Jutul.convergence_criterion(model::SimulationModel{<:Any, <:Any, <:Any, <:KernelAbstractionsContext}, storage, eq::Jutul.JutulEquation, eq_s, r; dt = 1.0, update_report = missing)
    n = number_of_equations_per_entity(model, eq)
    r_abs = abs.(r)
    r_cpu = Array(r_abs)
    # Max over columns (entities) for each row (equation component)
    if n == 1
        e = [maximum(r_cpu)]
        names = "R"
    else
        e = vec(maximum(r_cpu, dims = 2))
        names = map(i -> "R_$i", 1:n)
    end
    R = (AbsMax = (errors = e, names = names), )
    return R
end

# ──────────────────────────────────────────────────────────────
# 13. GPU-compatible variable_change_report
# ──────────────────────────────────────────────────────────────

function Jutul.variable_change_report(X::GPUArrays.AbstractGPUArray, X0::GPUArrays.AbstractGPUArray{T}, pvar) where T<:Real
    # Copy to CPU for the report
    X_cpu = Array(X)
    X0_cpu = Array(X0)
    return Jutul.variable_change_report(X_cpu, X0_cpu, pvar)
end
