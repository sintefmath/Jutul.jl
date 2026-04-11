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
