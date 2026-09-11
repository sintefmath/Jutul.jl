function same_pattern(H::AMGHierarchy, A::StaticSparsityMatrixCSR)
    matrix_nrows(A) == size(H, 1) || return false
    matrix_ncols(A) == size(H, 2) || return false
    matrix_nonzeros(A) == length(H.pattern_colval) || return false
    if A.rowptr isa Array && A.colval isa Array
        @inbounds for i in 1:(matrix_nrows(A)+1)
            A.rowptr[i] == H.pattern_rowptr[i] || return false
        end
        @inbounds for k in 1:matrix_nonzeros(A)
            A.colval[k] == H.pattern_colval[k] || return false
        end
        return true
    end
    # Device comparison would force a host synchronization. Length and dimensions
    # are checked here; callers owning mutable device patterns should use :memory.
    true
end

function galerkin!(coarse::StaticSparsityMatrixCSR, fine::StaticSparsityMatrixCSR, P::Prolongation, G::GalerkinMap)
    k! = galerkin_kernel!(matrix_backend(fine), matrix_block_size(fine))
    k!(coarse.nzval, fine.nzval, P.nzval, G.offsets, G.p_left, G.a_index,
       G.p_right, matrix_nonzeros(coarse); ndrange=matrix_nonzeros(coarse))
    coarse
end

function galerkin!(coarse::StaticSparsityMatrixCSR{Tv,Ti,<:Vector,<:Vector,<:Vector},
                    fine::StaticSparsityMatrixCSR{Tv,Ti,<:Vector,<:Vector,<:Vector},
                    P::Prolongation, G::GalerkinMap) where {Tv,Ti}
    foreach_cpu_row(matrix_nonzeros(coarse)) do k
        value = zero(Tv)
        @inbounds @simd for t in G.offsets[k]:(G.offsets[k+1]-1)
            value += conj(P.nzval[G.p_left[t]]) * fine.nzval[G.a_index[t]] *
                     P.nzval[G.p_right[t]]
        end
        @inbounds coarse.nzval[k] = value
    end
    coarse
end

function update_prolongation!(level::AMGLevel)
    A, P = level.A, level.P
    isnothing(P) && return level
    k! = update_p_kernel!(matrix_backend(A), matrix_block_size(A))
    k!(P.nzval, P.rowptr, P.colval, A.rowptr, A.colval, A.nzval,
       level.cf, level.coarse_map, level.strength, matrix_nrows(A); ndrange=matrix_nrows(A))
    level
end

function numeric_reset!(H::AMGHierarchy, A::StaticSparsityMatrixCSR, update_p::Bool)
    copy_matrix_values!(H.levels[1].A, A)
    update_level_smoother!(H.levels[1].smoother, H.levels[1].A, H.options)
    for l in 1:(length(H.levels)-1)
        level = H.levels[l]
        update_p && !(H.options.coarsening isa Aggregation) && update_prolongation!(level)
        next = H.levels[l+1]
        galerkin!(next.A, level.A, level.P, level.galerkin)
        update_level_smoother!(next.smoother, next.A, H.options)
    end
    coarse = H.levels[end]
    isnothing(coarse.coarse_solver) || update_coarse_solver!(coarse.coarse_solver, coarse.A)
    H.last_iterations = 0
    H.last_residual = Inf
    synchronize_backend(H.backend)
    H
end

function replace_hierarchy!(H::AMGHierarchy, fresh::AMGHierarchy)
    H.levels = fresh.levels
    H.workspace = fresh.workspace
    H.options = fresh.options
    H.backend = fresh.backend
    H.block_size = fresh.block_size
    H.pattern_rowptr = fresh.pattern_rowptr
    H.pattern_colval = fresh.pattern_colval
    H.last_iterations = 0
    H.last_residual = Inf
    H
end

function rebuild_memory!(H::AMGHierarchy, host_finest::StaticSparsityMatrixCSR)
    # The finest graph is fixed by the discretization. Rebuild all strength,
    # splitting, interpolation, and coarse symbolic data, but retain level 1's
    # structural arrays and use the supplied host values for symbolic setup.
    finest = H.levels[1].A
    H.levels = build_hierarchy(finest, H.options; reuse_levels=H.levels,
                                workspace=H.workspace, host_finest=host_finest,
                                reuse_finest_structure=true)
    H.block_size = H.options.block_size
    H.last_iterations = 0
    H.last_residual = Inf
    H
end

function memory_reset!(H::AMGHierarchy{Tv,Ti}, A::StaticSparsityMatrixCSR) where {Tv,Ti}
    same_backend(matrix_backend(A), H.backend) ||
        throw(ArgumentError("matrix and hierarchy must use the same backend"))
    size(A) == size(H) || throw(DimensionMismatch("matrix and hierarchy sizes differ"))
    matrix_nonzeros(A) == matrix_nonzeros(H.levels[1].A) ||
        throw(ArgumentError("reuse=:memory requires an unchanged finest-level sparsity pattern"))
    finest = H.levels[1].A
    finest.nzval === A.nzval || copy_matrix_values!(finest, A)
    if H.backend isa KernelAbstractions.CPU
        host_finest = finest
    else
        stage = H.workspace.stage_matrix1
        host_values = if stage isa StaticSparsityMatrixCSR{Tv,Ti,<:Vector,<:Vector,<:Vector}
            stage.nzval
        else
            Tv[]
        end
        resize!(host_values, matrix_nonzeros(A))
        copyto!(host_values, 1, A.nzval, 1, matrix_nonzeros(A))
        host_finest = csr_matrix(
            H.pattern_rowptr, H.pattern_colval, host_values,
            matrix_nrows(A), matrix_ncols(A); block_size = H.block_size)
    end
    rebuild_memory!(H, host_finest)
end

"""
    resetup_amg!(H, A, reuse=:operators)

Refresh a hierarchy for new coefficients.

* `:operators` preserves C/F splits, prolongation values, sparse patterns, and
  every array; only Galerkin products and SPAI(0) entries are recomputed.
* `:sparsity` additionally recomputes Extended+i weights in place.
* `:memory` recomputes all symbolic data while recycling compatible hierarchy
  buffers. The finest sparsity pattern must be unchanged; coefficients may alter
  strength, splitting, interpolation, and every coarse pattern.
* `:none` is a completely fresh setup and is useful as a reference baseline.

The first two modes perform no hierarchy-array allocation.
"""
function resetup_amg!(H::AMGHierarchy, A::StaticSparsityMatrixCSR, reuse::Symbol=:operators)
    reuse in (:memory, :sparsity, :operators, :none) ||
        throw(ArgumentError("reuse must be :memory, :sparsity, :operators, or :none"))
    if reuse in (:operators, :sparsity)
        same_pattern(H, A) || throw(ArgumentError("reuse=$reuse requires an unchanged CSR pattern"))
        same_backend(matrix_backend(A), H.backend) ||
            throw(ArgumentError("matrix and hierarchy must use the same backend"))
        return numeric_reset!(H, A, reuse == :sparsity)
    end
    reuse == :memory && return memory_reset!(H, A)
    replace_hierarchy!(H, setup_amg(A, H.options))
end

function resetup_amg!(H::AMGHierarchy{Tv,Ti}, A::SparseMatrixCSC,
                      reuse::Symbol=:operators) where {Tv,Ti}
    eltype(A) === Tv || throw(ArgumentError("matrix value type must match the hierarchy"))
    if reuse == :memory
        size(A) == size(H) || throw(DimensionMismatch("matrix and hierarchy sizes differ"))
        nnz(A) == matrix_nonzeros(H.levels[1].A) ||
            throw(ArgumentError("reuse=:memory requires an unchanged finest-level sparsity pattern"))
        finest = H.levels[1].A
        if H.backend isa KernelAbstractions.CPU
            host_values = finest.nzval
        else
            stage = H.workspace.stage_matrix1
            host_values = if stage isa StaticSparsityMatrixCSR{Tv,Ti,<:Vector,<:Vector,<:Vector}
                stage.nzval
            else
                Tv[]
            end
        end
        copy_csc_values_to_csr!(host_values, A, H.pattern_rowptr,
                                 H.workspace.ti1)
        H.backend isa KernelAbstractions.CPU ||
            copyto!(finest.nzval, 1, host_values, 1, nnz(A))
        host_finest = csr_matrix(
            H.pattern_rowptr, H.pattern_colval, host_values,
            size(A, 1), size(A, 2); block_size = H.block_size)
        return rebuild_memory!(H, host_finest)
    end
    C = csr_matrix(A; backend=H.backend, block_size=H.block_size, index_type=Ti)
    resetup_amg!(H, C, reuse)
end

resetup_amg!(H::AMGHierarchy, A; reuse::Symbol=:operators) = resetup_amg!(H, A, reuse)
