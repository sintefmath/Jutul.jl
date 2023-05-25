module JutulHypreExt
    using Jutul, HYPRE, SparseArrays

    function Jutul.setup_hypre_precond(type = :boomeramg; kwarg...)
        @assert type == :boomeramg
        HYPRE.Init()
        prec = HYPRE.BoomerAMG(; PrintLevel = 0, Tol = 0.0, MaxIter = 1, kwarg...)
        return prec
    end

    function Jutul.generate_hypre_assembly_helper(J::AbstractSparseMatrix, executor, ilower = 1, iupper = size(J, 1); column_major = isa(J, SparseMatrixCSC))
        max_width = 0
        min_width = 1_000_000
        n = iupper - ilower + 1
        for i in 1:n
            nz_width = length(nzrange(J, i))
            max_width = max(max_width, nz_width)
            min_width = min(min_width, nz_width)
        end
        V = Dict{Int, Matrix{Float64}}()
        for i in min_width:max_width
            if column_major
                V[i] = zeros(Float64, i, 1)
            else
                V[i] = zeros(Float64, 1, i)
            end
        end
        return (
            I = zeros(Int, 1),
            J = zeros(Int, max_width),
            V = V, indices = HYPRE.HYPRE_BigInt.(ilower:iupper),
            native_zeroed_buffer = zeros(n),
            n = n
            )
    end

    function Jutul.update_preconditioner!(preconditioner::BoomerAMGPreconditioner, J, r, ctx, executor)
        D = preconditioner.data

        if !haskey(D, :assembly_helper)
            D[:assembly_helper] = Jutul.generate_hypre_assembly_helper(J, executor)
        end

        if haskey(D, :hypre_system)
            J_h, r_h, x_h = D[:hypre_system]
            reassemble_matrix!(J_h, D, J, executor)
        else
            r_h = HYPRE.HYPREVector(r)
            x_h = HYPRE.HYPREVector(copy(r))
            D[:J] = J
            # D[:n] = n
            # (; ilower, iupper) = r_h
            # D[:vector_indices] = HYPRE.HYPRE_BigInt.(ilower:iupper)
            J_h = transfer_matrix_to_hypre(J, D, executor)
        end
        D[:hypre_system] = (J_h, r_h, x_h)
        HYPRE.@check HYPRE.HYPRE_BoomerAMGSetup(preconditioner.prec, J_h, r_h, x_h)
        return preconditioner
    end

    function transfer_vector_to_hypre(r, D, executor)
        return HYPRE.HYPREVector(r)
    end

    function transfer_matrix_to_hypre(J, D, executor)
        n, m = size(J)
        @assert n == m
        return HYPRE.HYPREMatrix(J)
    end

    function transfer_matrix_to_hypre(J::Jutul.StaticSparsityMatrixCSR, D, executor)
        n, m = size(J)
        @assert n == m
        J_h = HYPRE.HYPREMatrix(J.At)
        reassemble_matrix!(J_h, D, J, executor)
        return J_h
    end

    function transfer_matrix_to_hypre!(J::HYPRE.HYPREMatrix, D, executor)
        return J
    end

    function reassemble_matrix!(J_h::HYPRE.HYPREMatrix, D, J::HYPRE.HYPREMatrix, executor)
        # Already HYPRE system, assembled
        @assert J_h === J
    end

    function reassemble_matrix!(J_h, D, J, executor)
        I_buf, J_buf, V_buffers = D[:assembly_helper]
        reassemble_internal_boomeramg!(I_buf, J_buf, V_buffers, J, J_h, executor)
    end

    function Jutul.operator_nrows(p::BoomerAMGPreconditioner)
        return p.data[:n]
    end

    function Jutul.apply!(x, p::BoomerAMGPreconditioner, y, arg...)
        ix = p.data[:assembly_helper].indices
        zbuf = p.data[:assembly_helper].native_zeroed_buffer
        J_h, y_h, x_h = p.data[:hypre_system]
        inner_apply!(x, y, p.prec, x_h, J_h, y_h, ix, zbuf)
    end

    function inner_apply!(x, y, prec, x_h, J_h, y_h, ix, zbuf)
        # Safe and allocating version that uses HYPRE.jl API
        # asm = HYPRE.start_assemble!(x_h)
        # HYPRE.finish_assemble!(asm)
        # assembler = HYPRE.start_assemble!(y_h)
        # HYPRE.assemble!(assembler, ix, y)
        # HYPRE.finish_assemble!(assembler)
        # HYPRE.@check HYPRE.HYPRE_BoomerAMGSolve(prec, J_h, y_h, x_h)
        # copy!(x, x_h)

        # Fast and less allocating version that uses low level HYPRE calls
        local_copy!(y_h, y, ix)
        local_copy!(x_h, zbuf, ix)
        HYPRE.@check HYPRE.HYPRE_BoomerAMGSolve(prec, J_h, y_h, x_h)
        local_copy!(x, x_h, ix)
    end

    function inner_apply!(x::T, y::T, prec, x_h::T, J_h, y_h::T, ix, zbuf) where T<:HYPRE.HYPREVector
        local_copy!(x, zbuf, ix)
        HYPRE.@check HYPRE.HYPRE_BoomerAMGSolve(prec, J_h, y, x)
    end

    function hypre_check(dst::HYPRE.HYPREVector, src::Vector, ix)
        @assert dst.parvector != C_NULL
        nvalues = length(src)
        @assert length(ix) == nvalues
        return nvalues
    end

    function local_copy!(dst::Vector{HYPRE.HYPRE_Complex}, src::HYPRE.HYPREVector, ix::Vector{HYPRE.HYPRE_BigInt})
        nvalues = hypre_check(src, dst, ix)
        HYPRE.@check HYPRE.HYPRE_IJVectorGetValues(src, nvalues, ix, dst)
    end

    function local_copy!(dst::HYPRE.HYPREVector, src::Vector{HYPRE.HYPRE_Complex}, ix::Vector{HYPRE.HYPRE_BigInt})
        nvalues = hypre_check(dst, src, ix)
        HYPRE.@check HYPRE.HYPRE_IJVectorSetValues(dst, nvalues, ix, src)
        HYPRE.Internals.assemble_vector(dst)
    end

    function reassemble_internal_boomeramg!(single_buf, longer_buf, V_buffers, Jac::SparseMatrixCSC, J_h, executor)
        nzval = nonzeros(Jac)
        rows = rowvals(Jac)

        n = size(Jac, 2)
        @assert length(single_buf) == 1
        assembler = HYPRE.start_assemble!(J_h)
        @inbounds for col in 1:n
            pos_ix = nzrange(Jac, col)
            k = length(pos_ix)
            J = single_buf
            J[1] = Jutul.executor_index_to_global(executor, col, :column)
            I = longer_buf
            V_buf = V_buffers[k]
            resize!(I, k)
            @inbounds for ki in 1:k
                ri = pos_ix[ki]
                V_buf[ki] = nzval[ri]
                I[ki] = Jutul.executor_index_to_global(executor, rows[ri], :row)
            end
            HYPRE.assemble!(assembler, I, J, V_buf)
        end
        HYPRE.finish_assemble!(assembler)
    end

    function reassemble_internal_boomeramg!(single_buf, longer_buf, V_buffers, Jac::Jutul.StaticSparsityMatrixCSR, J_h, executor)
        nzval = nonzeros(Jac)
        cols = Jutul.colvals(Jac)

        n = size(Jac, 1)
        @assert length(single_buf) == 1
        (; iupper, ilower) = J_h
        @assert n == iupper - ilower + 1
        assembler = HYPRE.start_assemble!(J_h)
        @inbounds for row in 1:n
            pos_ix = nzrange(Jac, row)
            k = length(pos_ix)
            I = single_buf
            I[1] = Jutul.executor_index_to_global(executor, row, :row)
            J = longer_buf
            resize!(J, k)
            V_buf = V_buffers[k]
            @inbounds for ki in 1:k
                ri = pos_ix[ki]
                V_buf[ki] = nzval[ri]
                J[ki] = Jutul.executor_index_to_global(executor, cols[ri], :column)
            end
            HYPRE.assemble!(assembler, I, J, V_buf)
        end
        HYPRE.finish_assemble!(assembler)
    end
end
