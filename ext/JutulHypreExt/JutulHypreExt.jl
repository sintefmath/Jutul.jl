module JutulHypreExt
    using Jutul, HYPRE, SparseArrays

    function Jutul.setup_hypre_precond(type = :boomeramg; kwarg...)
        @assert type == :boomeramg
        HYPRE.Init()
        prec = HYPRE.BoomerAMG(; PrintLevel = 0, Tol = 0.0, MaxIter = 1, kwarg...)
        return prec
    end

    function Jutul.update_preconditioner!(preconditioner::BoomerAMGPreconditioner, J, r, ctx, executor)
        n, m = size(J)
        @assert n == m
        D = preconditioner.data
        # nzval = nonzeros(J)
        # rows = rowvals(J)
        if haskey(D, :J) && D[:J] === J
            J_h, r_h, x_h = D[:converted]
            if true
                reassemble_matrix!(J_h, D, J)
            else
                J_h = HYPRE.HYPREMatrix(J)
            end
        else
            max_width = 0
            min_width = 1_000_000
            for i in 1:n
                nz_width = length(nzrange(J, i))
                max_width = max(max_width, nz_width)
                min_width = min(min_width, nz_width)
            end
            V = Dict{Int, Matrix{Float64}}()
            for i in min_width:max_width
                if J isa SparseMatrixCSC
                    V[i] = zeros(Float64, i, 1)
                else
                    V[i] = zeros(Float64, 1, i)
                end
            end
            D[:asm_buffers] = (I = zeros(Int, 1), J = zeros(Int, max_width), V = V)
            r_h = HYPRE.HYPREVector(r)
            x_h = HYPRE.HYPREVector(copy(r))
            D[:J] = J
            D[:n] = n
            (; ilower, iupper) = r_h
            D[:vector_indices] = HYPRE.HYPRE_BigInt.(ilower:iupper)
            J_h = transfer_matrix_to_hypre!(J, D)
        end
        D[:converted] = (J_h, r_h, x_h)
        HYPRE.@check HYPRE.HYPRE_BoomerAMGSetup(preconditioner.prec, J_h, r_h, x_h)
        return preconditioner
    end

    function transfer_matrix_to_hypre!(J, D)
        return HYPRE.HYPREMatrix(J)
    end

    function transfer_matrix_to_hypre!(J::Jutul.StaticSparsityMatrixCSR, D)
        J_h = HYPRE.HYPREMatrix(J.At)
        reassemble_matrix!(J_h, D, J)
        return J_h
    end

    function reassemble_matrix!(J_h, D, J)
        I_buf, J_buf, V_buffers = D[:asm_buffers]
        reassemble_internal_boomeramg!(I_buf, J_buf, V_buffers, J, J_h)
    end

    function Jutul.operator_nrows(p::BoomerAMGPreconditioner)
        return p.data[:n]
    end

    function Jutul.apply!(x, p::BoomerAMGPreconditioner, y, arg...)
        ix = p.data[:vector_indices]
        J_h, y_h, x_h = p.data[:converted]
        inner_apply!(x, y, p.prec, x_h, J_h, y_h, ix)
    end

    function inner_apply!(x, y, prec, x_h, J_h, y_h, ix)
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
        @. x = 0.0
        local_copy!(x_h, x, ix)
        HYPRE.@check HYPRE.HYPRE_BoomerAMGSolve(prec, J_h, y_h, x_h)
        local_copy!(x, x_h, ix)
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

    function reassemble_internal_boomeramg!(single_buf, longer_buf, V_buffers, Jac::SparseMatrixCSC, J_h)
        nzval = nonzeros(Jac)
        rows = rowvals(Jac)

        n = size(Jac, 2)
        @assert length(single_buf) == 1
        assembler = HYPRE.start_assemble!(J_h)
        @inbounds for col in 1:n
            pos_ix = nzrange(Jac, col)
            k = length(pos_ix)
            J = single_buf
            J[1] = col
            I = longer_buf
            V_buf = V_buffers[k]
            resize!(I, k)
            @inbounds for ki in 1:k
                ri = pos_ix[ki]
                V_buf[ki] = nzval[ri]
                I[ki] = rows[ri]
            end
            HYPRE.assemble!(assembler, I, J, V_buf)
        end
        HYPRE.finish_assemble!(assembler)
    end

    function reassemble_internal_boomeramg!(single_buf, longer_buf, V_buffers, Jac::Jutul.StaticSparsityMatrixCSR, J_h)
        nzval = nonzeros(Jac)
        cols = Jutul.colvals(Jac)

        n = size(Jac, 1)
        @assert length(single_buf) == 1
        assembler = HYPRE.start_assemble!(J_h)
        @inbounds for row in 1:n
            pos_ix = nzrange(Jac, row)
            k = length(pos_ix)
            I = single_buf
            I[1] = row
            J = longer_buf
            resize!(J, k)
            V_buf = V_buffers[k]
            @inbounds for ki in 1:k
                ri = pos_ix[ki]
                V_buf[ki] = nzval[ri]
                J[ki] = cols[ri]
            end
            HYPRE.assemble!(assembler, I, J, V_buf)
        end
        HYPRE.finish_assemble!(assembler)
    end
end
