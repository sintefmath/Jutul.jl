module JutulHypreExt
    using Jutul, HYPRE, SparseArrays

    function Jutul.setup_hypre_precond(type = :boomeramg; kwarg...)
        @assert type == :boomeramg
        HYPRE.Init()
        prec = HYPRE.BoomerAMG(; Tol = 0.0, MaxIter = 1, kwarg...)
        return prec
    end

    function Jutul.update!(preconditioner::BoomerAMGPreconditioner, J, r, ctx)
        n, m = size(J)
        @assert n == m
        D = preconditioner.data
        nzval = nonzeros(J)
        rows = rowvals(J)
        if haskey(D, :J) && D[:J] == J
            J_h, r_h, x_h = D[:converted]
            if true
                I_buf, J_buf, V_buffers = D[:asm_buffers]
                reassemble_internal_boomeramg!(I_buf, J_buf, V_buffers, nzval, J, J_h, rows)
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
                V[i] = zeros(Float64, 1, i)
            end
            D[:asm_buffers] = (I = zeros(Int, 1), J = zeros(Int, max_width), V = V)
            r_h = HYPRE.HYPREVector(r)
            x_h = HYPRE.HYPREVector(copy(r))
            J_h = HYPRE.HYPREMatrix(J)
            D[:J] = J
            D[:n] = n
            (; ilower, iupper) = r_h
            D[:vector_indices] = HYPRE.HYPRE_BigInt.(ilower:iupper)
        end
        D[:converted] = (J_h, r_h, x_h)
        HYPRE.@check HYPRE.HYPRE_BoomerAMGSetup(preconditioner.prec, J_h, r_h, x_h)
        return preconditioner
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

    function reassemble_internal_boomeramg!(I_buf, J_buf, V_buffers, nzval, J, J_h, rows)
        n = size(J, 1)
        @assert length(I_buf) == 1
        assembler = HYPRE.start_assemble!(J_h)
        @inbounds for row in 1:n
            row_ix = nzrange(J, row)
            k = length(row_ix)
            ROWS = I_buf
            ROWS[1] = row
            COLS = J_buf
            V_buf = V_buffers[k]
            resize!(COLS, k)
            @inbounds for ki in 1:k
                ri = row_ix[ki]
                V_buf[ki] = nzval[ri]
                COLS[ki] = rows[ri]
            end
            HYPRE.assemble!(assembler, ROWS, COLS, V_buf)
        end
        HYPRE.finish_assemble!(assembler)
    end
end
