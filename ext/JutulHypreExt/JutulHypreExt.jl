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
            D[:indices] = collect(1:n)
        end
        D[:converted] = (J_h, r_h, x_h)
        HYPRE.@check HYPRE.HYPRE_BoomerAMGSetup(preconditioner.prec, J_h, r_h, x_h)
        return preconditioner
    end

    function Jutul.operator_nrows(p::BoomerAMGPreconditioner)
        return p.data[:n]
    end

    function Jutul.apply!(x, p::BoomerAMGPreconditioner, y, t)
        ix = p.data[:indices]
        J_h, y_h, x_h = p.data[:converted]
        asm = HYPRE.start_assemble!(x_h)
        HYPRE.finish_assemble!(asm)
        assembler = HYPRE.start_assemble!(y_h)
        HYPRE.assemble!(assembler, ix, y)
        HYPRE.finish_assemble!(assembler)
        # HYPRE.@check HYPRE.HYPRE_BoomerAMGSetup(p.prec, J_h, y_h, x_h)
        HYPRE.@check HYPRE.HYPRE_BoomerAMGSolve(p.prec, J_h, y_h, x_h)
        copy!(x, x_h)
    end

    function reassemble_internal_boomeramg!(I_buf, J_buf, V_buffers, nzval, J, J_h, rows)
        n = size(J, 1)
        @assert length(I_buf) == 1
        assembler = HYPRE.start_assemble!(J_h)
        for row in 1:n
            row_ix = nzrange(J, row)
            k = length(row_ix)
            ROWS = I_buf
            @inbounds ROWS[1] = row
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
