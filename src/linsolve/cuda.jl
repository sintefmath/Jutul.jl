function build_jacobian(sparse_arg, context::SingleCUDAContext, layout)
    @assert sparse_arg.layout == layout
    I, J, n, m = ijnm(sparse_arg)
    bz = block_size(sparse_arg)
    Jt = jacobian_eltype(context, layout, bz)
    Ft = float_type(context)
    It = index_type(context)

    V = zeros(Jt, length(I))
    jac_cpu = sparse(It.(I), It.(J), V, n, m)
    jac = CUDA.CUSPARSE.CuSparseMatrixCSC{Ft}(jac_cpu)
    # A = CUDA.CUSPARSE.CuSparseMatrixBSR{Float64}(sparse(rand(6, 6)), 2)
    nzval = nonzeros(jac)
    if Ft == Jt
        V_buf = nzval
    else
        V_buf = reinterpret(reshape, Ft, nzval)
    end
    return (jac, V_buf, bz)
end

function get_jacobian_vector(n, context::SingleCUDAContext, layout, v = nothing, bz = 1)
    Ft = float_type(context)
    It = index_type(context)
    @assert isnothing(v)
    @assert bz == 1
    v = CuArray(zeros(Ft, n))
    return (v, v)
end

#
function transfer(context::SingleCUDAContext, lsys::LinearizedSystem)
    F_t = float_type(context)
    I_t = index_type(context)
    
    # I, J, V, n, m = sparse_arg

    A = lsys.jac
    n = size(A, 1)

    # A = sparse(I_t.(I), I_t.(J), F_t.(V), I_t(n), I_t(m))
    jac = CUDA.CUSPARSE.CuSparseMatrixCSC{F_t}(A)

    V_buf = nonzeros(jac)

    r = CuArray{F_t}(undef, n)
    dx = CuArray{F_t}(undef, n)

    r_buf = r
    dx_buf = dx
    return LinearizedSystem(jac, r, dx, V_buf, r_buf, dx_buf, lsys.matrix_layout)
end

