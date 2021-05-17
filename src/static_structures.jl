export get_sparsity_pattern

#function get_incomp_matrix(G::MinimalTPFADomain)
#    n = number_of_cells(G)
#    get_incomp_matrix(n, G.conn_data)
#end

function get_incomp_matrix(n, hfd)
    # map is efficient on GPU 
    I = map(x -> x.self, hfd)
    J = map(x -> x.other, hfd)
    V = map(x -> x.T, hfd)
    d = zeros(n)
    for i in eachindex(I)
        d[I[i]] += V[i]
    end
    A = sparse(I, J, -V, n, n)
    A = A + spdiagm(d)
    return A
end

#function get_sparsity_pattern(G::MinimalTPFADomain, arg...)
#    n = number_of_cells(G)
#    get_sparsity_pattern(n, G.conn_data, arg...)
#end

function to_sparse(i, j, v, n, m)
    return sparse(i, j, v, n, m)
end

function to_sparse(i::CuArray, j::CuArray, v::CuArray, n, m)
    id = Array(i)
    jd = Array(j)
    vd = Array(j)
    A = sparse(id, jd, vd, n, m)
    A = CUDA.CUSPARSE.CuSparseMatrixCSC(A)
    #  CUDA.CUSPARSE.sparse(i, j, v, n, m)
end

function get_sparsity_pattern(n, hfd, nrows = 1, ncols = 1)
    # map is efficient on GPU
    hfd = Array(hfd)
    I = map(x -> x.self, hfd)
    J = map(x -> x.other, hfd)
    V = map(x -> x.T, hfd)

    D = [i for i in 1:n]

    I = vcat(I, D)
    J = vcat(J, D)
    V = vcat(V, ones(n))

    if nrows > 1
        I = vcat(map((x) -> (x-1)*n .+ I, 1:nrows)...)
        J = repeat(J, nrows)
        V = repeat(V, nrows)
    end
    if ncols > 1
        I = repeat(I, ncols)
        J = vcat(map((x) -> (x-1)*n .+ J, 1:ncols)...)
        V = repeat(V, ncols)
    end
    A = to_sparse(I, J, V, n*nrows, n*ncols)
    return A
end

function get_sparsity_pattern(n, hfd::CuArray)
    # Do this on CPU
    tmp = Array(hfd)
    return get_sparsity_pattern(n, tmp)
end