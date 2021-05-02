

#function get_sparsity(G::MRSTSimGraph)
#    n = G.ncells
#    e = ones(length(G.cells));
#    A = SparseMatrixCSC(n, n, G.facePos, G.cells, e)
#    A = A + sparse(1.0I, n, n)
#    return A
#end

function get_incomp_matrix(G::MinimalTPFAGrid)
    n = number_of_cells(G)
    get_incomp_matrix(n, G.conn_data)
end

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

function get_sparsity_pattern(G::MinimalTPFAGrid)
    n = number_of_cells(G)
    get_sparsity_pattern(n, G.conn_data)
end

function get_sparsity_pattern(n, hfd)
    # map is efficient on GPU 
    I = map(x -> x.self, hfd)
    J = map(x -> x.other, hfd)
    V = map(x -> x.T, hfd)

    D = [i for i in 1:n]

    I = vcat(I, D)
    J = vcat(J, D)
    V = vcat(V, ones(n))

    A = sparse(I, J, V, n, n)
    return A
end

function get_sparsity_pattern(n, hfd::CuArray)
    # Do this on CPU
    tmp = Array(hfd)
    return get_sparsity_pattern(n, tmp)
end