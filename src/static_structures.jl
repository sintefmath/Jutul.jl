

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
    I = [x.self for x in hfd]
    J = [x.other for x in hfd]
    V = [x.T for x in hfd]

    d = zeros(n)
    for i in eachindex(I)
        d[I[i]] += V[i]
    end
    A = sparse(I, J, -V, n, n)
    A = A + spdiagm(d)
    return A
end