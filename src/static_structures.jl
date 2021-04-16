export TPFAHalfFaceData

# Helpers follow
struct TPFAHalfFaceData{R<:Real,I<:Integer}
    T::R
    dz::R
    self::I
    other::I
end

struct HalfFaceData{R<:Real,I<:Integer}
    T::R
    self::I
    other::I
end

function get_sparsity(G::MRSTSimGraph)
    n = G.ncells
    e = ones(length(G.cells));
    A = SparseMatrixCSC(n, n, G.facePos, G.cells, e)
    A = A + sparse(1.0I, n, n)
    return A
end

function get_incomp_matrix(G::MRSTSimGraph)
    # Grab TPFA matrix
    n = G.ncells
    hfd = G.HalfFaceData
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

function get_incomp_matrix(G::MinimalTPFAGrid)
    # Grab TPFA matrix
    n = G.ncells
    hfd = G.TPFAHalfFaceData
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

