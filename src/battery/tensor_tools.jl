using Terv

##############
# Tensormaps #
##############
# TODO: These should all be computed initially, to avoid serach later
# TODO: Use standard way to loop through neigh. (as fill_jac_entries!)

function get_cell_index_vec(c, n, i, tbl)
    """ 
    Returns cni, the index of a vector defined on cells, with dependence
    on neighbouring cells. v[cni] is the i'th component of the field in
    cell c, dependent on cell n (for partial derivatives).
    """
    bool = map(x -> x.cell == c && x.cell_dep == n && x.vec == i, tbl)
    indx = findall(bool)
    @assert size(indx) == (1,) "Invalid or duplicate face cell combo, size = $(size(indx)) for (c, n, i) = ($c, $n, $i)"
    return indx[1] # cni
end

function get_face_index(f, c, conn_data)
    """
    Retuns i, b. i is the index of a vector defined on faces. v[i] is the
    value on face f. b is true if f is a face of the cell c, and false else
    """
    bool = map(x -> [x.face == f x.self == c], conn_data)
    indx = findall(x -> x[1] == 1, bool)
    @assert size(indx) == (2,) "Invalid or duplicate face cell combo, size $(size(indx)) for (f, c) = ($f, $c)"
    #! Very ugly, can this be done better?
    if bool[indx[1]][2]
        return indx[1], true
    elseif bool[indx[2]][2]
        return indx[2], true
    else   
        return indx[1][1], false
    end
end

function get_cell_index_scalar(c, n, tbl)
    """
    Returns cn, the index of a scalar defined on cells, which also depends
    on neighbouring cells. v[cn] is the value at cell c, dependent on c.
    """
    bool = map(x -> x.cell == c && x.cell_dep == n, tbl)
    indx = findall(bool)
    @assert size(indx) == (1,) "Invalid or duplicate face cell combo, size = $(size(indx)) for (c, n, i) = ($c, $n)"
    return indx[1]
end


# ! Boundary current is not included
function face_to_cell!(j_cell, J, c, P, ccv, conn_data)
    """
    Take a field defined on faces, J, finds its value at the cell c.
    j_cell is a vector that has 2 coponents per cell
    P maps between values defined on the face, and vectors defined in cells
    P_[c, f, i] = P_[2*(c-1) + i, f], (c=cell, i=space, f=face)
    j_c[c, c', i] = P_[c, f, i] * J_[f, c'] (c'=cell dependence)
    """

    # TODO: Combine the loop over neigh_self with the self
    cell_mask = map(x -> x.self==c, conn_data)
    neigh_self = conn_data[cell_mask] # ? is this the best way??
    for neigh in neigh_self
        f = neigh.face
        for i in 1:2 #! Only valid in 2D for now
            cic = get_cell_index_vec(c, c, i, ccv)
            fc, bool = get_face_index(f, c, conn_data)
            @assert bool
            ci = 2*(c-1) + i
            j_cell[cic] += P[ci, f] * J[fc]
        end
    end

    # what is the best order to loop through?
    for neigh in neigh_self
        n = neigh.other
        for neigh2 in neigh_self
            f = neigh2.face
            for i in 1:2
                cin = get_cell_index_vec(c, n, i, ccv)
                fn, bool = get_face_index(f, n, conn_data)

                # The value should only depend on cell n
                if bool
                    Jfn = J[fn]
                else
                    Jfn = value(J[fn])
                end

                j_cell[cin] += P[2*(c-1) + i, f] * Jfn
            end
        end
    end # the end is near
end

function vec_to_scalar(jsq, j, c, S, ccv, cctbl, conn_data)
    """
    Takes in vector valued field defined on the cell, and returns the
    modulus square
    jsq[c, c'] = S[c, 2*(c-1) + i] * j[c, c', i]^2
    """
    cell_mask = map(x -> x.self==c, conn_data)
    neigh_self = conn_data[cell_mask]
    cc = get_cell_index_scalar(c, c, cctbl)
    for i in 1:2
        cci = get_cell_index_vec(c, c, i, ccv)
        jsq[cc] += S[c, 2*(c-1) + i] * j[cci]^2
    end
    for neigh in neigh_self
        n = neigh.other
        cn = get_cell_index_scalar(c, n, cctbl)
        for i in 1:2
            cni = get_cell_index_vec(c, n, i, ccv)
            jsq[cn] += S[c, 2*(c-1) + i] * j[cni]^2
        end
    end 
end


# TODO: Use thses to find map to linear index 
function get_cellcellvec_map(neigh)
    """ Creates cellcellvectbl """
    dim = 2
    # Must have 2 copies of each neighbourship
    neigh = [
        [neigh[1, i] neigh[2, i]; neigh[2, i] neigh[1, i]] 
        for i in 1:size(neigh, 2)
        ]
    neigh = reduce(vcat, neigh)'
    cell1 = [repeat([i], dim) for i in neigh[1, :]]
    cell2 = [repeat([i], dim) for i in neigh[2, :]]
    num_neig = size(cell1)[1]
    vec = [1:dim for i in 1:num_neig]
    cell, cell_dep, vec = map(x -> reduce(vcat, x), [cell1, cell2, vec])
    # must add self dependence
    for i in 1:maximum(cell) #! Probably not the best way to find nc
        push!(cell, i); push!(cell, i)
        push!(cell_dep, i); push!(cell_dep, i)
        push!(vec, 1); push!(vec, 2)
    end
    # ? Should these be sorted in som way ?

    tbl = [
        (cell = cell[i], cell_dep = cell_dep[i], vec = vec[i]) 
            for i in 1:size(cell, 1)
        ]
    return tbl
end

function get_cellcell_map(neigh)
    """ Creates cellcelltbl """
    neigh = [
        [
            neigh[1, i] neigh[2, i]; 
            neigh[2, i] neigh[1, i]
        ] 
        for i in 1:size(neigh, 2)
        ]
    neigh = reduce(vcat, neigh)'
    cell1 = neigh[1, :]
    cell2 = neigh[2, :]
    num_neig = size(cell1)[1]
    cell, cell_dep = map(x -> reduce(vcat, x), [cell1, cell2])
    for i in 1:maximum(cell) #! Probably not the best way to find nc
        push!(cell, i);
        push!(cell_dep, i);
    end
    tbl = [
        (cell = cell[i], cell_dep = cell_dep[i])
        for i in 1:size(cell, 1)
        ]
    return tbl
end

