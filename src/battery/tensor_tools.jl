using Terv

export vec_to_scalar!, face_to_cell!, get_cellcell_map, get_cellcellvec_map
export get_neigh

################
# Helper funcs #
################

@inline function get_neigh(c, model)
    """
    Retruns a vector of NamedTuples of the form
    (T = a, face = b, self = c, other = d)
    of all neighbours of cell c.
    """
    mf = model.domain.discretizations.charge_flow
    conn_pos = mf.conn_pos
    conn_data = mf.conn_data
    indx = conn_pos[c]:(conn_pos[c + 1] - 1)
    return conn_data[indx]
end


##############
# Tensormaps #
##############
# TODO: These should all be computed initially, to avoid serach later


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
function face_to_cell!(j_cell, J, c, model)
    """
    Take a field defined on faces, J, finds its value at the cell c.
    j_cell is a vector that has 2 coponents per cell
    P maps between values defined on the face, and vectors defined in cells
    P_[c, f, i] = P_[2*(c-1) + i, f], (c=cell, i=space, f=face)
    j_c[c, c', i] = P_[c, f, i] * J_[f, c'] (c'=cell dependence)
    """

    P = model.domain.grid.P
    mf = model.domain.discretizations.charge_flow
    cfcv_tbl = mf.cellfacecellvec.tbl
    cfcv_pos = mf.cellfacecellvec.pos
    conn_data = mf.conn_data

    neigh_c = get_neigh(c, model)

    for neigh in neigh_c
        f = neigh.face
        for i in 1:2 #! Only valid in 2D for now
            cic = get_cell_index_vec(c, c, i, ccv_tbl)
            fc, bool = get_face_index(f, c, conn_data)
            @assert bool
            ci = 2*(c-1) + i
            j_cell[cic] += P[ci, f] * J[fc]
        end

        for neigh2 in neigh_c
            n = neigh2.other
            for i in 1:2
                cin = get_cell_index_vec(c, n, i, ccv_tbl)
                fn, bool = get_face_index(f, n, conn_data)

                # The value should only depend on cell n
                if bool
                    Jfn = J[fn]
                else
                    Jfn = value(J[fn])
                end

                j_cell[cin] += P[2*(c-1) + i, f] * Jfn
            end
        end # the end is near
    end

end

function vec_to_scalar!(jsq, j, c, model)
    """
    Takes in vector valued field defined on the cell, and returns the
    modulus square
    jsq[c, c'] = S[c, 2*(c-1) + i] * j[c, c', i]^2
    """
    S = model.domain.grid.S
    cctbl = model.domain.discretizations.charge_flow.cellcellvec.tbl
    ccv_tbl = model.domain.discretizations.charge_flow.cellcellvec.tbl

    cc = get_cell_index_scalar(c, c, cctbl)
    for i in 1:2
        cci = get_cell_index_vec(c, c, i, ccv_tbl)
        jsq[cc] += S[c, 2*(c-1) + i] * j[cci]^2
    end
    for neigh in get_neigh(c, model)
        n = neigh.other
        cn = get_cell_index_scalar(c, n, cctbl)
        for i in 1:2
            cni = get_cell_index_vec(c, n, i, ccv_tbl)
            jsq[cn] += S[c, 2*(c-1) + i] * j[cni]^2
        end
    end 
end


# TODO: Use thses to find map to linear index 
function get_cellfacecellvec_tbl(neigh, face_pos, conn_data, faces)
    """ Creates cellcellvectbl """
    dim = 2
    nc = maximum(neigh) #! Probably not the best way to find nc
    # ! This is now genereated 2 times, as conn_data is in disc. (Why?)

    # Must have 2 copies of each neighbourship

    neigh = [
        [neigh[1, i] neigh[2, i]; neigh[2, i] neigh[1, i]] 
        for i in 1:size(neigh, 2)
        ]
    neigh = reduce(vcat, neigh)

    cell1 = neigh[:, 1]
    cell2 = neigh[:, 2]
    print(cell1)
    indx = sortperm(cell1)
    cell1, cell2 = [cell1[indx], cell2[indx]]
    
    d = (diff(cell1) .== 1)
    c_pos = findall(vcat(true, d, true))

    cell = []
    cell_dep = []
    face =  []
    for c in 1:nc
        for fp in face_pos[c]:(face_pos[c+1]-1)
            f = faces[fp]
            push!(cell, c)
            push!(cell_dep, c)
            push!(face, f)

            for np in c_pos[c]:(c_pos[c+1]-1)
                n = cell2[np]
                # print(c,  ", ", fp,", ",np, "\n")
                push!(cell, c)
                push!(cell_dep, n)
                push!(face, f)

            end
        end
    end
    cell, face, cell_dep = map(x -> reduce(vcat, x), [cell, face, cell_dep])

    n = size(cell, 1)
    vec = [1:dim for i in 1:n]
    cell_dep, cell, face = map(
        x -> [repeat([a], dim) for a in x],
        [cell_dep, cell, face]
        )
    cell, face, cell_dep, vec = map(x -> reduce(vcat, x), [cell, face, cell_dep, vec])


    d = (diff(cell) .== 1)
    ccv_pos = findall(vcat(true, d, true))

    tbl = [
        (cell = cell[i], face = face[i], cell_dep = cell_dep[i], vec = vec[i]) 
            for i in 1:size(cell, 1)
        ]
    return (tbl=tbl, pos=ccv_pos)
end

function get_cellcellvec_tbl(neigh, face_pos, conn_data)
    """ Creates cellcelltbl """
    dim = 2
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

    cell, cell_dep = map(x -> reduce(vcat, x), [cell1, cell2])
    for i in 1:maximum(cell) #! Probably not the best way to find nc
        push!(cell, i);
        push!(cell_dep, i);
    end

    n = size(cell, 1)
    vec = [1:dim for i in 1:n]
    cell_dep, cell = map(
        x -> [repeat([a], dim) for a in x],
        [cell_dep, cell]
        )
    cell, cell_dep, vec = map(x -> reduce(vcat, x), [cell, cell_dep, vec])

    tbl = [
        (cell = cell[i], cell_dep = cell_dep[i], vec = vec[i])
        for i in 1:size(cell, 1)
        ]
    return (tbl=tbl,)
end

