
export vec_to_scalar!, face_to_cell!, get_cellcell_map, get_cellcellvec_map
export get_cfcv2ccv_map, get_cfcv2cc_map, get_cfcv2fc_map
export get_neigh, get_ccv_index, get_cc_index

################
# Helper funcs #
################

function get_neigh(c, model)
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
    cfcv = mf.cellfacecellvec
    cfcv2ccv = mf.maps.cfcv2ccv
    cfcv2fc = mf.maps.cfcv2fc
    bool = mf.maps.cfcv2fc_bool

    for j in cfcv.pos[c]:(cfcv.pos[c+1]-1)
        c, f, n, i = cfcv.tbl[j]
        cni = cfcv2ccv[j]
        fc = cfcv2fc[j]
        Jfc = bool[j] ? J[fc] : value(J[fc])
        j_cell[cni] += P[2*(c-1) + i, f] * Jfc
    end
end

function vec_to_scalar!(jsq, j, c, model)
    """
    Takes in vector valued field defined on the cell, and returns the
    modulus square
    jsq[c, c'] = S[c, 2*(c-1) + i] * j[c, c', i]^2
    """
    S = model.domain.grid.S
    mf = model.domain.discretizations.charge_flow
    ccv = mf.cellcellvec
    ccv2cc = mf.maps.ccv2cc
    for cni in ccv.pos[c]:(ccv.pos[c+1]-1)
        c, n, i = ccv.tbl[cni]
        cn = ccv2cc[cni]
        jsq[cn] += S[c, 2*(c-1) + i] * j[cni]^2
    end
end


##############################
# Table generating functions #
##############################
# Thses functions generate the tables of indices
# TODO: Make sure the order of the loops give optimal performance

function get_cellfacecellvec_tbl(cdata, cpos)
    cfcv_tbl = []
    cfcv_pos = [1]
    nc = length(cpos) - 1

    for c in 1:nc
        for fp in cpos[c]:(cpos[c+1]-1)
            f = cdata[fp].face

            for i in 1:2
                cfcv = (cell=c, face=f, cell_dep=c, vec=i)
                push!(cfcv_tbl, cfcv)
            end

            for np in cpos[c]:(cpos[c+1]-1)
                n = cdata[np].other
                for i in 1:2
                    cfcv = (cell=c, face=f, cell_dep=n, vec=i)
                    push!(cfcv_tbl, cfcv)
                end
            end

        end
        push!(cfcv_pos, size(cfcv_tbl, 1)+1)
    end

    return (tbl=cfcv_tbl, pos=cfcv_pos)
end


function get_cellcellvec_tbl(cdata, cpos)
    ccv_tbl = []
    ccv_pos = [1]
    nc = length(cpos) - 1

    for c in 1:nc

        for i in 1:2
            ccv = (cell=c, cell_dep=c, vec=i)
            push!(ccv_tbl, ccv)
        end

        for np in cpos[c]:(cpos[c+1]-1)
            n = cdata[np].other
            for i in 1:2
                ccv = (cell=c, cell_dep=n, vec=i)
                push!(ccv_tbl, ccv)
            end
        end

        push!(ccv_pos, size(ccv_tbl, 1)+1)
    end

    return (tbl=ccv_tbl, pos=ccv_pos)
end

function get_cellcell_tbl(cdata, cpos)
    cc_tbl = []
    cc_pos = [1]
    nc = length(cpos) - 1

    for c in 1:nc

        cc = (cell=c, cell_dep=c)
        push!(cc_tbl, cc)
        for np in cpos[c]:(cpos[c+1]-1)
            n = cdata[np].other
            cc = (cell=c, cell_dep=n)
            push!(cc_tbl, cc)
        end

        push!(cc_pos, size(cc_tbl, 1)+1)
    end
    return (tbl=cc_tbl, pos=cc_pos)
end

##############
# Index maps #
##############
# Maps between linear indices
# TODO: the serach functions (get_xx_index) may have improvement potential
# TODO: Improve using search by exploiting that we have c

function get_ccv_index(c, n, i, tbl)
    """ 
    Returns cni, the index of a vector defined on cells, with dependence
    on neighbouring cells. v[cni] is the i'th component of the field in
    cell c, dependent on cell n (for partial derivatives).
    """
    bool = map(x -> x.cell == c && x.cell_dep == n && x.vec == i, tbl)
    indx = findall(bool)
    @assert size(indx, 1) == 1 "Invalid or duplicate face cell combo, size = $(size(indx)) for (c, n, i) = ($c, $n, $i)"
    return indx[1] # cni
end

function get_fc_index(f, c, conn_data)
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

function get_cc_index(c, n, tbl)
    """
    Returns cn, the index of a scalar defined on cells, which also depends
    on neighbouring cells. v[cn] is the value at cell c, dependent on c.
    """
    bool = map(x -> x.cell == c && x.cell_dep == n, tbl)
    indx = findall(bool)
    @assert size(indx, 1) == 1 "Invalid or duplicate face cell combo, size = $(size(indx)) for (c, n) = ($c, $n)"
    return indx[1]
end


function get_cfcv2ccv_map(cfcv, ccv)
    """
    Returns a vector that maps index of cellfacecellvec to cellcellvec
    """
    cfcv2ccv = []
    for a in cfcv.tbl
        c, n, i = a.cell, a.cell_dep, a.vec
        indx = get_ccv_index(c, n, i, ccv.tbl)
        push!(cfcv2ccv, indx)
    end
    @assert size(cfcv.tbl) == size(cfcv2ccv)
    return cfcv2ccv
end

function get_cfcv2fc_map(cfcv, cdata)
    """
    Returns a vector that maps index of cell1facecell2vec to facecell2,
    as well as a vector of booleans. If true, only value of AD should be used
    """
    cfcv2fc = []
    cfcv2fc_bool = []
    for a in cfcv.tbl
        n, f = a.cell_dep, a.face
        indx, bool = get_fc_index(f, n, cdata)
        push!(cfcv2fc, indx)
        push!(cfcv2fc_bool, bool)
    end
    @assert size(cfcv.tbl) == size(cfcv2fc)
    return cfcv2fc, cfcv2fc_bool
end

function get_ccv2cc_map(ccv, cc)
    """
    Returns a vector that maps index of cell1cell2vec to cell1cell2,
    as well as a vector of booleans. If true, only value of AD should be used
    """
    ccv2cc = []
    for a in ccv.tbl
        c, n = a.cell, a.cell_dep
        indx = get_cc_index(c, n, cc.tbl)
        push!(ccv2cc, indx)
    end
    @assert size(ccv.tbl) == size(ccv2cc)
    return ccv2cc
end
