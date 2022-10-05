
# Specialization for Cells()

function index_map(index, m::FiniteVolumeGlobalMap, from_set::EquationSet, to_set::VariableSet, ce::Cells)
    # Previously full_cell
    # @inline full_cell(c, m::FiniteVolumeGlobalMap) = m.inner_to_full_cells[c]
    return m.inner_to_full_cells[index]
end

Base.@propagate_inbounds function index_map(c, m::FiniteVolumeGlobalMap{R}, from_set::VariableSet, to_set::GlobalSet, ce::Cells) where R
    # Previously global_cell
    # Base.@propagate_inbounds global_cell(c, m::FiniteVolumeGlobalMap{R}) where R = m.cells[c]::R
    return m.cells[c]::R
end

function index_map(c_global, m::FiniteVolumeGlobalMap{R}, from_set::GlobalSet, to_set::VariableSet, ce::Cells) where R
    # Previously local_cell
    # local_cell(c_global, m::FiniteVolumeGlobalMap{R}) where R = only(findfirst(isequal(c_global), m.cells))::R
    return only(findfirst(isequal(c_global), m.cells))::R
end

function index_map(c, m::FiniteVolumeGlobalMap{R}, from_set::VariableSet, to_set::EquationSet, ce::Cells) where R
    # Previously interior_cell
    c_i = m.full_to_inner_cells[c]
    return c_i == 0 ? nothing : c_i
end

# Specialization for Faces()

function index_map(f_global, m::FiniteVolumeGlobalMap, from_set::GlobalSet, to_set::VariableSet, ce::Faces)
    # Previously local_face
    # local_face(f_global, m::FiniteVolumeGlobalMap) = only(indexin(f_global, m.faces))
    return only(indexin(f_global, m.faces))
end

function index_map(f, m::FiniteVolumeGlobalMap, from_set::VariableSet, to_set::GlobalSet, ce::Faces)
    # Previously global_face
    # global_face(f, m::FiniteVolumeGlobalMap) = m.faces[f]
    return m.faces[f]
end

global_cell_inside_domain(c, m::FiniteVolumeGlobalMap) = any(isequal(c), m.cells)
Base.@propagate_inbounds cell_is_boundary(c, m::FiniteVolumeGlobalMap) = m.cell_is_boundary[c]::Bool

active_entities(d, m::FiniteVolumeGlobalMap, f::Faces; for_variables = true) = 1:count_entities(d, f)

function active_view(x::AbstractVector, map::FiniteVolumeGlobalMap; for_variables = true)
    if for_variables && map.variables_always_active
        return x
    else
        return view(x, map.inner_to_full_cells)
    end
end

function active_view(x::AbstractMatrix, map::FiniteVolumeGlobalMap; for_variables = true)
    if for_variables && map.variables_always_active
        return x
    else
        return view(x, :, map.inner_to_full_cells)
    end
end


function map_to_active(V, domain, m::FiniteVolumeGlobalMap, ::Cells)
    W = similar(V, 0)
    for i in V
        ix = interior_cell(i, m)
        if !isnothing(ix)
            push!(W, ix)
        end
    end
    return W
    # return filter(i -> m.cell_is_boundary[i], V)
end

function map_ij_to_active(I, J, domain, m::FiniteVolumeGlobalMap, ::Cells)
    n = length(I)
    @assert n == length(J)
    In = copy(I)
    Jn = copy(J)
    active = Vector{Bool}(undef, n)
    for k in 1:n
        i_new = interior_cell(I[k], m)
        j_new = interior_cell(J[k], m)
        keep = !isnothing(i_new) && !isnothing(j_new)
        if keep
            In[k] = i_new
            Jn[k] = j_new
        end
        active[k] = keep
    end
    return (In[active], Jn[active])
end

entity_partition(m::FiniteVolumeGlobalMap, ::Cells) = m.cells
entity_partition(m::FiniteVolumeGlobalMap, ::Faces) = m.faces
