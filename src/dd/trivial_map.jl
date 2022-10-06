@enum EntityStatus Inner Boundary NotPresent

function index_map(index, ::TrivialGlobalMap, from_set::AbstractVariableSet, to_set::AbstractVariableSet, c::JutulEntity)
    # The trivial map always gives the same index
    return index
end

function entity_status(index, ::TrivialGlobalMap, var_type::AbstractVariableSet, ::JutulEntity)
    # Trivial map means nothing is on the boundary or missing
    return Inner
end

function is_local_boundary(index, m, var_type, e::JutulEntity)
    return entity_status(index, m, var_type, e) == Boundary
end

global_map(model::SimulationModel) = global_map(model.domain)
global_map(domain::DiscretizedDomain) = domain.global_map
global_map(domain_or_model) = TrivialGlobalMap()

"Local face -> global face (full set)"
global_face(f, m) = index_map(f, m, VariableSet(), GlobalSet(), Faces())
"Local cell -> global cell (full set)"
global_cell(c, m) = index_map(c, m, VariableSet(), GlobalSet(), Cells())

"Global cell -> local cell (full set)"
local_cell(c, m) = index_map(c, m, GlobalSet(), VariableSet(), Cells())
cell_is_boundary(c, map) = is_local_boundary(c, map, Cells())

"Global face -> local face (full set)"
local_face(f, m) = index_map(f, m, GlobalSet(), VariableSet(), Faces())
"Local cell in full set -> inner cell (or zero)"
interior_cell(c, m) = index_map(c, m, VariableSet(), EquationSet(), Cells())

"Inner cell to local cell (full set)"
@inline full_cell(c, m) = index_map(c, m, EquationSet(), VariableSet(), Cells())

global_cell_inside_domain(c, m) = true

interior_face(f, m) = f

map_to_active(V, domain::DiscretizedDomain, entity) = map_to_active(V, domain, global_map(domain), entity)
map_to_active(V, domain, entity) = V
map_to_active(V, domain, m, entity) = V

map_ij_to_active(I, J, domain, entity) = map_ij_to_active(I, J, domain, domain.global_map, entity)
map_ij_to_active(I, J, domain, m::TrivialGlobalMap, entity) = (I, J)

export active_entities
# Specialize cells, leave faces be (probably already filtered)
active_cells(model; kwarg...) = active_entities(model.domain, Cells(); kwarg...)

function active_entities(d, m::FiniteVolumeGlobalMap, c::Cells; for_variables = true)
    if for_variables && m.variables_always_active
        return 1:count_entities(d, c)
    else
        return m.inner_to_full_cells
    end
end
active_entities(d::DiscretizedDomain, entity; kwarg...) = active_entities(d, d.global_map, entity; kwarg...)
active_entities(d::DiscretizedDomain, ::TrivialGlobalMap, entity; kwarg...) = 1:count_entities(d, entity)
active_entities(d, entity; kwarg...) = 1:count_entities(d, entity)

active_view(x, map; kwarg...) = x

# TODO: Probably a bit inefficient
count_active_entities(d, m, e; kwarg...) = length(active_entities(d, m, e; kwarg...))
entity_partition(m::TrivialGlobalMap, e) = Colon()
