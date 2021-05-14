export ScalarTestSystem, ScalarTestDomain

struct ScalarTestSystem <: TervSystem end

struct ScalarTestDomain <: TervDomain end

function number_of_cells(::ScalarTestDomain) 1 end

# Equations
struct ScalarTestEquation <: TervEquation
    equation
    equation_jac_pos
    function ScalarTestEquation(G::TervDomain, npartials::Integer; context = DefaultContext())
        I = index_type(context)
        nc = number_of_cells(G)
        @assert nc == 1 # We use nc for clarity of the interface - but it should always be one!
        e = allocate_array_ad(nc, 1, context = context, npartials = npartials)
        v = zeros(I, npartials, nc)
        new(e, v)
    end
end

function declare_sparsity(model, e::ScalarTestEquation)
    return (1, 1)
end

function align_to_linearized_system!(eq::ScalarTestEquation, lsys::LinearizedSystem, model; row_offset = 0, col_offset = 0)
    eq.equation_jac_pos .= find_sparse_position(lsys.jac, row_offset + 1, col_offset + 1)
end

# Model features
function allocate_equations!(eqs, storage, model::SimulationModel{G, S}) where {G<:ScalarTestDomain, S<:ScalarTestSystem}
    eqs["TestEquation"] = ScalarTestEquation(model.domain, 1, context = model.context)
end

struct XVar <: ScalarPrimaryVariable
    symbol
end

function XVar()
    XVar(:XVar)
end

function select_primary_variables(system::ScalarTestSystem, formulation, discretization)
    return [XVar()]
end
