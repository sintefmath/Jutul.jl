export ScalarTestSystem, ScalarTestDomain, ScalarTestForce

struct ScalarTestSystem <: TervSystem end

struct ScalarTestDomain <: TervDomain end

function number_of_cells(::ScalarTestDomain) 1 end

function get_domain_intersection(u::TervUnit, target_d::ScalarTestDomain, source_d::ScalarTestDomain)
    # This domain always interacts with the single cell in instances of itself, and nothing else
    return (1, 1, Cells())
end

# Driving force for the test equation
struct ScalarTestForce 
    value
end

# Equations
struct ScalarTestEquation <: TervEquation
    equation
    equation_jac_pos
    function ScalarTestEquation(G::TervDomain, npartials::Integer; context = DefaultContext())
        I = index_type(context)
        nc = number_of_cells(G)
        @assert nc == 1 # We use nc for clarity of the interface - but it should always be one!
        e = allocate_array_ad(1, nc, context = context, npartials = npartials)
        v = zeros(I, npartials, nc)
        new(e, v)
    end
end

function declare_sparsity(model, e::ScalarTestEquation)
    return (1, 1, 1, 1)
end

function align_to_jacobian!(eq::ScalarTestEquation, jac, model; row_offset = 0, col_offset = 0)
    eq.equation_jac_pos .= find_sparse_position(jac, row_offset + 1, col_offset + 1)
end

# Model features
function allocate_equations!(eqs, storage, model::SimulationModel{G, S}) where {G<:ScalarTestDomain, S<:ScalarTestSystem}
    eqs[:TestEquation] = ScalarTestEquation(model.domain, 1, context = model.context)
end

function update_equation!(eq::ScalarTestEquation, storage, model, dt)
    X = storage.state.XVar
    X0 = storage.state0.XVar
    @. eq.equation = (X - X0)/dt
end

function build_forces(model::SimulationModel{G, S}; sources = nothing) where {G<:ScalarTestDomain, S<:ScalarTestSystem}
    return (sources = sources,)
end

function apply_forces_to_equation!(storage, model, eq::ScalarTestEquation, force::ScalarTestForce)
    @. eq.equation -= force.value
end

struct XVar <: ScalarPrimaryVariable
    symbol
end

function XVar()
    XVar(:XVar)
end

function select_primary_variables(domain, system::ScalarTestSystem, formulation)
    return [XVar()]
end
