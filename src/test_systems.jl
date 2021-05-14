export ScalarTestSystem, ScalarTestDomain

struct ScalarTestSystem <: TervSystem end

struct ScalarTestDomain <: TervDomain end

function number_of_cells(::ScalarTestDomain) 1 end

struct ScalarTestEquation <: TervEquation
    equation
    function ScalarTestEquation(G::TervDomain, npartials::Integer; context = DefaultContext())
        e = allocate_array_ad(number_of_cells(G), 1, context = context, npartials = npartials)
        new(e)
    end    
end

function declare_sparsity(model, e::ScalarTestEquation)
    return (1, 1)
end

function allocate_equations!(eqs, storage, model::SimulationModel{G, S}) where {G<:ScalarTestDomain, S<:ScalarTestSystem}
    @debug "Allocating equations ScalarTestSystem"
    law = ScalarTestEquation(model.domain, 1, context = model.context)
    eqs["TestEquation"] = law
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
