export ScalarTestSystem, ScalarTestDomain

struct ScalarTestSystem <: TervSystem end

struct ScalarTestDomain <: TervDomain end

function number_of_cells(::ScalarTestDomain) 1 end

function allocate_equations!(storage, model::SimulationModel{G, S}) where {G<:ScalarTestDomain, S<:ScalarTestSystem}
    @debug "Allocating equations ScalarTestSystem"
    eqs = Dict()
    law = ScalarTestEquation(model.domain, npartials, context = model.context)
    eqs["TestEquation"] = law
    storage["Equations"] = eqs
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
