export ScalarTestSystem, ScalarTestDomain

struct ScalarTestSystem <: TervSystem end

struct ScalarTestDomain <: TervDomain end

function number_of_cells(::ScalarTestDomain) 1 end

function update_equations!(model::SimulationModel{G, S}, storage; 
    dt = nothing, sources = nothing) where {G<:ScalarTestDomain, S<:ScalarTestSystem}
    
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
