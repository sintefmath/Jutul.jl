export ScalarTestSystem, ScalarTestDomain

struct ScalarTestSystem <: TervSystem end

struct ScalarTestDomain <: TervDomain end


function update_equations!(model::SimulationModel{G, S}, storage; 
    dt = nothing, sources = nothing) where {G<:ScalarTestDomain, S<:ScalarTestSystem}
    
end

struct XTest <: ScalarPrimaryVariable
    symbol
end

function XTest()
    XTest(:XTest)
end

function select_primary_variables(system::ScalarTestSystem, formulation, discretization)
    return [XTest()]
end
