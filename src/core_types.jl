export TervSystem, DefaultPrimaryVariables, TervPrimaryVariables
export SimulationModel, TervPrimaryVariables, DefaultPrimaryVariables


# Physical system
abstract type TervSystem end
# Models 
abstract type TervModel end
# Concrete models follow
struct SimulationModel <: TervModel
    system::TervSystem
    formulation::TervFormulation
    primary_variables::TervPrimaryVariables
    context::TervContext
end

function SimulationModel(system; formulation = FullyImplicit(), 
                                 primary_variables = DefaultPrimaryVariables(), 
                                 context = DefaultContext())
    return SimulationModel(system, formulation, primary_variables, context)
end

# Formulation
abstract type TervFormulation end
struct FullyImplicit <: TervFormulation end


abstract type TervEquation end

abstract type TervPrimaryVariables end
struct DefaultPrimaryVariables <: TervPrimaryVariables end