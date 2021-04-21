export TervSystem, DefaultPrimaryVariables, TervPrimaryVariables
export SimulationModel, TervPrimaryVariables, DefaultPrimaryVariables, TervFormulation


# Physical system
abstract type TervSystem end
# Context
abstract type TervContext end
abstract type GPUTervContext <: TervContext end
abstract type CPUTervContext <: TervContext end


struct SingleCUDAContext <: GPUTervContext

end

struct SharedMemoryContext <: CPUTervContext

end

struct DefaultContext <: CPUTervContext

end
# Formulation
abstract type TervFormulation end
struct FullyImplicit <: TervFormulation end
# Primary variables
abstract type TervPrimaryVariables end
struct DefaultPrimaryVariables <: TervPrimaryVariables end

# Equations
abstract type TervEquation end


# Models 
abstract type TervModel end
# Concrete models follow
struct SimulationModel <: TervModel
    G::TervGrid
    system::TervSystem
    formulation::TervFormulation
    primary_variables::TervPrimaryVariables
    context::TervContext
end

function allocate_storage(model::TervModel)
    d = Dict()
    allocate_storage!(d, model)
    return d
end

function allocate_storage!(d, model::TervModel)
    allocate_storage!(d, model.G, model.system)
end

function SimulationModel(G, system; formulation = FullyImplicit(), 
                                 primary_variables = DefaultPrimaryVariables(), 
                                 context = DefaultContext())
    return SimulationModel(G, system, formulation, primary_variables, context)
end



