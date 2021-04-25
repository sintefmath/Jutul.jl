export TervSystem, TervGrid, DefaultPrimaryVariables, TervPrimaryVariables
export SimulationModel, TervPrimaryVariables, DefaultPrimaryVariables, TervFormulation
export setup_parameters

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
# Grids
abstract type TervGrid end

# Formulation
abstract type TervFormulation end
struct FullyImplicit <: TervFormulation 
    primary_variables
end


# Primary variables
abstract type TervPrimaryVariables end
struct DefaultPrimaryVariables <: TervPrimaryVariables end

# Equations
abstract type TervEquation end


# Models 
abstract type TervModel end

# Concrete models follow
struct SimulationModel{G<:TervGrid, 
                       S<:TervSystem,
                       F<:TervFormulation,
                       C<:TervContext} <: TervModel
    grid::G
    system::S
    formulation::F
    context::C
end

function allocate_storage(model::TervModel)
    d = Dict()
    allocate_storage!(d, model)
    return d
end

function allocate_storage!(d, model::TervModel)
    allocate_storage!(d, model.grid, model.system)
end

function SimulationModel(G, system;
                                 formulation = FullyImplicit(DefaultPrimaryVariables()), 
                                 context = DefaultContext())
    return SimulationModel(G, system, formulation, context)
end



function setup_parameters(model)
    return Dict{String, Any}()
end