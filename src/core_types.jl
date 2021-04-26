export TervSystem, TervGrid, DefaultPrimaryVariables, TervPrimaryVariables
export SimulationModel, TervPrimaryVariables, DefaultPrimaryVariables, TervFormulation
export setup_parameters

export SingleCUDAContext, SharedMemoryContext, DefaultContext

export transfer_storage_to_context, transfer

# Physical system
abstract type TervSystem end
# Context
abstract type TervContext end
abstract type GPUTervContext <: TervContext end
abstract type CPUTervContext <: TervContext end


struct SingleCUDAContext <: GPUTervContext
    float_t::Type
    index_t::Type

    function SingleCUDAContext(float_t::Type = Float32, index_t::Type = Int32)
        return new(float_t, index_t)
    end
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
    context::C
    formulation::F
end

function allocate_storage(model::TervModel)
    d = Dict()
    allocate_storage!(d, model)
    return d
end

function allocate_storage!(d, model::TervModel)
    allocate_storage!(d, model.grid, model.system)
end

function transfer(context::TervContext, v)
    return v
end


function transfer(context::SingleCUDAContext, v::AbstractArray{I}) where {I<:Integer}
    return CuArray{context.index_t}(v)
end

function transfer(context::SingleCUDAContext, v::AbstractArray{I}) where {I<:AbstractFloat}
    return CuArray{context.float_t}(v)
end

function transfer_storage_to_context(model::TervModel, storage)
    new_storage = transfer_storage_to_context(model.context, storage)
    return new_storage
end

function transfer_storage_to_context(context::TervContext, storage)
    return storage
end


function SimulationModel(G, system;
                                 formulation = FullyImplicit(DefaultPrimaryVariables()), 
                                 context = DefaultContext())
    return SimulationModel(G, system, context, formulation)
end


# context stuff
function transfer_model_to_context(model::SimulationModel)
    grid = transfer_grid_to_context(model.context, model.grid)
    return typeof(model)(grid, model.system, model.context, model.formulation)
end


function setup_parameters(model)
    return Dict{String, Any}()
end