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

function float_type(c::TervContext)
    return Float64
end

function index_type(c::TervContext)
    return Int64
end


struct SingleCUDAContext <: GPUTervContext
    float_t::Type
    index_t::Type

    function SingleCUDAContext(float_t::Type = Float32, index_t::Type = Int32)
        return new(float_t, index_t)
    end
end

function float_type(c::SingleCUDAContext)
    return c.float_t
end

function index_type(c::SingleCUDAContext)
    return c.index_t
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
    # Do nothing for Any.
end

function allocate_vector(context::TervContext, value::T, n) where {T<:Real}
    v = Vector{T}(undef, n)
    fill!(v, value)
    return v
end

function allocate_vector(context::SingleCUDAContext, value::T, n) where {T<:Real}
    v = CuVector{T}(undef, n)
    fill!(v, value)
    return v
end

function transfer(context::TervContext, v)
    return v
end

function transfer(context::SingleCUDAContext, v::AbstractArray{I}) where {I<:Integer}
    return CuArray{context.index_t}(v)
end

function transfer(context::SingleCUDAContext, v::AbstractArray{F}) where {F<:AbstractFloat}
    return CuArray{context.float_t}(v)
end


function SimulationModel(G, system;
                                 formulation = FullyImplicit(DefaultPrimaryVariables()), 
                                 context = DefaultContext())
    grid = transfer(context, G)
    return SimulationModel(grid, system, context, formulation)
end

function setup_parameters(model)
    return Dict{String, Any}()
end