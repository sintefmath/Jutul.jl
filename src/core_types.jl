export TervSystem, TervDomain, TervPrimaryVariables
export SimulationModel, TervPrimaryVariables, TervFormulation
export setup_parameters, kernel_compatibility
export Cells, Nodes, Faces

export SingleCUDAContext, SharedMemoryContext, DefaultContext

export  transfer, allocate_array

# Physical system
abstract type TervSystem end

# Discretization - currently unused
abstract type TervDiscretization end
struct DefaultDiscretization <: TervDiscretization end

# Primary variables
abstract type TervPrimaryVariables end
abstract type ScalarPrimaryVariable <: TervPrimaryVariables end
abstract type GroupedPrimaryVariables <: TervPrimaryVariables end

# abstract type PrimaryVariableConstraint end

# Context
abstract type TervContext end
abstract type GPUTervContext <: TervContext end
abstract type CPUTervContext <: TervContext end
# Traits for context
abstract type KernelSupport end
struct KernelAllowed <: KernelSupport end
struct KernelDisallowed <: KernelSupport end

kernel_compatibility(::Any) = KernelDisallowed()
# Trait if we are to use broadcasting
abstract type BroadcastSupport end
struct BroadcastAllowed <: BroadcastSupport end
struct BroadcastDisallowed <: BroadcastSupport end
broadcast_compatibility(::Any) = BroadcastAllowed()

# CUDA context - everything on the single CUDA device attached to machine
struct SingleCUDAContext <: GPUTervContext
    float_t::Type
    index_t::Type
    block_size
    device

    function SingleCUDAContext(float_t::Type = Float32, index_t::Type = Int32, block_size = 256)
        @assert CUDA.functional() "CUDA must be functional for this context."
        return new(float_t, index_t, block_size, CUDADevice())
    end
end

kernel_compatibility(::SingleCUDAContext) = KernelAllowed()

"Context that uses KernelAbstractions for GPU parallelization"
struct SharedMemoryKernelContext <: CPUTervContext
    block_size
    device
    function SharedMemoryKernelContext(block_size = Threads.nthreads())
        # Remark: No idea what block_size means here.
        return new(block_size, CPU())
    end
end

kernel_compatibility(::SharedMemoryKernelContext) = KernelAllowed()

"Context that uses threads etc to accelerate loops"
struct SharedMemoryContext <: CPUTervContext
    
end

broadcast_compatibility(::SharedMemoryContext) = BroadcastDisallowed()

"Default context - not really intended for threading"
struct DefaultContext <: CPUTervContext

end

# Domains
abstract type TervDomain end

struct DiscretizedDomain{G} <: TervDomain
    grid::G
    discretizations
    units
end

function DiscretizedDomain(grid, disc)
    units = declare_units(grid)
    u = Dict{Any, Int64}() # Is this a good definition?
    for unit in units
        num = unit[2]
        @assert num >= 0 "Units must have non-negative counts."
        u[unit[1]] = num
    end
    DiscretizedDomain(grid, disc, u) 
end

# Formulation
abstract type TervFormulation end
struct FullyImplicit <: TervFormulation end

# Equations
abstract type TervEquation end

# Models
abstract type TervModel end

struct SimulationModel{O<:TervDomain, 
                       S<:TervSystem,
                       F<:TervFormulation,
                       C<:TervContext} <: TervModel
    domain::O
    system::S
    context::C
    formulation::F
    primary_variables
end


function SimulationModel(domain, system;
    formulation = FullyImplicit(), 
    context = DefaultContext())
    domain = transfer(context, domain)
    primary = select_primary_variables(domain, system, formulation)
    function check_prim(pvar)
        a = map(associated_unit, pvar)
        for u in unique(a)
            ut = typeof(u)
            deltas =  diff(findall(typeof.(a) .== ut))
            if any(deltas .!= 1)
                error("All primary variables of the same type must come sequentially: Error ocurred for $ut:\nPrimary: $pvar\nTypes: $a")
            end
        end
    end
    check_prim(primary)
    return SimulationModel(domain, system, context, formulation, primary)
end
## Grid
abstract type TervGrid end

## Discretized units
abstract type TervUnit end

struct Cells <: TervUnit end
struct Faces <: TervUnit end
struct Nodes <: TervUnit end
