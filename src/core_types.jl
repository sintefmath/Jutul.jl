export TervSystem, TervDomain, DefaultPrimaryVariables, TervPrimaryVariables
export SimulationModel, TervPrimaryVariables, DefaultPrimaryVariables, TervFormulation
export setup_parameters, kernel_compatibility

export SingleCUDAContext, SharedMemoryContext, DefaultContext

export transfer_storage_to_context, transfer, allocate_array

# Physical system
abstract type TervSystem end


function select_primary_variables(system::TervSystem, formulation, discretization)
    return nothing
end

# Discretization - currently unused
abstract type TervDiscretization end
struct DefaultDiscretization <: TervDiscretization end

# Primary variables
abstract type TervPrimaryVariables end

function number_of_units(model, ::TervPrimaryVariables)
    # By default, each primary variable exists on all cells of a discretized domain
    return number_of_cells(model.domain)
end

function number_of_degrees_of_freedom(model, pvars::TervPrimaryVariables)
    return number_of_units(model, pvars)*degrees_of_freedom_per_unit(pvars)
end

function absolute_increment_limit(::TervPrimaryVariables) nothing end
function relative_increment_limit(::TervPrimaryVariables) nothing end
function maximum_value(::TervPrimaryVariables) nothing end
function minimum_value(::TervPrimaryVariables) nothing end

function update_state!(state, p::TervPrimaryVariables, model, dx)
    names = get_names(p)
    nu = number_of_units(model, p)
    abs_max = absolute_increment_limit(p)
    rel_max = relative_increment_limit(p)
    maxval = maximum_value(p)
    minval = minimum_value(p)

    for (index, name) in enumerate(names)
        offset = nu*(index-1)
        v = state[Symbol(name)] # TODO: Figure out this.
        dv = view(dx, (1:nu) .+ offset)
        @. v = update_value(v, dv, abs_max, rel_max, minval, maxval)
    end
end

@inline function choose_increment(v::F, dv::F, abs_change, rel_change, minval, maxval) where {F<:AbstractFloat}
    dv = limit_abs(dv, abs_change)
    dv = limit_rel(v, dv, rel_change)
    dv = limit_lower(v, dv, minval)
    dv = limit_upper(v, dv, maxval)
    return dv
end

# Limit absolute
function limit_abs(dv, abs_change)
    dv = sign(dv)*min(abs(dv), abs_change)
end

function limit_abs(dv, ::Nothing) dv end

# Limit relative 
function limit_rel(v, dv, rel_change)
    dv = limit_abs(dv, rel_change*abs(v))
end

function limit_rel(v, dv, ::Nothing) dv end
# Lower bounds
function limit_upper(v, dv, maxval)
    if dv > 0 && v + dv > maxval
        dv = maxval - v
    end
    return dv
end

function limit_upper(v, dv, maxval::Nothing) dv end

# Upper bounds
function limit_lower(v, dv, minval)
    if dv < 0 && v + dv < minval
        dv = minval - v
    end
    return dv
end

function limit_lower(v, dv, minval::Nothing) dv end

function update_value(v, dv, arg...)
    return v + choose_increment(value(v), dv, arg...)
end

abstract type ScalarPrimaryVariable <: TervPrimaryVariables end

function degrees_of_freedom_per_unit(::ScalarPrimaryVariable)
    return 1
end

function initialize_primary_variable_ad(state, model, pvar::ScalarPrimaryVariable, offset, npartials)
    name = get_name(pvar)
    state[name] = allocate_array_ad(state[name], diag_pos = offset + 1, context = model.context, npartials = npartials)
    return state
end

function initialize_primary_variable_value(state, model, pvar::ScalarPrimaryVariable, val::Union{Dict, AbstractFloat})
    n = number_of_degrees_of_freedom(model, pvar)
    name = get_name(pvar)
    if isa(val, Dict)
        val = val[name]
    end

    if isa(val, AbstractVector)
        V = deepcopy(val)
        @assert length(val) == n "Variable was neither scalar nor the expected dimension"
    else
        V = repeat([val], n)
    end
    state[name] = transfer(model.context, V)
    return state
end


function get_names(v::TervPrimaryVariables)
    return [get_name(v)]
end

function get_symbol(v::TervPrimaryVariables)
    return v.symbol
end

function get_name(v::TervPrimaryVariables)
    return String(get_symbol(v))
end

function number_of_primary_variables(model)
    # TODO: Bit of a mess (number of primary variables, vs number of actual primary variables realized on grid. Fix.)
    return length(get_primary_variable_names(model))
end



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


function float_type(c::TervContext)
    return Float64
end

function index_type(c::TervContext)
    return Int64
end


"Synchronize backend after allocations if needed"
function synchronize(::TervContext)
    # Do nothing
end

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

function synchronize(::SingleCUDAContext)
    CUDA.synchronize()
end

# For many GPUs we want to use single precision. Specialize interfaces accordingly.
function float_type(c::SingleCUDAContext)
    return c.float_t
end

function index_type(c::SingleCUDAContext)
    return c.index_t
end

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

# Formulation
abstract type TervFormulation end
struct FullyImplicit <: TervFormulation
end

# Equations
abstract type TervEquation end

function number_of_equations_per_unit(::TervEquation)
    # Default: One equation per unit (= cell,  face, ...)
    return 1
end

function number_of_equations(model, e::TervEquation)
    # Default: Equations are per cell
    return number_of_equations_per_unit(e)*number_of_cells(model.domain)
end

"""
Update an equation so that it knows where to store its derivatives
in the given linearized system.
"""
function align_to_linearized_system!(::TervEquation, lsys, model) end

# Transfer operators

function context_convert(context::TervContext, v::Real)
    return context_convert(context, [v])
end

function context_convert(context::TervContext, v::AbstractArray)
    return Array(v)
end

function context_convert(context::SingleCUDAContext, v::AbstractArray)
    return CuArray(v)
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

