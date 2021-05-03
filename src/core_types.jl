export TervSystem, TervGrid, DefaultPrimaryVariables, TervPrimaryVariables
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
    # By default, each primary variable exists on all cells of a "grid"
    return number_of_cells(model.grid)
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
        v = state[name]
        dv = view(dx, (1:nu) .+ offset)
        @. v = update_value(v, dv, abs_max, rel_max, minval, maxval)
    end
end

function update_value(v, dv, abs_change, rel_change, minval, maxval)
    s = sign(dv)
    if !isnothing(abs_change)
        dv = s*min(abs(dv), abs_change)
    end
    if !isnothing(rel_change)
        dmax = rel_change*abs(v)
        dv = s*min(abs(dv), dmax)
    end
    if !isnothing(minval) && dv < 0
        if value(v) + dv < minval
            dv = minval - value(v)
        end
    end
    if !isnothing(maxval) && dv > 0
        if value(v) + dv > maxval
            dv = maxval - value(v)
        end
    end
    v += dv
end

abstract type ScalarPrimaryVariable <: TervPrimaryVariables end

function degrees_of_freedom_per_unit(::ScalarPrimaryVariable)
    return 1
end

function initialize_primary_variable_ad(state, model, pvar::ScalarPrimaryVariable, offset, npartials)
    name = get_name(pvar)
    v_n = state[name]
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


function get_names(v::ScalarPrimaryVariable)
    return [get_name(v)]
end

function get_name(v::ScalarPrimaryVariable)
    return v.name
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

struct SharedMemoryContext <: CPUTervContext
    block_size
    device
    function SharedMemoryContext(block_size = Threads.nthreads())
        # Remark: No idea what block_size means here.
        return new(block_size, CPU())
    end
end

kernel_compatibility(::SharedMemoryContext) = KernelAllowed()


struct DefaultContext <: CPUTervContext

end

# Grids
abstract type TervGrid end

# Formulation
abstract type TervFormulation end
struct FullyImplicit <: TervFormulation
end

# Equations
abstract type TervEquation end


# Models 
abstract type TervModel end

# Concrete models follow
struct SimulationModel{G<:TervGrid, 
                       S<:TervSystem,
                       F<:TervFormulation,
                       C<:TervContext,
                       D<:TervDiscretization} <: TervModel
    grid::G
    system::S
    context::C
    formulation::F
    discretization::D
    primary_variables
end


function get_primary_variable_names(model::SimulationModel)
    return map((x) -> get_name(x), get_primary_variables(model))
end

function get_primary_variables(model::SimulationModel)
    return model.primary_variables
end

function allocate_storage(model::TervModel)
    d = Dict()
    allocate_storage!(d, model)
    return d
end

function allocate_storage!(d, model::TervModel)
    # Do nothing for Any.
end

function convert(context::TervContext, v::Real)
    return convert(context, [v])
end

function convert(context::TervContext, v)
    return Array(v)
end

function convert(context::SingleCUDAContext, v)
    return CuArray(v)
end


function allocate_array(context::TervContext, value, n...)
    tmp = convert(context, [value])
    return repeat(tmp, n...)
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
                                 formulation = FullyImplicit(), 
                                 context = DefaultContext(),
                                 discretization = DefaultDiscretization())
    grid = transfer(context, G)
    primary = select_primary_variables(system, formulation, discretization)
    return SimulationModel(grid, system, context, formulation, discretization, primary)
end

function setup_parameters(model)
    return Dict{String, Any}()
end