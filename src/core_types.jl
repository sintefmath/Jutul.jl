export TervSystem, TervDomain, TervVariables
export SimulationModel, TervVariables, TervFormulation
export setup_parameters, kernel_compatibility
export Cells, Nodes, Faces
export ConstantVariables

export SingleCUDAContext, SharedMemoryContext, DefaultContext
export BlockMajorLayout, EquationMajorLayout, UnitMajorLayout

export transfer, allocate_array

export TervStorage

import Base: show

# Physical system
abstract type TervSystem end

# Discretization - currently unused
abstract type TervDiscretization end
# struct DefaultDiscretization <: TervDiscretization end

# Primary/secondary variables
abstract type TervVariables end
abstract type ScalarVariable <: TervVariables end
abstract type GroupedVariables <: TervVariables end


# Functions of the state
abstract type TervStateFunction <: TervVariables end

# Driving forces
abstract type TervForce end

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

# Traits etc for matrix ordering
abstract type TervMatrixLayout end
"""
Equations are stored sequentially in rows, derivatives of same type in columns:
"""
struct EquationMajorLayout <: TervMatrixLayout
    as_adjoint::Bool
end
function EquationMajorLayout() EquationMajorLayout(false) end
function is_cell_major(::EquationMajorLayout) false end
"""
Domain units sequentially in rows:
"""
struct UnitMajorLayout <: TervMatrixLayout
    as_adjoint::Bool
end
function UnitMajorLayout() UnitMajorLayout(false) end
function is_cell_major(::UnitMajorLayout) true end

"""
Same as UnitMajorLayout, but the nzval is a matrix
"""
struct BlockMajorLayout <: TervMatrixLayout
    as_adjoint::Bool
end
function BlockMajorLayout() BlockMajorLayout(false) end
function is_cell_major(::BlockMajorLayout) true end

matrix_layout(::Any) = EquationMajorLayout(false)
function represented_as_adjoint(layout)
    layout.as_adjoint
end

struct SparsePattern{L}
    I
    J
    n
    m
    block_n
    block_m
    layout::L
    function SparsePattern(I, J, n::T, m::T, layout, block_n::T = 1, block_m::T = block_n) where {T <: Integer}
        if isa(I, Integer)
            @assert isa(J, Integer)
            I = vec(I)
            J = vec(J)
        end
        if length(I) > 0
            I::AbstractVector{T}
            J::AbstractVector{T}
            @assert length(I) == length(J)
            @assert maximum(I) <= n
            @assert minimum(I) > 0
            @assert maximum(J) <= m
            @assert minimum(J) > 0
        else
            # Empty vectors might be Any, and the asserts above
            # cannot be used.
            I = Vector{T}()
            J = Vector{T}()
        end
        @assert n > 0
        @assert m > 0
        new{typeof(layout)}(I, J, n, m, block_n, block_m, layout)
    end
end

ijnm(p::SparsePattern) = (p.I, p.J, p.n, p.m)
block_size(p::SparsePattern) = (p.block_n, p.block_m)

# CUDA context - everything on the single CUDA device attached to machine
struct SingleCUDAContext <: GPUTervContext
    float_t::Type
    index_t::Type
    block_size
    device
    matrix_layout
    function SingleCUDAContext(float_t::Type = Float32, index_t::Type = Int64, block_size = 256, layout = EquationMajorLayout())
        @assert CUDA.functional() "CUDA must be functional for this context."
        return new(float_t, index_t, block_size, CUDADevice(), layout)
    end
end
matrix_layout(c::SingleCUDAContext) = c.matrix_layout
kernel_compatibility(::SingleCUDAContext) = KernelAllowed()

"Context that uses KernelAbstractions for GPU parallelization"
struct SharedMemoryKernelContext <: CPUTervContext
    block_size
    device
    function SharedMemoryKernelContext(block_size = nthreads())
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
    matrix_layout
    function DefaultContext(; matrix_layout = EquationMajorLayout())
        new(matrix_layout)
    end
end

matrix_layout(c::DefaultContext) = c.matrix_layout

function jacobian_eltype(context, layout, block_size)
    return float_type(context)
end

function r_eltype(context, layout, block_size)
    return float_type(context)
end

function jacobian_eltype(context::CPUTervContext, layout::BlockMajorLayout, block_size)
    return SMatrix{block_size..., float_type(context), prod(block_size)}
end

function r_eltype(context::CPUTervContext, layout::BlockMajorLayout, block_size)
    return SVector{block_size[1], float_type(context)}
end


# Domains
abstract type TervDomain end

struct DiscretizedDomain{G} <: TervDomain
    grid::G
    discretizations
    units
end

function DiscretizedDomain(grid, disc = nothing)
    units = declare_units(grid)
    u = Dict{Any, Int64}() # Is this a good definition?
    for unit in units
        num = unit.count
        @assert num >= 0 "Units must have non-negative counts."
        u[unit.unit] = num
    end
    DiscretizedDomain(grid, disc, u)
end

function transfer(context::SingleCUDAContext, domain::DiscretizedDomain)
    F = context.float_t
    I = context.index_t
    t = (x) -> transfer(context, x)

    g = t(domain.grid)
    d_cpu = domain.discretizations

    k = keys(d_cpu)
    val = map(t, values(d_cpu))
    d = (;zip(k, val)...)
    u = domain.units
    return DiscretizedDomain(g, d, u)
end


# Formulation
abstract type TervFormulation end
struct FullyImplicit <: TervFormulation end

# Equations
abstract type TervEquation end
abstract type DiagonalEquation <: TervEquation end

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
    secondary_variables
    equations
    output_variables
    function SimulationModel(domain, system;
                                            formulation = FullyImplicit(),
                                            context = DefaultContext(),
                                            output_level = :primary_variables
                                            )
        domain = transfer(context, domain)
        primary = select_primary_variables(domain, system, formulation)
        primary = transfer(context, primary)
        function check_prim(pvar)
            a = map(associated_unit, values(pvar))
            for u in unique(a)
                ut = typeof(u)
                deltas =  diff(findall(typeof.(a) .== ut))
                if any(deltas .!= 1)
                    error("All primary variables of the same type must come sequentially: Error ocurred for $ut:\nPrimary: $pvar\nTypes: $a")
                end
            end
        end
        check_prim(primary)
        secondary = select_secondary_variables(domain, system, formulation)
        secondary = transfer(context, secondary)

        equations = select_equations(domain, system, formulation)
        outputs = select_output_variables(domain, system, formulation, primary, secondary, output_level)

        D = typeof(domain)
        S = typeof(system)
        F = typeof(formulation)
        C = typeof(context)
        new{D, S, F, C}(domain, system, context, formulation, primary, secondary, equations, outputs)
    end
end

function Base.show(io::IO, t::MIME"text/plain", model::SimulationModel)
    println("SimulationModel:")
    for f in fieldnames(typeof(model))
        p = getfield(model, f)
        print("  $f:\n")
        if f == :primary_variables || f == :secondary_variables
            ctr = 1
            for (key, pvar) in p
                nv = degrees_of_freedom_per_unit(model, pvar)
                nu = number_of_units(model, pvar)
                u = associated_unit(pvar)
                print("   $ctr) $key (")
                if nv > 1
                    print("$nv×")
                end
                print("$nu")

                print(" ∈ $(typeof(u)))\n")
                ctr += 1
            end
            print("\n")
        elseif f == :domain
            if hasproperty(p, :grid)
                g = p.grid
                print("    grid: $(typeof(g))")
            else

            end
            print("\n\n")
        elseif f == :equations
            ctr = 1
            for (key, eq) in p
                println("   $ctr) $key implemented as $(eq[2]) × $(eq[1])")
                ctr += 1
            end
            print("\n")
        else
            println("    $p\n")
        end
    end
end

# Grids etc

## Grid
abstract type TervGrid end

## Discretized units
abstract type TervUnit end

struct Cells <: TervUnit end
struct Faces <: TervUnit end
struct Nodes <: TervUnit end

# Sim model

function SimulationModel(g::TervGrid, system; discretization = nothing, kwarg...)
    # Simple constructor that assumes
    d = DiscretizedDomain(g, discretization)
    SimulationModel(d, system; kwarg...)
end

"""
A set of constants, repeated over the entire set of Cells or some other unit
"""
struct ConstantVariables <: GroupedVariables
    constants
    unit::TervUnit
    single_unit::Bool
    function ConstantVariables(constants, unit = Cells(), single_unit = nothing)
        if isnothing(single_unit)
            single_unit = isa(constants, AbstractVector)
        end
        new(constants, unit, single_unit)
    end
end


struct TervStorage
    data::Union{Dict{Symbol, Any}, NamedTuple}
    function TervStorage(S = Dict{Symbol, Any}())
        new(S)
    end
end

function convert_to_immutable_storage(S::TervStorage)
    return TervStorage(convert_to_immutable_storage(S.data))
end

function Base.getproperty(S::TervStorage, name::Symbol)
    data = getfield(S, :data)
    if name == :data
        return data
    else
        return getproperty(data, name)
    end
end

function Base.setproperty!(S::TervStorage, name::Symbol, x)
    setproperty!(S.data, name, x)
end

function Base.setindex!(S::TervStorage, x, name::Symbol)
    setindex!(S.data, x, name)
end

function Base.getindex(S::TervStorage, name::Symbol)
    getindex(S.data, name)
end

function Base.haskey(S::TervStorage, name::Symbol)
    return haskey(S.data, name)
end

function Base.keys(S::TervStorage)
    return keys(S.data)
end


function Base.show(io::IO, t::MIME"text/plain", storage::TervStorage)
    data = storage.data
    if isa(data, AbstractDict)
        println("TervStorage (mutable) with fields:")
    else
        println("TervStorage (immutable) with fields:")
    end
    for key in keys(data)
        println("  $key: $(typeof(data[key]))")
    end
end

function Base.show(io::IO, t::TervStorage, storage::TervStorage)
    data = storage.data
    if isa(data, AbstractDict)
        println("TervStorage (mutable) with fields:")
    else
        println("TervStorage (immutable) with fields:")
    end
    for key in keys(data)
        println("  $key: $(typeof(data[key]))")
    end
end
