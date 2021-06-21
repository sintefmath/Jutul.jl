
# Transfer operators
function transfer(context, v::Real)
    return transfer(context, [v])
end

function transfer(context::TervContext, v::AbstractArray)
    return Array(v)
end

function transfer(context::SingleCUDAContext, v::AbstractArray)
    return CuArray(v)
end

function transfer(context, v)
    return v
end

function transfer(context::SingleCUDAContext, v::AbstractArray{I}) where {I<:Integer}
    return CuArray{context.index_t}(v)
end

function transfer(context::SingleCUDAContext, v::AbstractArray{F}) where {F<:AbstractFloat}
    return CuArray{context.float_t}(v)
end

function transfer(context, t::NamedTuple)
    k = keys(t)
    v = map((x) -> transfer(context, x), values(t))

    return (; zip(k, v)...)
end

function transfer(context, t::AbstractFloat)
    convert(float_type(context), t)
end

function transfer(context, t::Integer)
    convert(index_type(context), t)
end

"Synchronize backend after allocations if needed"
function synchronize(::TervContext)
    # Do nothing
end

function float_type(c::TervContext)
    return Float64
end

function index_type(c::TervContext)
    return Int64
end

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
