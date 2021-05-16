
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
