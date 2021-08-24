"""
Transfer v to the representation expected by a given context.

For the defalt context, the transfer function does nothing. For other
context such as the CUDA version, it may convert integers and floats
to other types (e.g. Float32) and Arrays to CuArrays.

You will likely have to implement some transfer operators for your own
types if you want to simulate with a non-default context.
"""
function transfer(context, v)
    return v
end

function transfer(context, v::Real)
    return transfer(context, [v])
end

function transfer(context::TervContext, v::AbstractArray)
    return Array(v)
end

function transfer(context::SingleCUDAContext, v::AbstractArray)
    return CuArray(v)
end

function transfer(context::SingleCUDAContext, v::AbstractArray{I}) where {I<:Integer}
    return CuArray{context.index_t}(v)
end

function transfer(context::SingleCUDAContext, v::AbstractArray{F}) where {F<:AbstractFloat}
    return CuArray{context.float_t}(v)
end

function transfer(context::SingleCUDAContext, v::SparseMatrixCSC)
    return CUDA.CUSPARSE.CuSparseMatrixCSC(v)
end

function transfer(context, t::NamedTuple)
    k = keys(t)
    v = map((x) -> transfer(context, x), values(t))

    return (; zip(k, v)...)
end

function transfer(context, t::AbstractDict)
    t = copy(t)
    for (key, val) in t
        t[key] = transfer(context, val)
    end
    return t
end

function transfer(context, t::AbstractFloat)
    convert(float_type(context), t)
end

function transfer(context, t::Integer)
    convert(index_type(context), t)
end

"""
Synchronize backend after allocations.

Some backends may require notification that
storage has been allocated.
"""
function synchronize(::TervContext)
    # Default: Do nothing
end

function float_type(context)
    return Float64
end

function index_type(context)
    return Int64
end

function synchronize(::SingleCUDAContext)
    # Needed because of an issue with kernel abstractions.
    CUDA.synchronize()
end

# For many GPUs we want to use single precision. Specialize interfaces accordingly.
function float_type(c::SingleCUDAContext)
    return c.float_t
end

function index_type(c::SingleCUDAContext)
    return c.index_t
end
