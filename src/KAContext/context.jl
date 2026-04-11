# Context interface overrides for KernelAbstractionsContext

Jutul.float_type(c::KernelAbstractionsContext) = c.float_t
Jutul.index_type(c::KernelAbstractionsContext) = c.index_t

"""
    device_array_type(backend, T, dims...)

Allocate a device array of element type `T` on `backend`.
"""
function device_array(backend, val::AbstractArray)
    return KernelAbstractions.allocate(backend, eltype(val), size(val)...)
end

function to_device(backend, val::AbstractArray)
    d = KernelAbstractions.allocate(backend, eltype(val), size(val)...)
    copyto!(d, val)
    return d
end

to_device(::Any, val) = val
