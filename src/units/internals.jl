
const UNIT_PREFIXES = (
    quetta = 1e30,
    ronna  = 1e27,
    yotta  = 1e24,
    zetta  = 1e21,
    exa    = 1e18,
    peta   = 1e15,
    tera   = 1e12,
    giga   = 1e9,
    mega   = 1e6,
    kilo   = 1e3,
    hecto  = 1e2,
    deca   = 1e1,
    deci   = 1e-1,
    centi  = 1e-2,
    milli  = 1e-3,
    micro  = 1e-6,
    nano   = 1e-9,
    pico   = 1e-12,
    femto  = 1e-15,
    atto   = 1e-18,
    zepto  = 1e-21,
    yocto  = 1e-24,
    ronto  = 1e-27,
    quecto = 1e-30
)

function all_units(; prefix = true)
    d = Dict{Symbol, Float64}()
    for k in available_units()
        d[k] = si_unit(k)
    end
    if prefix
        for (k, v) in pairs(UNIT_PREFIXES)
            d[k] = v
        end
    end
    return d
end

function available_units()
    unpack(::Val{T}) where T = [T]
    unpack(::Any) = []
    unpack(::Type{Tuple{V, T}}) where {V, T} = unpack(T)

    function unpack(::Type{T}) where T
        out = []
        if T isa Union
            push!(out, T.a)
            append!(out, unpack(T.b))
        else
            push!(out, T)
        end
        return out
    end
    unpack_val(x) = nothing
    unpack_val(::Type{Val{T}}) where T = T

    retval = Symbol[]
    for m in methods(si_unit)
        for el in unpack(m.sig)
            v = unpack_val(el)
            if v isa Symbol
                push!(retval, v)
            end
        end
    end
    return retval
end
