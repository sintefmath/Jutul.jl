const UNIT_PREFIXES = (
    quetta = 1.0e30,
    ronna = 1.0e27,
    yotta = 1.0e24,
    zetta = 1.0e21,
    exa = 1.0e18,
    peta = 1.0e15,
    tera = 1.0e12,
    giga = 1.0e9,
    mega = 1.0e6,
    kilo = 1.0e3,
    hecto = 1.0e2,
    deca = 1.0e1,
    deci = 1.0e-1,
    centi = 1.0e-2,
    milli = 1.0e-3,
    micro = 1.0e-6,
    nano = 1.0e-9,
    pico = 1.0e-12,
    femto = 1.0e-15,
    atto = 1.0e-18,
    zepto = 1.0e-21,
    yocto = 1.0e-24,
    ronto = 1.0e-27,
    quecto = 1.0e-30,
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
    unpack(::Val{T}) where {T} = [T]
    unpack(::Any) = []
    unpack(::Type{Tuple{V, T}}) where {V, T} = unpack(T)

    function unpack(::Type{T}) where {T}
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
    unpack_val(::Type{Val{T}}) where {T} = T

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
