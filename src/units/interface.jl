"""
    convert_to_si(value, unit_name::String)

Convert `value` to SI representation from value in the unit given by `unit_symbol`.


# Examples
```jldoctest
julia> convert_to_si(1.0, :hour) # Get 1 hour represented as seconds
3600.0
```
"""
function convert_to_si(value, unit_name::String)
    return convert_to_si(value, Symbol(unit_name))
end

function convert_to_si(value, unit_name::Symbol)
    return convert_to_si(value, Val(unit_name))
end

function convert_to_si(val, unit::Val)
    return val*si_unit(unit)
end

function convert_to_si(val, unit::Real)
    return val*unit
end

"""
    convert_from_si(value, unit_name::Union{Symbol, String})

Convert `value` from SI representation to the unit in `unit_symbol`.

# Examples
```jldoctest
julia> convert_from_si(3600.0, :hour) # Get 3600 s represented as hours
1.0
```
"""
function convert_from_si(value, unit_name::String)
    return convert_from_si(value, Symbol(unit_name))
end

function convert_from_si(value, unit_name::Symbol)
    return convert_from_si(value, Val(unit_name))
end

function convert_from_si(val, unit::Val)
    return val/si_unit(unit)
end

function convert_from_si(val, unit::Real)
    return val/unit
end

"""
    u1_val = si_units(u1)
    meter = si_units(:meter)
    meter, hour = si_units(:meter, :hour)

Get multiplicative SI unit conversion factors for multiple units simultaneously.
The return value will be a `Tuple` of values, one for each input argument. Each
input arguments can be either a `String` a `Symbol`.

# Examples
```jldoctest
julia> meter, hour = si_units(:meter, :hour)
(1.0, 3600.0)
```
"""
function si_units(arg...)
    return map(si_unit, arg)
end

"""
    si_unit(u::Union{String, Symbol})

Get the multiplicative SI unit conversion factor for a single unit. The return
value is given so that `x*si_unit(:name)` will convert `x` to the SI
representation of the unit with the given name.

# Examples
```jldoctest
julia> si_unit(:day) # Get days represented as seconds
86400.0
```
"""
function si_unit(uname::Symbol)
    return si_unit(Val(uname))::Float64
end

function si_unit(uname::String)
    return si_unit(Symbol(uname))
end

function si_unit(::Val{uname}) where uname
    prefix = get(UNIT_PREFIXES, uname, missing)
    if ismissing(prefix)
        error("Unknown unit: $uname")
    end
    return prefix
end

function all_units()
    d = Dict{Symbol, Float64}()
    for k in available_units()
        d[k] = si_unit(k)
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
