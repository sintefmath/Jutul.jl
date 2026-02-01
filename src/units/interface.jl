function convert_to_si(value, unit_name::String)
    ssym = Val(Symbol(unit_name))
    if ssym isa Jutul.CelsiusType || ssym isa Jutul.FahrenheitType
        ret = convert_to_si(value, ssym)
    else
        ret = value*si_unit(unit_name)
    end
    return ret
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
representation of the unit with the given name. Composite units are also supported via
strings, e.g. `si_unit("feet^3/second")`.

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
    return unit_convert(uname, to_si = true)
end

function si_unit(::Val{uname}) where uname
    prefix = get(UNIT_PREFIXES, uname, missing)
    if ismissing(prefix)
        # Could be a prefixed unit, e.g. millimeter
        base_unit = missing
        prefix_value = missing
        for (k, v) in pairs(UNIT_PREFIXES)
            ustr = string(uname)
            prefix_str = string(k)
            if startswith(ustr, prefix_str)
                base_unit = Symbol(replace(ustr, prefix_str => ""))
                prefix_value = v
                break
            end
        end
        if ismissing(base_unit)
            error("Unknown unit: $uname")
        else
            return prefix_value * si_unit(Val(base_unit))
        end
    end
    return prefix
end

function unit_convert(s::String; to_si::Bool = true)
    ex = Meta.parse(s)
    if ex isa Symbol
        ret = si_unit(ex)
    elseif ex isa Number
        ret = ex
    else
        ret = unit_convert(ex, to_si = to_si)
    end
    return ret
end

function unit_convert(ex::Expr; to_si::Bool = true)
    ops = (:/, :*, :^, :+, :-)
    for (idx, val) in enumerate(ex.args)
        if val in ops
            continue
        elseif val isa Number
            continue
        elseif val isa Symbol
            if Val(val) isa Jutul.CelsiusType || Val(val) isa Jutul.FahrenheitType
                msg = """
                "Cannot convert relative temperature units ($val) in expressions.
                Use convert_to_si and convert_from_si functions to convert the temperature unit to absolute units (Rankine or Kelvin) before using it in a composite expression.
                """
                error(msg)
            end
            u = si_unit(val)
            if to_si
                f = u
            else
                f = 1.0/u
            end
            ex.args[idx] = f
        elseif val isa Expr
            ex.args[idx] = unit_convert(val; to_si = to_si)
        else
            error("Unknown expression part: $val")
        end
    end
    return eval(ex)
end

export @si_str

"""
    si"kg/meter"

A string macro to convert unit strings to SI conversion factors. Convenience
function for `si_unit`.

# Examples
```jldoctest
julia> si"day" # Get days represented as seconds
86400.0
julia> si"kilometer/hour" # Get kilometers per hour represented as seconds
0.2777777777777778
julia> si"3.14*kilometer/hour" # Get 3.14 kilometers per hour represented as seconds
0.8722222222222222
julia> si"100gram + 0.1kilogram" # Get 100 grams + 0.1 kilogram represented as kilograms
0.2
```
"""
macro si_str(p)
    return si_unit(p)
end
