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

"""
    si_units(u1, arg...)

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
    return si_unit(Val(uname))
end

function si_unit(uname::String)
    return si_unit(Symbol(uname))
end

function si_unit(::Val{uname}) where uname
    error("Unknown unit: $uname")
end
