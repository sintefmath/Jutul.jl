export convert_from_si, convert_to_si, si_unit, si_units

include("internals.jl")

# Specific units follows
include("electrochemistry.jl")
include("energy.jl")
include("force.jl")
include("length.jl")
include("mass.jl")
include("misc.jl")
include("pressure.jl")
include("temperature.jl")
include("time.jl")
include("viscosity.jl")
include("volume.jl")
# Main interface
include("interface.jl")

const TIME_UNITS_FOR_PRINTING = (
    (si_unit(:year), :year),
    (7*si_unit(:day), :week),
    (si_unit(:day), :day),
    (si_unit(:hour), :hour),
    (si_unit(:minute), :minute),
    (si_unit(:second), :second),
    (si_unit(:milli)*si_unit(:second), :millisecond),
    (si_unit(:micro)*si_unit(:second), :microsecond),
    (si_unit(:nano)*si_unit(:second), :nanosecond),
)


"""
    convert_to_si(value, unit_name::String)

Convert `value` to SI representation from value in the unit given by `unit_symbol`.

# Available units
You can get a list of all available units via `Jutul.available_units()`. The
values in Jutul itself are:

$(join(sort(collect(keys(Jutul.all_units(prefix = false)))), ", ")).

In addition units can be prefixed with standard SI prefixes. Available prefixes are:

$(join([string(k) for k in keys(Jutul.UNIT_PREFIXES)], ", ")).

This utility can also handle composite units, e.g. `"kilometer/hour"` or
`"meter/second^2"`. Note that relative temperature units (Celsius and Fahrenheit) must be
converted to absolute units (Kelvin or Rankine) before being used in composite units.

# Examples
```jldoctest
julia> convert_to_si(1.0, :hour) # Get 1 hour represented as seconds
3600.0
julia> convert_to_si(5.0, "kilometer/hour") # Get 5 kilometers per hour represented as seconds
1.3888888888888888
julia> convert_to_si(1.0, "milligram") # Get 1 milligram represented as kilograms
1.0e-6
```
"""
function convert_to_si
    # Place docs here to avoid docstring duplication
end
