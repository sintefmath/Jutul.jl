export convert_from_si, convert_to_si, si_unit, si_units

include("interface.jl")
include("prefix.jl")
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

function available_units()
    # TODO: Clean up this internal helper so that it only provides the full
    # names
    return [
        :pascal,
        :atm,
        :bar,
        :newton,
        :dyne,
        :lbf,
        :liter,
        :stb,
        :gallon_us,
        :pound,
        :kg,
        :g,
        :tonne,
        :meter,
        :inch,
        :feet,
        :day,
        :hour,
        :year,
        :second,
        :psi,
        :btu,
        :kelvin,
        :rankine,
        :joule,
        :farad,
        :ampere,
        :watt,
        :site,
        :poise,
        :gal,
        :mol,
        :dalton,
        :darcy,
        :quetta,
        :ronna,
        :yotta,
        :zetta,
        :exa,
        :peta,
        :tera,
        :giga,
        :mega,
        :kilo,
        :hecto,
        :deca,
        :deci,
        :centi,
        :milli,
        :micro,
        :nano,
        :pico,
        :femto,
        :atto,
        :zepto,
        :yocto,
        :ronto,
        :quecto
    ]
end