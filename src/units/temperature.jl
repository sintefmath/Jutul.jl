# Temperature

# Need special dispatches for these two
const CelsiusType = Union{Val{:Celsius}, Val{:celsus}, Val{:degC}, Val{Symbol("°C")}}
const FahrenheitType = Union{Val{:Fahrenheit}, Val{:fahrenheit }, Val{:degF}, Val{Symbol("°F")}}

function si_unit(::Union{Val{:Kelvin}, Val{:kelvin}, Val{:K}})
    return 1.0
end

function si_unit(::Union{Val{:Rankine}, Val{:rankine}, Val{:R}})
    return 5.0/9.0
end

function convert_to_si(val, unit::FahrenheitType)
    return (val - 32.0) * (5.0 / 9.0) + 273.15
end

function convert_to_si(val, unit::CelsiusType)
    return val + 273.15
end

function convert_from_si(val, unit::FahrenheitType)
    return (val - 273.15)*9/5 + 32.0
end

function convert_from_si(val, unit::CelsiusType)
    return val - 273.15
end
