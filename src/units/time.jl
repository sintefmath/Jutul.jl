# Time
function si_unit(::Val{:day})
    return 60.0*60.0*24.0
end

function si_unit(::Val{:minute})
    return 60.0
end

function si_unit(::Val{:hour})
    return 60.0*60.0
end

function si_unit(::Val{:year})
    return 365.2425 * si_unit(Val(:day))
end

function si_unit(::Union{Val{:second}, Val{:s}})
    return 1.0
end

function si_unit(::Val{:psi})
    return si_unit(:lbf)/(si_unit(:inch)^2)
end

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
