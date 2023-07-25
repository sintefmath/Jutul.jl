# Length
function si_unit(::Union{Val{:meter}, Val{:m}})
    return 1.0
end

function si_unit(::Union{Val{:inch}, Val{:in}})
    centi, meter = si_units(:centi, :meter)
    return 2.54*centi*meter
end

function si_unit(::Union{Val{:feet}, Val{:ft}})
    return 0.3048
end
