# Volume
function si_unit(::Union{Val{:liter}, Val{:litre}, Val{:L}, Val{:l}})
    return si_unit(:milli)*si_unit(:meter)^3
end

function si_unit(::Union{Val{:stb}, Val{Symbol("Standard barrel")}})
    return 42.0*si_unit(:usgal)
end

function si_unit(::Union{Val{:gallon_us}, Val{:usgal}})
    return 231.0*si_unit(:inch)^3;
end
