# Energy
function si_unit(::Union{Val{:joule}, Val{:J}})
    N, m = si_units(:newton, :meter)
    return N*m
end

function si_unit(::Union{Val{:btu}, Val{:BTU}, Val{Symbol("British thermal unit")}})
    return 1054.3503
end
