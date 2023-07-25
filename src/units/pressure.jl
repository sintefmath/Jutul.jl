# Pressure
function si_unit(::Union{Val{:pascal}, Val{:Pa}})
    return 1.0
end

function si_unit(::Union{Val{:atmosphere}, Val{:atm}})
    return 101325.0
end

function si_unit(::Val{:bar})
    return 1e5
end

