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
