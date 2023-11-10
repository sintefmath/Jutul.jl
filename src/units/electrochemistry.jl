# Electrochemical
function si_unit(::Union{Val{:farad}, Val{:F}})
    return 1.0
end

function si_unit(::Union{Val{:ampere}, Val{:amp}, Val{:A}})
    return 1.0
end

function si_unit(::Union{Val{:watt}, Val{:W}})
    return si_unit(:joule)/si_unit(:second)
end
