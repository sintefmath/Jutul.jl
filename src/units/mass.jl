# Mass
function si_unit(::Union{Val{:pound}, Val{:lb}})
    return 0.45359237 # kg
end

function si_unit(::Union{Val{:kilogram}, Val{:kg}})
    return 1.0
end

function si_unit(::Union{Val{:gram}, Val{:g}})
    return 1e-3
end

function si_unit(::Val{:tonne})
    return 1000.0
end
