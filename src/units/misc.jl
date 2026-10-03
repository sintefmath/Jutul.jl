function si_unit(::Val{:site})
    return 6.00221413e23^-1
end

function si_unit(::Union{Val{:gal}, Val{:Gal}})
    return 0.01
end

function si_unit(::Val{:mol})
    return 1.0
end

function si_unit(::Union{Val{:dalton}, Val{:Da}})
    return 1.6605390402e-27
end

function si_unit(::Val{:darcy})
    return 9.86923266716013e-13
end
