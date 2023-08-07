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
    return 1.66053904020e-27
end

function si_unit(::Val{:darcy})
    return 9.869232667160130e-13
end
