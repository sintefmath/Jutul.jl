
# Viscosity
function si_unit(::Val{:poise})
    return 0.1*si_unit(:pascal)*si_unit(:second)
end
