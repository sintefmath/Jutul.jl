# Force
function si_unit(::Union{Val{:newton}, Val{:N}})
    return 1.0
end

function si_unit(::Union{Val{:dyne}, Val{:dyn}})
    return 1e-5*si_unit(:newton)
end

function si_unit(::Union{Val{Symbol("pound-force")}, Val{:lbf}})
    meter, second, pound = si_units(:meter, :second, :pound)
    g = 9.80665 * meter/(second^2);
    return pound * g;
end
