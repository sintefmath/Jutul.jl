export discretize_domain
function discretize_domain(d::DataDomain, system, dtype::Symbol = :default; kwarg...)
    return discretize_domain(d, system, Val(dtype); kwarg...)
end

function discretize_domain(d::DataDomain, system, ::Val{:default}; kwarg...)
    pr = physical_representation(d)
    return DiscretizedDomain(pr; kwarg...)
end
