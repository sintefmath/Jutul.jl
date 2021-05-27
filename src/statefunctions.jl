# TervStateFunction

struct SecondaryVariableAsStateFunction{} <: TervStateFunction end
struct PrimaryVariableAsStateFunction{} <: TervStateFunction end


function evaluate!(storage, model, tv::TervVariables)
    sf_storage = storage.state_functions
    parameters = storage.parameters
    s = get_symbol(tv)
    evaluate!(sf_storage[s], sf_storage, parameters, model, tv)
end

function evaluate(v, storage, param, model, sf::TervVariables)
    @assert false
end

# Initializers
function select_state_functions(domain, system, formulation)
    sf = OrderedDict()
    select_state_functions!(sf, domain, system, formulation)
    return sf
end

function select_state_functions!(sf, domain, system, formulation)
    select_state_functions!(sf, system)
end

function select_state_functions!(sf, system)

end

function allocate_state_functions!(sf_storage, storage, model; tag = nothing)
    for (sym, sf) in model.state_functions
        u = associated_unit(sf)
        n = degrees_of_freedom_per_unit(model, u)
        sf_storage[sym] = allocate_state_function(model, sf; npartials = n, tag = tag)
    end
end

function allocate_state_function(model, sf; kwarg...)
    dim = value_dim(model, sf)
    allocate_array_ad(dim...; kwarg...)
end