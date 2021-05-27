# TervStateFunction



# In some settings, variables also act like state functions
function evaluate(storage, model, tv)
    s = get_symbol(tv)
    evaluate(storage.state[s], storage, model, tv)
end

function evaluate(value, storage, model, sf::TervStateFunction)
    error("Not implemented for $sf")
end

function evaluate(value, storage, model, sf::TervVariables)
    error("Not implemented for $sf")
end
