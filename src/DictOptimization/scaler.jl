function apply_scaler(x::Real, s::Symbol)
    if s == :log
        base = 1e5
        x = log((base-1)*x + 1)/log(base)
    elseif s == :exp
        base = 1e5
        x = (base^x - 1)/(base - 1)
    elseif s == :standard_log
        x = log(x)
    elseif s == :log10
        x = log10(x)
    else
        error("Unknown scaler $s")
    end
    return x
end

function apply_scaler(x, s::Missing)
    return x
end

function apply_scaler(x::AbstractArray, s::Symbol)
    return map(v -> apply_scaler(v, s), x)
end

function undo_scaler(x::Real, s::Symbol)
    if s == :log
        base = 1e5
        x = (base^x - 1)/(base - 1)
    elseif s == :exp
        base = 1e5
        x = log((base-1)*x + 1)/log(base)
    elseif s == :standard_log
        x = exp(x)
    elseif s == :log10
        x = 10^x
    elseif s == :reciprocal
        x = 1.0/x - 1e-20
    else
        error("Unknown scaler $s")
    end
    return x
end

function undo_scaler(x, s::Missing)
    return x
end

function undo_scaler(x::AbstractArray, s::Symbol)
    return map(v -> undo_scaler(v, s), x)
end
