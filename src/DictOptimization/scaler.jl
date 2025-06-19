function apply_scaler(x::Real, s::Symbol)
    if s == :log
        x = log(x + 1e-20)
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
        x = exp(x) - 1e-20
    elseif s == :standard_log
        x = exp(x)
    elseif s == :log10
        x = 10^x
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
