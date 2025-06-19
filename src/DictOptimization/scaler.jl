function apply_scaler(x::Real, s::Symbol)
    if s == :log
        x = log(x)
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
        x = exp(x)
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
