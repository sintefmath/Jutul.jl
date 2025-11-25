function apply_scaler(x::Real, lower_limit, upper_limit, stats, s::Symbol)
    if s == :log
        s = BaseLogScaler()
        x = apply_scaler(x, lower_limit, upper_limit, stats, s)
    elseif s == :exp
        base = 1e5
        x = (base^x - 1)/(base - 1)
    elseif s == :standard_log
        x = log(x)
    elseif s == :log10
        x = log10(x)
    elseif s == :reciprocal
        x = 1.0/(x + 1e-20)
    elseif s == :linear_limits
        x = (x - lower_limit)/(upper_limit - lower_limit)
    else
        error("Unknown scaler $s")
    end
    return x
end

function apply_scaler(x, lower_limit, upper_limit, stats, s::Missing)
    return x
end

function undo_scaler(x::Real, lower_limit, upper_limit, stats, s::Symbol)
    if s == :log
        s = BaseLogScaler()
        x = undo_scaler(x, lower_limit, upper_limit, stats, s)
    elseif s == :exp
        base = 1e5
        x = log((base-1)*x + 1)/log(base)
    elseif s == :standard_log
        x = exp(x)
    elseif s == :log10
        x = 10^x
    elseif s == :linear_limits
        x = x*(upper_limit - lower_limit) + lower_limit
    elseif s == :reciprocal
        x = 1.0/x - 1e-20
    else
        error("Unknown scaler $s")
    end
    return x
end

function undo_scaler(x, lower_limit, upper_limit, stats, s::Missing)
    return x
end

function apply_scaler(x, lower_limit, upper_limit, stats, s::DictOptimizationScaler)
    error("Scaler of type $(typeof(s)) not implemented. You need to implement apply_scaler and undo_scaler for this type.")
end

function undo_scaler(x, lower_limit, upper_limit, stats, s::DictOptimizationScaler)
    error("Scaler of type $(typeof(s)) not implemented. You need to implement apply_scaler and undo_scaler for this type.")
end

function stats_bounds(stats, ϵ::Float64 = 1e-12, base_max::Float64 = Inf)
    x_min = stats.min
    x_max = stats.max
    if abs(x_max - x_min) < ϵ || x_min < ϵ
        base = 10000
    end
    base = min(x_max/x_min, base_max)
    return (x_min, x_max, base)
end

function stats_bounds(stats, s::BaseLogScaler)
    return stats_bounds(stats, s.epsilon, s.base_max)
end

function apply_scaler(x, lower_limit, upper_limit, stats, s::BaseLogScaler)
    m, M, base = stats_bounds(stats, s.epsilon)
    x = (x - m)/(M - m)
    x = log((base-1)*x + 1)/log(base)
end

function undo_scaler(x, lower_limit, upper_limit, stats, s::BaseLogScaler)
    m, M, base = stats_bounds(stats, s.epsilon)
    x = (base^x - 1)/(base - 1)
    x = x*(M - m) + m
end
