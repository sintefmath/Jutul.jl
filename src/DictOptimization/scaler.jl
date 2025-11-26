function apply_scaler(x::Real, bnds::LimitBounds, stats, s::Symbol)
    if s == :log
        x = (x - bnds.min_group)/(bnds.max_group - bnds.min_group)
        base = clamp(bnds.max_group/bnds.min_group, 1.01, 1e4)
        x = log((base-1)*x + 1)/log(base)
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
        x = (x - bnds.min)/(bnds.max - bnds.min)
    elseif s == :linear_limits_group
        x = (x - bnds.min_group)/(bnds.max_group - bnds.min_group)
    elseif s == :linear
        minv, maxv, = stats_bounds(stats)
        x = (x - minv)/(maxv - minv)
    else
        error("Unknown scaler $s")
    end
    return x
end

function apply_scaler(x, limit_bounds::LimitBounds, stats, s::Missing)
    return x
end

function undo_scaler(x::Real, bnds::LimitBounds, stats, s::Symbol)
    if s == :log
        base = clamp(bnds.max_group/bnds.min_group, 1.01, 1e4)
        x = (base^x - 1)/(base - 1)
        x = x*(bnds.max_group - bnds.min_group) + bnds.min_group
    elseif s == :exp
        base = 1e5
        x = log((base-1)*x + 1)/log(base)
    elseif s == :standard_log
        x = exp(x)
    elseif s == :log10
        x = 10^x
    elseif s == :linear_limits
        x = x*(bnds.max - bnds.min) + bnds.min
    elseif s == :linear_limits_group
        x = x*(bnds.max_group - bnds.min_group) + bnds.min_group
    elseif s == :linear
        minv, maxv, = stats_bounds(stats)
        x = x*(maxv - minv) + minv
    elseif s == :reciprocal
        x = 1.0/x - 1e-20
    else
        error("Unknown scaler $s")
    end
    return x
end

function undo_scaler(x, limit_bounds::LimitBounds, stats, s::Missing)
    return x
end

function apply_scaler(x, limit_bounds::LimitBounds, stats, s::DictOptimizationScaler)
    error("Scaler of type $(typeof(s)) not implemented. You need to implement apply_scaler and undo_scaler for this type.")
end

function undo_scaler(x, limit_bounds::LimitBounds, stats, s::DictOptimizationScaler)
    error("Scaler of type $(typeof(s)) not implemented. You need to implement apply_scaler and undo_scaler for this type.")
end

function stats_bounds(stats, ϵ::Float64 = 1e-12, base_max::Float64 = Inf)
    x_min = stats.min
    x_max = max(stats.max, x_min + ϵ)
    if abs(x_max - x_min) < ϵ || x_min < ϵ
        base = 10000
    end
    base = min(x_max/x_min, base_max)
    return (x_min, x_max, base)
end

function stats_bounds(stats, s::BaseLogScaler)
    return stats_bounds(stats, s.epsilon, s.base_max)
end

function apply_scaler(x, limit_bounds::LimitBounds, stats, s::BaseLogScaler)
    m, M, base = stats_bounds(stats, s.epsilon)
    x = (x - m)/(M - m)
    x = log((base-1)*x + 1)/log(base)
    return x
end

function undo_scaler(x, limit_bounds::LimitBounds, stats, s::BaseLogScaler)
    m, M, base = stats_bounds(stats, s.epsilon)
    x = (base^x - 1)/(base - 1)
    x = x*(M - m) + m
end
