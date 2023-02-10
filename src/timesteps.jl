export TimestepSelector, IterationTimestepSelector, VariableChangeTimestepSelector

abstract type AbstractTimestepSelector end

pick_first_timestep(sel, sim, config, dT) = min(dT*initial_relative(sel), initial_absolute(sel))
pick_next_timestep(sel, sim, config, dt_prev, dT, reports, current_reports, step_index, new_step) = dt_prev*increase_factor(sel)

pick_cut_timestep(sel, sim, config, dt, dT, reports, cut_count) = dt

decrease_factor(sel) = 2.0
increase_factor(sel) = Inf
initial_relative(sel) = 1.0
initial_absolute(sel) = Inf
maximum_timestep(sel) = Inf
minimum_timestep(sel) = 1e-20

valid_timestep(sel, dt) = min(max(dt, minimum_timestep(sel)), maximum_timestep(sel))

struct TimestepSelector <: AbstractTimestepSelector
    init_rel
    init_abs
    decrease
    increase
    max
    min
    function TimestepSelector(factor = Inf; decrease = 2.0, initial_relative = 1.0, initial_absolute = Inf, max = Inf, min = 0.0)
        if isnothing(decrease)
            decrease = factor
        end
        new(initial_relative, initial_absolute, decrease, factor, max, min)
    end
end

decrease_factor(sel::TimestepSelector) = sel.decrease
increase_factor(sel::TimestepSelector) = sel.increase
initial_relative(sel::TimestepSelector) = sel.init_rel
initial_absolute(sel::TimestepSelector) = sel.init_abs
maximum_timestep(sel::TimestepSelector) = sel.max
minimum_timestep(sel::TimestepSelector) = sel.min

function pick_cut_timestep(sel::TimestepSelector, sim, config, dt, dT, reports, cut_count)
    df = decrease_factor(sel)
    max_cuts = config[:max_timestep_cuts]
    if cut_count + 1 > max_cuts && dt <= dT/(df^max_cuts)
        dt = NaN
    else
        dt = dt/df
    end
    return dt
end

struct IterationTimestepSelector <: AbstractTimestepSelector
    target
    offset
    function IterationTimestepSelector(target_its = 5; offset = 1)
        @assert offset > 0
        new(target_its, offset)
    end
end

function pick_next_timestep(sel::IterationTimestepSelector, sim, config, dt_prev, dT, reports, current_reports, step_index, new_step)
    if new_step
        R = reports[step_index-1][:ministeps]
    else
        R = current_reports
    end
    r = R[end]
    # Previous number of iterations
    its_p = length(r[:steps]) - 1
    # Target
    its_t, ϵ = sel.target, sel.offset
    # Assume relationship between its and dt is linear (lol)
    return dt_prev*(its_t + ϵ)/(its_p + ϵ)
end

struct VariableChangeTimestepSelector <: AbstractTimestepSelector
    key::Symbol
    model_key::Union{Symbol, Nothing}
    target::Float64
    is_rel::Bool
    reduction::Symbol
    function VariableChangeTimestepSelector(key, target; model = nothing, relative = true, reduction = :max)
        @assert reduction == :max || reduction == :average
        new(key, model, target, relative, reduction)
    end
end

function pick_next_timestep(sel::VariableChangeTimestepSelector, sim, config, dt_prev, dT, reports, current_reports, step_index, new_step)
    if new_step
        R = reports[step_index-1][:ministeps]
    else
        R = current_reports
    end
    m = sel.model_key
    k = sel.key
    function objective(stats)
        if sel.reduction == :max
            dx = stats.dx.max
            x = stats.x.max
        else
            # average
            N = stats.n
            dx = stats.dx.sum/N
            x = stats.x.sum/N
        end
        if sel.is_rel
            obj = dx/x
        else
            obj = dx
        end
        return obj
    end
    if isnothing(m)
        dt_info = x -> (x[:dt], x[:post_update][k])
    else
        dt_info = x -> (x[:dt], x[:post_update][m][k])
    end

    r = R[end]
    dt1, stats1 = dt_info(r)
    if length(R) > 1
        dt0, stats0 = dt_info(R[end-1])
    else
        dt0, stats0 = dt1, stats1
    end
    x1 = objective(stats1)
    x0 = objective(stats0)
    x = sel.target
    return linear_timestep_selection(x, x0, x1, dt0, dt1)
end

function linear_timestep_selection(x, x0, x1, dt0, dt1)
    if x1 != x0
        # Linear approximation
        dt_next = dt0 + (x - x0)*(dt1 - dt0)/(x1 - x0)
    else
        # Fallback for missing / degenerate data
        dt_next = x*dt1/x1
    end
    return dt_next
end
