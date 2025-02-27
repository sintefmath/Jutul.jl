function pick_timestep(sim, config, dt_prev, dT, forces, reports, current_reports; step_index = NaN, new_step = false, remaining_time = dT)
    # Try to do the full step, unless one of our selectors/limits tells us otherwise.
    # No need to limit to whatever remains of the interval since that is fixed on the outside.
    dt = dT
    selectors = config[:timestep_selectors]
    is_first = new_step && step_index == 1
    if is_first
        for sel in selectors
            candidate = pick_first_timestep(sel, sim, config, dT, forces)
            dt = min(dt, candidate)
        end
    else
        for sel in selectors
            candidate = pick_next_timestep(sel, sim, config, dt_prev, dT, forces, reports, current_reports, step_index, new_step)
            dt = min(dt, candidate)
        end
        # The selectors might go crazy, so we have some safety bounds
        min_allowable = config[:timestep_max_decrease]*dt_prev
        max_allowable = config[:timestep_max_increase]*dt_prev
        dt = clamp(dt, min_allowable, max_allowable)
    end
    # Make sure that the final timestep is still within the limits of all selectors
    for sel in selectors
        dt = valid_timestep(sel, dt)
    end
    # If we are not going to reach the end anyway, we split the remaining part
    # into two to avoid a long step and a very short step.
    half_remain = remaining_time/2.0
    if dt > half_remain && dt < remaining_time
        dt = half_remain
    end
    dt = clamp(dt, config[:min_timestep], config[:max_timestep])
    if config[:info_level] > 1
        ratio = dt/dt_prev
        if ratio > 5
            t_sym = "⏫"
        elseif ratio > 1.1
            t_sym = "🔼"
        elseif ratio < 0.2
            t_sym = "⏬"
        elseif ratio < 0.9
            t_sym = "🔽"
        else
            t_sym = "🔄"
        end
        jutul_message("Next mini-step", "Δt = $(get_tstr(dt)) from previous $(get_tstr(dt_prev)) $t_sym", color = :default)
    end
    return dt
end

function cut_timestep(sim, config, dt, dT, forces, reports; step_index = NaN, cut_count = 0)
    if isapprox(dt, config[:min_timestep])
        return NaN
    end
    for sel in config[:timestep_selectors]
        candidate = pick_cut_timestep(sel, sim, config, dt, dT, forces, reports, cut_count)
        dt = min(dt, candidate)
    end
    return dt
end