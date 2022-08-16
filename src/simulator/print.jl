function set_global_timer!(enabled = true)
    if enabled
        enable_timer!()
        reset_timer!()
    else
        disable_timer!()
    end
end

set_global_timer!(::Nothing) = nothing

function print_global_timer(do_print = true; text = "Detailed timing")
    if do_print
        if !isnothing(text)
            @info text
        end
        print_timer()
    end
end

print_global_timer(::Nothing) = nothing

function start_simulation_message(info_level, timesteps)
    p = nothing
    n = length(timesteps)
    msg = "Simulating $n steps... "
    if info_level > 0
        @info msg
    end
    if info_level == 0
        p = Progress(n, dt = 0.5, desc = msg)
    end
    return p
end

function new_simulation_control_step_message(info_level, p, rec, step_no, no_steps, dT, t_tot)
    if info_level == 0
        r = rec.recorder
        frac = (r.time + dT)/t_tot
        perc = @sprintf("%2.2f", 100*frac)
        msg = "Solving step $step_no/$no_steps ($perc% of time interval complete)"
        next!(p; showvalues = [(:Status, msg)])
    elseif info_level > 0
        if info_level == 1
            prefix = ""
        else
            prefix = "🟢 "
        end
        @info "$(prefix)Solving step $step_no/$no_steps of length $(get_tstr(dT))."
    end
end

function final_simulation_message(simulator, p, reports, timesteps, config, aborted)
    info_level = config[:info_level]
    print_end_report = config[:end_report]
    verbose = info_level >= 0
    if verbose || print_end_report
        stats = report_stats(reports)
    else
        stats = nothing
    end
    # Summary message.
    if verbose && length(reports) > 0
        if aborted
            endstr = "🔴 Simulation aborted. Completed $(stats.steps) of $(length(timesteps)) timesteps"
        else
            endstr = "✅ Simulation complete. Completed $(stats.steps) timesteps"
        end
        t_tot = stats.time_sum.total
        final_message = "$endstr in $(get_tstr(t_tot)) seconds with $(stats.newtons) iterations."
        if info_level == 0
            if aborted
                cancel(p, final_message)
            else
                finish!(p)
            end
        else
            @info final_message
        end
    elseif info_level == 0
        cancel(p)
    end
    # Optional table of performance numbers etc.
    if print_end_report
        print_stats(stats, table_formatter = config[:table_formatter])
    end
    # Detailed timing through @timeit instrumentation (can be a lot)
    print_global_timer(config[:extra_timing])
    if aborted && config[:error_on_incomplete]
        error("Simulation did not complete successfully.")
    end
end


