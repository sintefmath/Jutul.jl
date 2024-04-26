function set_global_timer!(enabled = true)
    if enabled
        if !isdefined(Jutul, :timeit_debug_enabled)
            @tic "tmp" sleep(0.001)
        end
        TimerOutputs.enable_debug_timings(Jutul)
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
            jutul_message(text, "If empty, the timer was enabled but not compiled. Re-run and it should show.")
        end
        print_timer()
    end
end

print_global_timer(::Nothing) = nothing

function start_simulation_message(info_level, timesteps, config)
    p = nothing
    n = length(timesteps)
    msg = "Simulating $(get_tstr(sum(timesteps), 2)) as $n report steps"
    if info_level > 0
        jutul_message("Jutul", msg, color = :light_green)
    end
    if info_level == 0
        bg = config[:progress_glyphs]
        if bg isa Symbol
            if bg == :default
                bg = missing
            elseif bg == :futuristic
                bg = BarGlyphs(' ','▰', '▰', '▱',' ',)
            elseif bg == :thin
                bg = BarGlyphs(' ','━', '╸', ' ',' ',)
            elseif bg == :dotted
                bg = BarGlyphs(' ','█', '▒', '░',' ',)
            else
                error("Unknown option $bg for glyphs")
            end
        end
        if ismissing(bg)
            arg = NamedTuple()
        else
            arg = (barglyphs = bg)
        end
        p = Progress(n+1;
            desc = msg,
            dt = 0.1,
            color = config[:progress_color],
            arg...
        )
    end
    return p
end

function new_simulation_control_step_message(info_level, p, rec, elapsed, step_no, no_steps, dT, t_tot, start_date)
    if info_level == 0
        msgvals = progress_showvalues(rec, elapsed, step_no, no_steps, dT, t_tot, start_date)
        next!(p; showvalues = msgvals)
    elseif info_level > 0
        r = rec.recorder

        count_str = "$no_steps"
        ndig = length(count_str)
        fstr = lpad("$step_no", ndig)
        t = r.time
        t_now = t + dT
        if isnothing(start_date)
            fmt = x -> get_tstr(x, 2)
        else
            fmt = x -> Dates.format(start_date + Microsecond(ceil(x*1e6)), raw"u. dd Y")
        end
        start_time = fmt(t)
        end_time = fmt(t_now)

        jutul_message("Step $fstr/$count_str", "Solving $start_time to $end_time, Δt = $(get_tstr(dT, 2)) ", color = :blue)
        # @info "$(prefix)Solving step $step_no/$no_steps of length $(get_tstr(dT))."
    end
end

function progress_showvalues(rec, elapsed, step_no, no_steps, dT, t_tot, start_date)
    r = rec.recorder
    frac = (r.time + dT)/t_tot
    perc = @sprintf("%2.2f", 100*frac)
    its = rec.recorder.iterations + rec.subrecorder.iterations
    elapsed_each = elapsed/its
    done = step_no == no_steps + 1
    if done
        msg_status = "Solved step $no_steps/$no_steps"
    else
        msg_status = "Solving step $step_no/$no_steps ($perc% of time interval complete)"
    end
    msg_timing = "$its iterations in $(autoformat_time(elapsed)) ($(autoformat_time(elapsed_each)) each)"
    # msg_timing = "$(autoformat_time(elapsed)) elapsed ($(autoformat_time(elapsed/(step_no-1))))"

    msgvals = [
        (:Progress, msg_status),
        (:Stats, msg_timing)
        ]

    if !isnothing(start_date)
        t_format = raw"u. dd YY"
        push!(msgvals, (:Date, "$(Dates.format(start_date + Microsecond(ceil(r.time*1e6)), t_format))"))
    end
    return msgvals
end

function final_simulation_message(simulator, p, rec, t_elapsed, reports, timesteps, config, start_date, aborted)
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
            start_str = "Simulation aborted"
            endstr = "$(length(reports)) of $(length(timesteps))"
            str_c = :red
        else
            start_str = "Simulation complete"
            endstr = "$(stats.steps)"
            str_c = :light_green
        end
        t_tot = stats.time_sum.total
        final_message = "Completed $endstr report steps in $(get_tstr(t_tot)) and $(stats.newtons) iterations."
        if info_level == 0
            if aborted
                cancel(p, "$start_str $final_message")
            else
                n = length(timesteps)
                msgvals = progress_showvalues(rec, t_elapsed, n+1, n, 0.0, t_tot, start_date)
                next!(p; showvalues = msgvals)
                finish!(p)
            end
        else
            jutul_message(start_str, final_message, color = str_c, underline = true)
        end
    elseif info_level == 0
        cancel(p)
    end
    # Optional table of performance numbers etc.
    if print_end_report
        n = simulator_reports_per_step(simulator)
        print_stats(stats,
            table_formatter = config[:table_formatter],
            scale = n
        )
    end
    # Detailed timing through @tic instrumentation (can be a lot)
    print_global_timer(config[:extra_timing])
    if aborted
        msg = "Simulation did not complete successfully."
        if config[:error_on_incomplete]
            error(msg)
        elseif verbose
            @error(msg)
        end
    end
end

export jutul_message
function jutul_message(prestr, substr = nothing; color = :light_blue, kwarg...)
    if isnothing(substr)
        fmt = "$prestr"
        substr = ""
    else
        fmt = "$prestr:"
        substr = " $substr"
    end
    print(Crayon(foreground = color, bold = true; kwarg...), fmt)
    println(Crayon(reset = true), substr)
end

function simulator_reports_per_step(simulator)
    return 1
end
