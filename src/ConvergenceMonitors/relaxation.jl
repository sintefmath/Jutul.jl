struct ConvergenceMonitorRelaxation <: Jutul.NonLinearRelaxation
    w_min::Float64
    w_max::Float64
    dw_decrease::Float64
    dw_increase::Float64
    warmup_iterations::Int
end

"""
    ConvergenceMonitorRelaxation(; w_min = 0.1, dw = 0.2, dw_increase = nothing, dw_decrease = nothing, w_max = 1.0)

Relaxation strategy based on convergence monitoring. Requires that the
 simulation has been configured with a `ConvergenceMonitorCuttingCriterion` so
that the convergence status is available in the reports. See corresponding
iplementation of `select_nonlinear_relaxation_model` below for details.
"""
function ConvergenceMonitorRelaxation(; w_min = 0.1, dw = 0.2, dw_increase = nothing, dw_decrease = nothing, w_max = 1.0, warmup_iterations = 4)
    if isnothing(dw_increase)
        dw_increase = dw/2
    end
    if isnothing(dw_decrease)
        dw_decrease = dw
    end
    return ConvergenceMonitorRelaxation(w_min, w_max, dw_decrease, dw_increase, warmup_iterations)
end

"""
    set_convergence_monitor_relaxation!(config; max_nonlinear_iterations = 50, convergence_monitor_args = NamedTuple(), relaxation_args...)

Utility for setting `ConvergenceMonitorRelaxation` to the simulator config. This
function also sets the a `ConvergenceMonitorCuttingCriterion` to the config --
see `set_convergence_monitor_cutting_criterion!`.
"""
function set_convergence_monitor_relaxation!(config; 
    max_nonlinear_iterations = 50,
    convergence_monitor_args = NamedTuple(),
    relaxation_args...
    )
    
    set_convergence_monitor_cutting_criterion!(config; 
    max_nonlinear_iterations = max_nonlinear_iterations, 
    convergence_monitor_args...)

    rel = ConvergenceMonitorRelaxation(; relaxation_args...)
    config[:relaxation] = rel

end

"""
    Jutul.select_nonlinear_relaxation_model(model, rel_type::ConvergenceMonitorRelaxation, reports, ω)

Relaxation strategy based on convergence monitoring. The relaxation factor is
decreased if the convergence status is "bad", and increased if the status is
"good", determined by contraction factors computed from distance to convergence
(in a user-defined metric) between consecutive iterates. These are computed
during the cutting criterion check and stored in the reports.
"""
function Jutul.select_nonlinear_relaxation_model(model, rel_type::ConvergenceMonitorRelaxation, reports, ω)

    n = length(reports)-1

    if n > rel_type.warmup_iterations
        (; dw_decrease, dw_increase, w_max, w_min) = rel_type

        d, _ = compute_distance(reports[1][:errors]; pools=:all)
        distances = zeros(Float64, length(reports)-1, length(d))
        for k in 1:(length(reports)-1)
            d, _ = compute_distance(reports[k][:errors]; pools=:all)
            distances[k, :] .= d
        end
        status, _, _, oscillating = analyze_step(distances, 1, 8, 8, 0.1, 0.9, Inf)
        status = status[1]
        oscillating = oscillating[1]

        if oscillating
            ω = ω - dw_decrease
        elseif status >= 0
            ω = ω + dw_increase
        else
            @assert status == -1 "Unknown status: $status"
        end
        ω = clamp(ω, w_min, w_max)

    end

    return ω

end