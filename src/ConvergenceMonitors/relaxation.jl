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
function ConvergenceMonitorRelaxation(; w_min = 0.1, dw = 0.2, dw_increase = nothing, dw_decrease = nothing, w_max = 1.0, warmup_iterations = 5)
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

    if length(reports) > rel_type.warmup_iterations
        (; dw_decrease, dw_increase, w_max, w_min) = rel_type
        
        report = reports[end-1]
        @assert haskey(report, :convergence_monitor)
        distance = report[:convergence_monitor][:distance]
        status = report[:convergence_monitor][:status]
        oscillating = report[:convergence_monitor][:oscillation]
        oscillating = any(oscillating .&& (distance .> 1.0))
        if any(status .== -1)
            status = :bad
        elseif all(status .== 1)
            status = :good
        else
            status = :ok
        end

        if oscillating
            ω = ω - dw_decrease
            println("Relaxation factor decreased to $ω due to oscillations.")
        elseif status ∈ [:good, :ok]
            ω = ω + dw_increase
            println("Relaxation factor increased to $ω.")
        elseif status == :none
            # No change
        else
            @assert status == :bad "Unknown status: $status"
        end
        ω = clamp(ω, w_min, w_max)

    end

    return ω

end