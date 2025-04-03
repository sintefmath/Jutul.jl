struct ConvergenceMonitorRelaxation <: Jutul.NonLinearRelaxation
    w_min::Float64
    w_max::Float64
    dw_decrease::Float64
    dw_increase::Float64
end

function ConvergenceMonitorRelaxation(; w_min = 0.1, dw = 0.2, dw_increase = nothing, dw_decrease = nothing, w_max = 1.0)
    if isnothing(dw_increase)
        dw_increase = dw/2
    end
    if isnothing(dw_decrease)
        dw_decrease = dw
    end
    return ConvergenceMonitorRelaxation(w_min, w_max, dw_decrease, dw_increase)
end

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

function Jutul.select_nonlinear_relaxation_model(model, rel_type::ConvergenceMonitorRelaxation, reports, ω)
    if length(reports) > 1
        (; dw_decrease, dw_increase, w_max, w_min) = rel_type
        
        report = reports[end-1]
        @assert haskey(report, :convergence_monitor)
        status = report[:convergence_monitor][:status]

        if status == :bad
            ω = ω - dw_decrease
        elseif status == :good
            ω = ω + dw_increase
        end
        ω = clamp(ω, w_min, w_max)

        println("Relaxation: ", ω)

    end
    return ω
end