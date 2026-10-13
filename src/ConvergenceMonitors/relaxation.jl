struct ConvergenceMonitorRelaxation <: Jutul.NonLinearRelaxation
    w_min::Float64
    w_max::Float64
    dw_decrease::Float64
    dw_increase::Float64
    warmup_iterations::Int
end

"""
    ConvergenceMonitorRelaxation(; w_min = 0.1, dw = 0.2, dw_increase = nothing, dw_decrease = nothing, w_max = 1.0, warmup_iterations = 4)

Relaxation strategy based on convergence monitoring. Requires that the
simulation has been configured with a `ConvergenceMonitorCuttingCriterion` so
that the convergence status is available in the reports. The relaxation factor
is kept in `[w_min, w_max]`, decreased by `dw_decrease` (default `dw`) and
increased by `dw_increase` (default `dw/2`). The relaxation factor is not
changed during the first `warmup_iterations` iterations. See
`select_nonlinear_relaxation_model` for details.
"""
function ConvergenceMonitorRelaxation(; w_min = 0.1, dw = 0.2, dw_increase = nothing, dw_decrease = nothing, w_max = 1.0, warmup_iterations = 4)
    if isnothing(dw_increase)
        dw_increase = dw / 2
    end
    if isnothing(dw_decrease)
        dw_decrease = dw
    end
    return ConvergenceMonitorRelaxation(w_min, w_max, dw_decrease, dw_increase, warmup_iterations)
end

"""
    set_convergence_monitor_relaxation!(config; max_nonlinear_iterations = 50, convergence_monitor_args = NamedTuple(), relaxation_args...)

Utility for setting `ConvergenceMonitorRelaxation` to the simulator config. This
function also sets a `ConvergenceMonitorCuttingCriterion` to the config, with
`convergence_monitor_args` passed to `set_convergence_monitor_cutting_criterion!`,
while `relaxation_args` are passed to the `ConvergenceMonitorRelaxation`
constructor.
"""
function set_convergence_monitor_relaxation!(
        config;
        max_nonlinear_iterations = 50,
        convergence_monitor_args = NamedTuple(),
        relaxation_args...
    )

    set_convergence_monitor_cutting_criterion!(
        config;
        max_nonlinear_iterations = max_nonlinear_iterations,
        convergence_monitor_args...
    )

    rel = ConvergenceMonitorRelaxation(; relaxation_args...)
    config[:relaxation] = rel
    return config

end

# function Jutul.select_nonlinear_relaxation_model___(model, rel_type::ConvergenceMonitorRelaxation, reports, ω)

#     n = length(reports)-1

#     if n > rel_type.warmup_iterations
#         (; dw_decrease, dw_increase, w_max, w_min) = rel_type

#         d, _ = compute_distance(reports[1][:errors]; pools=:all)
#         distances = zeros(Float64, length(reports), length(d))
#         for k in 1:(length(reports))
#             d, _ = compute_distance(reports[k][:errors]; pools=:all)
#             distances[k, :] .= d
#         end
#         status, _, _, oscillating = analyze_step(distances, 1, 8, 8, 0.1, 0.9, Inf)
#         status = status[1]
#         oscillating = oscillating[1]

#         if oscillating
#             ω = ω - dw_decrease
#         elseif status >= 0
#             ω = ω + dw_increase
#         else
#             @assert status == -1 "Unknown status: $status"
#         end
#         ω = clamp(ω, w_min, w_max)

#     end

#     return ω

# end

"""
    Jutul.select_nonlinear_relaxation_model(model, rel_type::ConvergenceMonitorRelaxation, reports, ω)

Relaxation strategy based on convergence monitoring. After the warmup
iterations, the convergence monitor report of the previous iteration is used to
update the relaxation factor: it is decreased if any unconverged measure is
oscillating, and otherwise increased if no measure is classified as bad. Bad but
non-oscillating iterates leave the relaxation factor unchanged.
"""
function Jutul.select_nonlinear_relaxation_model(model, rel_type::ConvergenceMonitorRelaxation, reports, ω)

    if length(reports) > rel_type.warmup_iterations
        (; dw_decrease, dw_increase, w_max, w_min) = rel_type

        report = reports[end - 1]
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
        elseif status in (:good, :ok)
            ω = ω + dw_increase
        else
            @assert status == :bad "Unknown status: $status"
        end
        ω = clamp(ω, w_min, w_max)

    end

    return ω

end
