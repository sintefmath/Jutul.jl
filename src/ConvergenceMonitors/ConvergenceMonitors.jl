module ConvergenceMonitors

    using Jutul

    export ConvergenceMonitorCuttingCriterion
    export compute_distance
    export set_convergence_monitor_cutting_criterion!
    export set_convergence_monitor_relaxation!

    include("cutting_criterions.jl")
    include("distance_functions.jl")
    include("contraction_factors.jl")
    include("convergence_monitors.jl")
    include("relaxation.jl")
    include("utils.jl")

end