module ConvergenceMonitors

    using Jutul

    export ConvergenceMonitorCuttingCriterion
    export set_convergence_monitor_cutting_criterion!

    include("distance_functions.jl")
    include("contraction_factors.jl")
    include("cutting_criterions.jl")
    include("utils.jl")

end