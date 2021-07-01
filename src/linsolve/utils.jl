export IterativeSolverConfig

mutable struct IterativeSolverConfig
    relative_tolerance
    absolute_tolerance
    max_iterations
    verbose
    arguments
    function IterativeSolverConfig(; relative_tolerance = 1e-3, absolute_tolerance = nothing, max_iterations = 100, verbose = false, kwarg...)
        new(relative_tolerance, absolute_tolerance, max_iterations, verbose, kwarg)
    end
end
