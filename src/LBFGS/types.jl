mutable struct LimitedMemoryHessian{T}
    init_scale::Float64
    init_strategy::Symbol
    m::T
    nullspace
    S
    Y
    it_count::T
    sign::T
end

function LimitedMemoryHessian(;
        init_scale = 1.0,
        init_strategy = :static,
        m = 5,
        it_count = 0,
        sign = 1
    )
    return LimitedMemoryHessian(init_scale, init_strategy, m, [], [], [], it_count, sign)
end
