export get_1d_interpolator

function get_1d_interpolator(xs, ys; method = DataInterpolations.LinearInterpolation, cap_endpoints = true)
    if cap_endpoints
        ϵ = 100*sum(abs.(xs))
        xs = vcat(xs[1] - ϵ, xs, xs[end] + ϵ);
        ys = vcat(ys[1], ys, ys[end]);
    end
    return method(ys, xs)
end


