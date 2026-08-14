"""
Integration methods for streamline tracing
"""

"""
    integrate_step(integrator, tracer, point, step_size, direction)

Perform one integration step using the specified integrator.

Returns the next point along the streamline, or nothing if the point is outside the domain.
"""
function integrate_step end

"""
    integrate_step(::EulerIntegrator, tracer, point, step_size, direction)

Euler integration step: x_{n+1} = x_n + h * v(x_n) / ||v(x_n)||

This is a simple first-order method.
"""
function integrate_step(::EulerIntegrator, tracer::StreamlineTracer{D, T}, 
                       point::SVector{D, T}, step_size::T, direction::T) where {D, T}
    # Find sub-cell and get velocity
    subcell_idx = find_subcell_at_point(tracer, point)
    if isnothing(subcell_idx)
        return nothing
    end
    
    velocity = tracer.subcells[subcell_idx].velocity
    speed = norm(velocity, 2)
    
    if speed < 1e-10
        return nothing
    end
    
    # Euler step
    next_point = point + direction * step_size * velocity / speed
    
    return next_point
end

"""
    integrate_step(::RK2Integrator, tracer, point, step_size, direction)

RK2 (Heun's method) integration step:
    k1 = v(x_n)
    k2 = v(x_n + h * k1 / ||k1||)
    x_{n+1} = x_n + h * (k1 + k2) / (2 * ||(k1 + k2)||)

This is a second-order method that provides better accuracy than Euler.
"""
function integrate_step(::RK2Integrator, tracer::StreamlineTracer{D, T}, 
                       point::SVector{D, T}, step_size::T, direction::T) where {D, T}
    # k1: velocity at current point
    subcell_idx = find_subcell_at_point(tracer, point)
    if isnothing(subcell_idx)
        return nothing
    end
    
    k1 = tracer.subcells[subcell_idx].velocity
    speed1 = norm(k1, 2)
    
    if speed1 < 1e-10
        return nothing
    end
    
    # Normalized k1
    k1_norm = k1 / speed1
    
    # Intermediate point for k2
    point_mid = point + direction * step_size * k1_norm
    
    # k2: velocity at intermediate point
    subcell_idx2 = find_subcell_at_point(tracer, point_mid)
    if isnothing(subcell_idx2)
        # If intermediate point is outside, fall back to Euler
        return point + direction * step_size * k1_norm
    end
    
    k2 = tracer.subcells[subcell_idx2].velocity
    
    # Average velocity
    k_avg = (k1 + k2) / 2
    speed_avg = norm(k_avg, 2)
    
    if speed_avg < 1e-10
        return nothing
    end
    
    # RK2 step
    next_point = point + direction * step_size * k_avg / speed_avg
    
    return next_point
end

"""
    integrate_step(::RK4Integrator, tracer, point, step_size, direction)

RK4 (classical 4th-order Runge-Kutta) integration step:
    k1 = v(x_n)
    k2 = v(x_n + h/2 * k1 / ||k1||)
    k3 = v(x_n + h/2 * k2 / ||k2||)
    k4 = v(x_n + h * k3 / ||k3||)
    x_{n+1} = x_n + h * (k1 + 2*k2 + 2*k3 + k4) / (6 * ||...||)

This is a fourth-order method providing high accuracy.
"""
function integrate_step(::RK4Integrator, tracer::StreamlineTracer{D, T}, 
                       point::SVector{D, T}, step_size::T, direction::T) where {D, T}
    # k1: velocity at current point
    subcell_idx = find_subcell_at_point(tracer, point)
    if isnothing(subcell_idx)
        return nothing
    end
    
    k1 = tracer.subcells[subcell_idx].velocity
    speed1 = norm(k1, 2)
    
    if speed1 < 1e-10
        return nothing
    end
    
    k1_norm = k1 / speed1
    
    # k2: velocity at first intermediate point
    point2 = point + direction * (step_size / 2) * k1_norm
    subcell_idx2 = find_subcell_at_point(tracer, point2)
    if isnothing(subcell_idx2)
        # Fall back to Euler if intermediate point is outside
        return point + direction * step_size * k1_norm
    end
    
    k2 = tracer.subcells[subcell_idx2].velocity
    speed2 = norm(k2, 2)
    
    if speed2 < 1e-10
        return nothing
    end
    
    k2_norm = k2 / speed2
    
    # k3: velocity at second intermediate point
    point3 = point + direction * (step_size / 2) * k2_norm
    subcell_idx3 = find_subcell_at_point(tracer, point3)
    if isnothing(subcell_idx3)
        # Fall back to RK2-like step
        k_avg = (k1 + k2) / 2
        speed_avg = norm(k_avg, 2)
        return point + direction * step_size * k_avg / speed_avg
    end
    
    k3 = tracer.subcells[subcell_idx3].velocity
    speed3 = norm(k3, 2)
    
    if speed3 < 1e-10
        return nothing
    end
    
    k3_norm = k3 / speed3
    
    # k4: velocity at third intermediate point
    point4 = point + direction * step_size * k3_norm
    subcell_idx4 = find_subcell_at_point(tracer, point4)
    if isnothing(subcell_idx4)
        # Fall back to RK3-like step
        k_avg = (k1 + 2*k2 + 2*k3) / 5
        speed_avg = norm(k_avg, 2)
        return point + direction * step_size * k_avg / speed_avg
    end
    
    k4 = tracer.subcells[subcell_idx4].velocity
    
    # Weighted average velocity
    k_avg = (k1 + 2*k2 + 2*k3 + k4) / 6
    speed_avg = norm(k_avg, 2)
    
    if speed_avg < 1e-10
        return nothing
    end
    
    # RK4 step
    next_point = point + direction * step_size * k_avg / speed_avg
    
    return next_point
end
