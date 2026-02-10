"""
    PlaneCut(point, normal)

Define a planar cutting constraint by a point on the plane and a unit normal
vector. The normal defines the positive half-space.
"""
struct PlaneCut{T<:Real}
    point::SVector{3, T}
    normal::SVector{3, T}
    function PlaneCut(point::SVector{3, T}, normal::SVector{3, T}) where T<:Real
        n = normalize(normal)
        new{T}(point, n)
    end
end

function PlaneCut(point, normal)
    p = SVector{3, Float64}(point...)
    n = SVector{3, Float64}(normal...)
    return PlaneCut(p, n)
end

"""
    PolygonalSurface(polygons, normals)

Define a cutting surface made up of multiple planar polygons. Each polygon is a
vector of 3D points (SVector{3}). If `normals` is omitted, they are computed
from the polygon vertices.
"""
struct PolygonalSurface{T<:Real}
    polygons::Vector{Vector{SVector{3, T}}}
    normals::Vector{SVector{3, T}}
end

function PolygonalSurface(polygons::Vector{Vector{SVector{3, T}}}) where T
    normals = SVector{3, T}[]
    for poly in polygons
        n = polygon_normal(poly)
        push!(normals, n)
    end
    return PolygonalSurface{T}(polygons, normals)
end

"""
    polygon_normal(poly)

Compute the unit normal of a planar polygon from its vertices using Newell's method.
"""
function polygon_normal(poly::Vector{SVector{3, T}}) where T
    n = zero(SVector{3, T})
    np = length(poly)
    for i in 1:np
        j = mod1(i + 1, np)
        a = poly[i]
        b = poly[j]
        n += cross(a, b)
    end
    nn = norm(n)
    if nn < eps(T)
        return SVector{3, T}(0, 0, 1)
    end
    return n / nn
end
