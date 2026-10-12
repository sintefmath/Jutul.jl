# Shared planar geometry for cutting, gluing and lower-dimensional meshes.
const Point2{T} = SVector{2, T}
const Point3{T} = SVector{3, T}

"""Euclidean length without squaring unscaled, possibly tiny coordinates."""
function stable_norm(vector)
    scale = maximum(abs, vector)
    length = zero(float(eltype(vector)))
    if scale > 0
        length = scale*norm(vector/scale)
    end
    return length
end

"""Normalize a finite nonzero vector without overflow or underflow."""
function unit_vector(vector)
    all(isfinite, vector) || throw(ArgumentError("normal/direction must be finite"))
    scale = maximum(abs, vector)
    scale > 0 || throw(ArgumentError("normal/direction must be nonzero"))
    return normalize(vector/scale)
end

"""
    PlanarFrame(vertices; float_type=nothing, tolerance=nothing, convention=:axis)

Local coordinates and convex parts of a planar surface.
The constructor validates a simple planar boundary supplied as 3D points or
a 3-by-n matrix. Coordinate lengths use the mesh's units. `float_type` defaults
to the input floating-point type; automatic tolerance accounts for geometry
size and absolute coordinate precision.

Fields:
- `origin`: Reference point in 3D.
- `normal`: Unit plane normal.
- `u`, `v`: Orthonormal in-plane axes with `cross(u, v) == normal`.
- `parts`: Nonoverlapping convex polygons for the full surface in local 2D
  coordinates. `unproject(frame, p) = origin + p[1]*u + p[2]*v`.
"""
struct PlanarFrame{T <: AbstractFloat}
    origin::Point3{T}
    normal::Point3{T}
    u::Point3{T}
    v::Point3{T}
    parts::Vector{Vector{Point2{T}}}
end

"""
    plane_basis(normal; convention=:cutting)

Return orthonormal in-plane axes `(u, v)` with `cross(u, v) == normal` for a
unit normal. `:cutting` preserves the existing cut/glue projection convention.
`:axis` projects the least-aligned Cartesian axis onto the plane, preserving
the default axes of embedded fracture objects.
"""
function plane_basis(normal::SVector{3, T}; convention = :cutting) where T
    if convention == :cutting
        if abs(normal[1]) < T(0.9)
            reference = SVector{3, T}(1, 0, 0)
        else
            reference = SVector{3, T}(0, 1, 0)
        end
        u = normalize(cross(normal, reference))
    elseif convention == :axis
        reference = setindex(zero(SVector{3, T}), one(T), argmin(abs.(normal)))
        u = normalize(reference-dot(reference, normal)*normal)
    else
        throw(ArgumentError("plane basis convention must be :cutting or :axis"))
    end
    return u, cross(normal, u)
end

function project_to_plane(p, origin::SVector{3, T}, u, v) where T
    delta = p-origin
    return SVector{2, T}(dot(delta, u), dot(delta, v))
end

project(frame, p) = project_to_plane(p, frame.origin, frame.u, frame.v)
unproject(frame, p) = frame.origin+p[1]*frame.u+p[2]*frame.v

function PlanarFrame(points::AbstractVector{SVector{3, T}}, normal::SVector{3, T}, tol;
        convention = :axis, resolve_winding = false) where T
    origin = first(points)
    u, v = plane_basis(normal; convention = convention)
    parts = convex_parts([project_to_plane(p, origin, u, v) for p in points], tol;
        resolve_winding = resolve_winding)
    return PlanarFrame(origin, normal, u, v, parts)
end

"""
    planar_polygon_vertices(vertices; float_type=nothing, tolerance=nothing)

Validate and convert a simple planar 3D boundary, supplied as points or a
3-by-n matrix. Return `(points, unit_normal, tolerance)`, removing an optional
closing copy. Concave polygons are accepted; repeated adjacent vertices,
self-intersections, nonplanarity and zero area are rejected. Automatic tolerance
accounts for both polygon size and absolute coordinate precision.
"""
function planar_polygon_vertices(vertices; float_type = nothing, tolerance = nothing)
    if isnothing(float_type)
        if vertices isa AbstractMatrix
            isempty(vertices) && throw(ArgumentError("a polygon requires at least three vertices"))
            float_type = typeof(float(first(vertices)))
        else
            isempty(vertices) && throw(ArgumentError("a polygon requires at least three vertices"))
            float_type = typeof(float(first(first(vertices))))
        end
    end
    T = float_type
    if vertices isa AbstractMatrix
        size(vertices, 1) == 3 || throw(ArgumentError("vertices must be a 3-by-n matrix"))
        points = [SVector{3, T}(vertices[:, i]) for i in axes(vertices, 2)]
    else
        points = SVector{3, T}.(vertices)
    end
    if length(points) > 1 && first(points) == last(points)
        pop!(points)
    end
    length(points) >= 3 || throw(ArgumentError("a polygon requires at least three vertices"))
    all(p -> all(isfinite, p), points) || throw(ArgumentError("vertices must be finite"))
    origin = first(points)
    normal = 2polygon_area_vector(points)
    scale = maximum(p -> norm(p-origin), points)
    stable_norm(normal) > 1e-12*scale^2 || throw(ArgumentError("polygon has zero area"))
    normal = unit_vector(normal)
    tol = tolerance
    if isnothing(tol)
        tol = max(1e-10*scale, 64eps(maximum(p -> maximum(abs, p), points)))
    end
    isfinite(tol) && tol >= 0 || throw(ArgumentError("polygon tolerance must be nonnegative and finite"))
    all(p -> abs(dot(normal, p-origin)) <= tol, points) || throw(ArgumentError("polygon is not planar"))
    u, v = plane_basis(normal; convention = :axis)
    validate_polygon([project_to_plane(p, origin, u, v) for p in points], tol)
    return points, normal, tol
end

function PlanarFrame(vertices; float_type = nothing, tolerance = nothing, convention = :axis)
    points, normal, tol = planar_polygon_vertices(vertices; float_type = float_type, tolerance = tolerance)
    return PlanarFrame(points, normal, tol; convention = convention)
end

function polygon_area_vector(points::AbstractVector{SVector{3, T}}) where T
    area = zero(SVector{3, T})
    if length(points) >= 3
        origin = first(points)
        for i in 2:length(points)-1
            area += cross(points[i]-origin, points[i+1]-origin)/2
        end
    end
    return area
end

"""
Area/centroid of a 3D boundary. Planar polygons use signed contributions, so
concave regions are measured correctly. Warped polygons retain the historical
unsigned face-centre triangle fan used by cut-cell geometry.
"""
function polygon_area_centroid(points::AbstractVector{SVector{3, T}}) where T
    area = zero(T)
    centroid = zero(SVector{3, T})
    if !isempty(points)
        origin = first(points)
        centroid = origin
        vector = polygon_area_vector(points)
        area = stable_norm(vector)
        planar = false
        if area > 0
            normal = vector/area
            scale = maximum(p -> norm(p-origin), points)
            magnitude = maximum(p -> maximum(abs, p), points)
            tolerance = 64eps(magnitude)+64eps(float(T))*scale
            planar = all(p -> abs(dot(p-origin, normal)) <= tolerance, points)
        end
        if planar
            # Work relative to origin to avoid a large-coordinate centroid sum.
            relative = zero(SVector{3, T})
            for i in 2:length(points)-1
                a, b = points[i]-origin, points[i+1]-origin
                weight = dot(cross(a, b), normal)/2
                relative += weight*(a+b)/3
            end
            centroid = origin+relative/area
        elseif length(points) >= 3
            center = sum(points)/length(points)
            area = zero(T)
            weighted = zero(SVector{3, T})
            for i in eachindex(points)
                a, b = points[i], points[mod1(i+1, length(points))]
                weight = norm(cross(a-center, b-center))/2
                area += weight
                weighted += weight*(a+b+center)/3
            end
            if area > 0
                centroid = weighted/area
            end
        end
    end
    return area, centroid
end
