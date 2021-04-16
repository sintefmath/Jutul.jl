export half_face_flux, half_face_flux!, tp_flux, half_face_flux_kernel


function half_face_flux(mob, p, G)
    flux = similar(p, 2*G.nfaces)
    half_face_flux!(flux, mob, p, G)
    return flux
end

function half_face_flux!(flux, mob, p, G::MRSTSimGraph)
    half_face_flux!(flux, mob, p, G.HalfFaceData)
end

function half_face_flux!(flux, mob, p, fd::Vector{HalfFaceData{Float64, Int64}})
    Threads.@threads for i in eachindex(flux)
        flux[i] = tp_flux(fd[i].self, fd[i].other, fd[i].T, mob, p)
    end
end

@kernel function half_face_flux_kernel(flux, @Const(mob), @Const(p), @Const(fd))
    i = @index(Global)
    @inbounds flux[i] = tp_flux(fd[i].self, fd[i].other, fd[i].T, mob, p)
end

@inline function tp_flux(c_self::I, c_other::I, t_ij, mob::AbstractArray{R}, p::AbstractArray{R}) where {R<:Real, I<:Integer}
    dp = p[c_self] - value(p[c_other])
    if dp > 0
        m = mob[c_self]
    else
        m = value(mob[c_other])
    end
    return m*t_ij*dp
end

function value(x)
    return ForwardDiff.value(x)
end

