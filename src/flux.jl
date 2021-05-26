export half_face_flux, half_face_flux!, tp_flux, half_face_flux_kernel
export SPU, TPFA

abstract type TwoPointDiscretization <: TervDiscretization end

abstract type PotentialFlowDiscretization <: TervDiscretization end
abstract type KGradDiscretization <: PotentialFlowDiscretization end

abstract type UpwindDiscretization <: TervDiscretization end

"""
Two-point flux approximation.
"""
struct TPFA <: KGradDiscretization end
"""
Single-point upwinding.
"""
struct SPU <: UpwindDiscretization end

"Discretization of kgradp + upwind"
abstract type FlowDiscretization <: TervDiscretization end

function get_connection(face, cell, faces, N, T, z)
    D = Dict()
    if N[1, face] == cell
        other = N[2, face]
    else
        other = N[1, face]
    end
    D[:self] = cell
    D[:other] = other
    D[:face] = face
    if !isnothing(T)
        D[:T] = T[face]
    end
    if !isnothing(z)
        D[:dz] = z[cell] - z[other]
    end
    return convert_to_immutable_storage(D)
end

struct TwoPointPotentialFlow{U <:UpwindDiscretization, K <:PotentialFlowDiscretization} <: FlowDiscretization
    upwind::U
    grad::K
    conn_pos
    conn_data
    function TwoPointPotentialFlow(u, k, grid, T = nothing, z = nothing)
        N = grid.neighborship
        faces, face_pos = get_facepos(N)

        nhf = length(faces)
        nc = number_of_cells(grid)

        get_el = (face, cell) -> get_connection(face, cell, faces, N, T, z)
        el = get_el(1, 1) # Could be junk, we just need eltype
        
        conn_data = Vector{typeof(el)}(undef, nhf)
        Threads.@threads for cell = 1:nc
            @inbounds for fpos = face_pos[cell]:(face_pos[cell+1]-1)
                conn_data[fpos] = get_el(faces[fpos], cell)
            end
        end
        new{typeof(u), typeof(k)}(u, k, face_pos, conn_data)
    end
end


function half_face_flux(mob, p, G)
    flux = similar(p, 2*G.nfaces)
    half_face_flux!(flux, mob, p, G)
    return flux
end

function half_face_flux!(flux, model, flux_disc, mob, p)
    conn_data = flux_disc.conn_data
    half_face_flux!(flux, mob, p, conn_data, model.context, kernel_compatibility(model.context))
end

"Half face flux using standard loop"
function half_face_flux!(flux, mob, p, conn_data, context, ::KernelDisallowed)
    Threads.@threads for i in eachindex(conn_data)
        c = conn_data[i]
        for phaseNo = 1:size(mob, 1)
            @inbounds flux[phaseNo, i] = tp_flux(c.self, c.other, c.T, view(mob, phaseNo, :), p)
        end
    end
end

"Half face flux using kernel (GPU/CPU)"
function half_face_flux!(flux, mob, p, conn_data, context, ::KernelAllowed)
    m = length(conn_data)
    kernel = half_face_flux_kernel(context.device, context.block_size, m)
    event = kernel(flux, mob, p, conn_data, ndrange=m)
    wait(event)
end

@kernel function half_face_flux_kernel(flux, @Const(mob), @Const(p), @Const(fd))
    i = @index(Global, Linear)
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
