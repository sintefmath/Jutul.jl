export SegmentTotalVelocity, BottomHolePressure, SurfacePhaseRates

abstract type WellGrid <: TervGrid end
struct MultiSegmentWell <: WellGrid 
    volumes
    cells
    dz
    WI
    neighborship
    reference_depth
    function MultiSegmentWell(cells::AbstractVector, volumes::AbstractVector, dz::AbstractVector, WI::AbstractVector, N = nothing; reference_depth = 0)
        nseg = length(dz)
        n = length(cells)
        if isnothing(N)
            N = hcat(0:n-1, 1:n)
        end
        @assert size(N, 1) == 2
        @assert size(N, 1) == 2
        @assert size(N, 2) == nseg "Topology must have one row per segment"
        @assert length(volumes) == n "Must have one volume per node"
        @assert length(WI) == n "Must have one well index per perforated cell"
        new(cells, volumes, dz, WI, N, reference_depth)
    end
end

# Well segments
struct WellSegment <: TervUnit end
# Variables that we have one of per well
struct WellVariable <: TervUnit end


# Total velocity in each well segment
struct SegmentTotalVelocity <: ScalarPrimaryVariable end
function associated_unit(::SegmentTotalVelocity) WellSegment end

# Bottom hole pressure for the well
struct BottomHolePressure <: ScalarPrimaryVariable end
function associated_unit(::BottomHolePressure) WellVariable end

# Phase rates for well at surface conditions
struct SurfacePhaseRates <: GroupedPrimaryVariables
    phases
end
function associated_unit(::SurfacePhaseRates) WellVariable end

function degrees_of_freedom_per_unit(v::SurfacePhaseRates)
    return length(v.phases)
end

# Selection of primary variables
function select_primary_variables(domain::DiscretizedDomain{G}, system, arg...) where {G<:MultiSegmentWell}
    p_base = select_primary_variables(nothing, system, arg...)

    phases = get_phases(system)
    p_w = [SegmentTotalVelocity(), SurfacePhaseRates(phases), BottomHolePressure()]

    return vcat(p_base, p_w)
end

