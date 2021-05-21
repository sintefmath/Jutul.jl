export SegmentTotalVelocity, BottomHolePressure, SurfacePhaseRates
export WellGrid, MultiSegmentWell

abstract type WellGrid <: TervGrid end
struct MultiSegmentWell <: WellGrid 
    volumes          # One per cell
    perforations     # (self -> local cells, reservoir -> reservoir cells, WI -> connection factor)
    dz               # One per connection
    neighborship     # Well cell connectivity
    accumulator_node # "Top" node where scalar well quantities live
    function MultiSegmentWell(volumes::AbstractVector, reservoir_cells;
                                                        dz = nothing,
                                                        WI = nothing,
                                                        N = nothing,
                                                        perforation_cells = nothing,
                                                        reference_depth = 0,
                                                        accumulator_dz = 0)
        nseg = length(dz)
        nc = length(volumes)
        nr = length(reservoir_cells)

        if isnothing(N)
            @debug "No connectivity. Assuming nicely ordered linear well."
            N = vcat((0:nc-1)', (1:nc)')
        end
        if isnothing(dz)
            @warn "No connection dz provided. Using 0. Gravity will not affect this well."
            dz = zeros(nseg)
        end
        if isnothing(WI)
            @warn "No well indices provided. Using 1e-12."
            dz = repeat(1e-12, nr)
        end
        if !isnothing(reservoir_cells) && isnothing(perforation_cells)
            @assert length(reservoir_cells) == nc "If no perforation cells are given, we must 1->1 correspondence between well volumes and reservoir cells."
            perforation_cells = collect(1:nc)
        end
        @assert size(N, 1) == 2
        @assert size(N, 1) == 2
        @assert size(N, 2) == nseg "Topology must have one row per segment"
        @assert length(dz) == nseg "dz must have one entry per segment"
        @assert length(WI) == nr  "Must have one well index per perforated cell"
        @assert length(perforation_cells) == nr

        perf = (self = perforation_cells, reservoir = reservoir_cells, WI = WI)
        accumulator = (reference_depth = reference_depth, dz = accumulator_dz)
        new(volumes, perf, dz, N, accumulator)
    end
end

# Well segments
"""
Perforations are connections from well cells to reservoir vcells
"""
struct Perforations <: TervUnit end
"""
Well segments - connections between cells inside a well
"""
struct WellSegments <: TervUnit end
"""
Well variables - units that we have exactly one of per well (and usually relates to the surface connection)
"""
struct WellVariables <: TervUnit end

function declare_units(W::MultiSegmentWell)
    c = (Cells(),         length(W.volumes))
    f = (WellSegments(),  size(W.neighborship, 2))
    p = (Perforations(),  length(W.perforations.self))
    w = (WellVariables(), 1)
    return [c, f, p, w]
end

# Total velocity in each well segment
struct SegmentTotalVelocity <: ScalarPrimaryVariable end
function associated_unit(::SegmentTotalVelocity) WellSegments end

# Bottom hole pressure for the well
struct BottomHolePressure <: ScalarPrimaryVariable end
function associated_unit(::BottomHolePressure) WellVariables end

# Phase rates for well at surface conditions
struct SurfacePhaseRates <: GroupedPrimaryVariables
    phases
end
function associated_unit(::SurfacePhaseRates) WellVariables end

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

