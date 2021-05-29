export SegmentTotalVelocity, BottomHolePressure, SurfacePhaseRates
export WellGrid, MultiSegmentWell
export SegmentTotalVelocity, BottomHolePressure, SurfacePhaseRates

export InjectorControl, ProducerControl, SinglePhaseRateTarget, BottomHolePressureTarget

export Well, Perforations
export MixedWellSegmentFlow


abstract type WellPotentialFlowDiscretization <: PotentialFlowDiscretization end

"""
Two point approximation with flux for wells
"""
struct MixedWellSegmentFlow <: WellPotentialFlowDiscretization end

abstract type WellGrid <: TervGrid end
struct MultiSegmentWell <: WellGrid 
    volumes          # One per cell
    perforations     # (self -> local cells, reservoir -> reservoir cells, WI -> connection factor)
    neighborship     # Well cell connectivity
    top              # "Top" node where scalar well quantities live
    function MultiSegmentWell(volumes::AbstractVector, reservoir_cells;
                                                        WI = nothing,
                                                        N = nothing,
                                                        perforation_cells = nothing,
                                                        reference_depth = 0,
                                                        accumulator_volume = 1e-3*mean(volumes))
        nv = length(volumes)
        nc = nv + 1
        nr = length(reservoir_cells)
        if isnothing(N)
            @debug "No connectivity. Assuming nicely ordered linear well."
            N = vcat((1:nv)', (2:nc)')
        elseif maximum(N) == nv
            N = vcat([1, 2], N+1)
        end
        volumes = vcat([accumulator_volume], volumes)
        @show volumes
        if isnothing(WI)
            @warn "No well indices provided. Using 1e-12."
            WI = repeat(1e-12, nr)
        end
        if !isnothing(reservoir_cells) && isnothing(perforation_cells)
            @assert length(reservoir_cells) == nv "If no perforation cells are given, we must 1->1 correspondence between well volumes and reservoir cells."
            perforation_cells = collect(2:nc)
        end
        @assert size(N, 1) == 2
        # @assert length(dz) == nseg "dz must have one entry per segment, plus one for the top segment"
        @assert length(WI) == nr  "Must have one well index per perforated cell"
        @assert length(perforation_cells) == nr

        perf = (self = perforation_cells, reservoir = reservoir_cells, WI = WI)
        accumulator = (reference_depth = reference_depth, )
        new(volumes, perf, N, accumulator)
    end
end

struct PotentialDropBalanceWell <: TervEquation
    # Equation: pot_diff(p) - pot_diff_model(v, p)
    potential_difference::TervAutoDiffCache       # Differentiated with respect to both Cells
    potential_drop_model_cells::TervAutoDiffCache # Differentiated with respect to Cells
    potential_drop_model_faces::TervAutoDiffCache # Differentiated with respect to Velocity
end

function PotentialDropBalanceWell(model, number_of_equations; kwarg...)
    D = model.domain
    cell_unit = Cells()
    face_unit = Faces()
    nf = count_units(D, face_unit)

    alloc = (n, unit) -> CompactAutoDiffCache(number_of_equations, n, model, unit = unit; kwarg...)
    # One potential drop per velocity
    dp = alloc(nf, face_unit)
    # Two cells per face -> 2*nf allocated
    dp_model_cells = alloc(2*nf, cell_unit)
    # One potential drop model per velocity
    dp_model_faces = alloc(nf, face_unit)

    PotentialDropBalanceWell(dp, dp_model_cells, dp_model_faces)
end

struct ControlEquationWell <: TervEquation
    # Equation:
    #        q_t - target = 0
    #        p|top cell - target = 0
    # We need to store derivatives with respect to q_t and the top cell
    target_well::TervAutoDiffCache
    target_topcell::TervAutoDiffCache
    function ControlEquationWell(model, number_of_equations; kwarg...)
        @assert number_of_equations == 1
        alloc = (unit) -> CompactAutoDiffCache(number_of_equations, 1, model, unit = unit; kwarg...)
        # One potential drop per velocity
        target_well = alloc(Well())
        target_topcell = alloc(Cells())
        new(target_well, target_topcell)
    end
end


# Well segments
"""
Perforations are connections from well cells to reservoir vcells
"""
struct Perforations <: TervUnit end
"""
Well variables - units that we have exactly one of per well (and usually relates to the surface connection)
"""
struct Well <: TervUnit end

## Well targets
abstract type WellTarget end
struct BottomHolePressureTarget <: WellTarget
    value::AbstractFloat
end

struct SinglePhaseRateTarget <: WellTarget
    value::AbstractFloat
    phase::AbstractPhase
end

## Well controls
abstract type WellForce <: TervForce end
abstract type WellControlForce <: WellForce end

struct InjectorControl <: WellControlForce
    target::WellTarget
    injection_mixture
end

struct ProducerControl <: WellControlForce
    target::WellTarget
end

function declare_units(W::MultiSegmentWell)
    c = (Cells(),         length(W.volumes))
    f = (Faces(),         size(W.neighborship, 2))
    p = (Perforations(),  length(W.perforations.self))
    w = (Well(), 1)
    return [c, f, p, w]
end

# Total velocity in each well segment
struct SegmentTotalVelocity <: ScalarVariable end
function associated_unit(::SegmentTotalVelocity) Faces() end

# Bottom hole pressure for the well
# struct BottomHolePressure <: ScalarVariable end
# function associated_unit(::BottomHolePressure) Well() end

# Phase rates for well at surface conditions
# struct SurfacePhaseRates <: GroupedVariables end
# function associated_unit(::SurfacePhaseRates) Well() end

struct TotalMassRateWell <: ScalarVariable end
function associated_unit(::TotalMassRateWell) Well() end

# function degrees_of_freedom_per_unit(model, v::SurfacePhaseRates)
#    return number_of_phases(model.system)
# end

# Selection of primary variables
function select_primary_variables!(S, domain::DiscretizedDomain{G}, system, arg...) where {G<:MultiSegmentWell}
    select_primary_variables!(S, system)
    S[:SegmentTotalVelocity] = SegmentTotalVelocity()
    S[:TotalWellMassRate] = TotalMassRateWell()
    # S[:SurfacePhaseRates] = SurfacePhaseRates()
    # S[:BottomHolePressure] = BottomHolePressure()
end

function select_equations!(eqs, domain::DiscretizedDomain{G}, system, arg...) where {G<:MultiSegmentWell}
    select_equations!(eqs, system)
    eqs[:potential_balance] = (PotentialDropBalanceWell, 1)
    eqs[:target_closure] = (TargetClosureWell, 1)
end
