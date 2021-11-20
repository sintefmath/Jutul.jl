export DeckViscosity, DeckShrinkage

function check_regions(regions, t)
    if !isnothing(regions)
        regions<:AbstractVector
        @assert maximum(regions) <= length(t)
        @assert minimum(regions) > 0
    end
end

abstract type DeckPhaseVariables <: PhaseVariables end

region(pv::DeckPhaseVariables, cell) = region(pv.regions, cell)
region(r::AbstractVector, cell) = @inbounds r[cell]
region(::Nothing, cell) = 1


function get_pvt(mu::DeckPhaseVariables, ph, cell)
    reg = region(mu, cell)
    return tab_by_region(mu.pvt[ph], reg)
end

function tab_by_region(pvt, reg)
    return pvt.tab[reg]
end

function get_sat(mu::DeckPhaseVariables, ph, cell)
    reg = region(mu, cell)
    return mu.sat[ph][reg]
end

struct DeckViscosity <: DeckPhaseVariables
    pvt::Tuple
    regions
    function DeckViscosity(pvt; regions = nothing)
        check_regions(regions, pvt)
        new(Tuple(pvt), regions)
    end
end

@terv_secondary function update_as_secondary!(mu, μ::DeckViscosity, model, param, Pressure)
    pvt, reg = μ.pvt, μ.regions
    @tullio mu[ph, i] = viscosity(pvt[ph], reg, Pressure[i], i)
end

struct DeckDensity <: DeckPhaseVariables
    pvt::Tuple
    regions
    function DeckDensity(pvt; regions = nothing)
        check_regions(regions, pvt)
        new(Tuple(pvt), regions)
    end
end

@terv_secondary function update_as_secondary!(rho, ρ::DeckDensity, model, param, Pressure)
    rhos = param[:reference_densities]
    pvt, reg = ρ.pvt, ρ.regions
    # Note immiscible assumption
    @tullio rho[ph, i] = rhos[ph]*shrinkage(pvt[ph], reg, Pressure[i], i)
end

# struct DeckRelativePermeability <: DeckPhaseVariables
#     sat::Tuple
#     regions
#     function DeckRelativePermeability(sat; regions = nothing)
#         check_regions(regions, sat)
#         new(Tuple(sat), regions)
#     end
# end

# @terv_secondary function update_as_secondary!(kr, kr_def::DeckRelativePermeability, model, param, Saturations)
#     @tullio kr[ph, i] = relative_permeability(get_sat(kr_def, ph, i), Saturations[ph, i])
# end

# struct DeckCapillaryPressure <: DeckPhaseVariables
#     sat::Tuple
#     regions
#     function DeckRelativePermeability(sat; regions = nothing)
#         check_regions(regions, sat)
#         new(Tuple(sat), regions)
#     end
# end

# degrees_of_freedom_per_entity(model, sf::DeckCapillaryPressure) = number_of_phases(model.system) - 1


# @terv_secondary function update_as_secondary!(kr, kr_def::DeckCapillaryPressure, model, param, Saturations)
#     @tullio kr[ph, i] = relative_permeability(get_sat(kr_def, ph, i), Saturations[ph, i])
# end
