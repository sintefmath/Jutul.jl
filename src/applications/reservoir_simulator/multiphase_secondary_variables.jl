export PhaseMassDensities, ConstantCompressibilityDensities, MassMobilities
export BrooksCoreyRelPerm, TabulatedRelPermSimple

abstract type PhaseVariables <: GroupedVariables end
abstract type ComponentVariable <: GroupedVariables end

function degrees_of_freedom_per_entity(model, sf::PhaseVariables) number_of_phases(model.system) end

# Single-phase specialization
function degrees_of_freedom_per_entity(model::SimulationModel{D, S}, sf::ComponentVariable) where {D, S<:SinglePhaseSystem} 1 end

# Immiscible specialization
function degrees_of_freedom_per_entity(model::SimulationModel{D, S}, sf::ComponentVariable) where {D, S<:ImmiscibleSystem}
    number_of_phases(model.system)
end

function select_secondary_variables_system!(S, domain, system::MultiPhaseSystem, formulation)
    nph = number_of_phases(system)
    S[:PhaseMassDensities] = ConstantCompressibilityDensities(nph)
    S[:TotalMasses] = TotalMasses()
    S[:PhaseViscosities] = ConstantVariables(1e-3*ones(nph)) # 1 cP for all phases by default
end

function select_secondary_variables_system!(S, domain, system::SinglePhaseSystem, formulation)
    S[:PhaseMassDensities] = ConstantCompressibilityDensities(1)
    S[:TotalMasses] = TotalMasses()
    S[:PhaseViscosities] = ConstantVariables([1e-3]) # 1 cP for all phases by default
    S[:Saturations] = ConstantVariables([1.0])
end


function minimum_output_variables(system::MultiPhaseSystem, primary_variables)
    [:TotalMasses]
end

abstract type RelativePermeabilities <: PhaseVariables end

struct BrooksCoreyRelPerm <: RelativePermeabilities
    exponents
    residuals
    endpoints
    residual_total
    function BrooksCoreyRelPerm(sys_or_nph::Union{MultiPhaseSystem, Integer}, exponents = 1.0, residuals = 0.0, endpoints = 1.0)
        if isa(sys_or_nph, Integer)
            nph = sys_or_nph
        else
            nph = number_of_phases(sys_or_nph)
        end

        expand(v::Real) = [v for i in 1:nph]
        function expand(v::AbstractVector)
            @assert length(v) == nph
            return v
        end
        e = expand(exponents)
        r = expand(residuals)
        epts = expand(endpoints)

        total = sum(residuals)
        new(e, r, epts, total)
    end
end

function transfer(c::SingleCUDAContext, kr::BrooksCoreyRelPerm)
    e = transfer(c, kr.exponents)
    r = transfer(c, kr.residuals)
    ept = transfer(c, kr.residual_total)

    nph = length(e)
    BrooksCoreyRelPerm(nph, e, r, ept)
end

@terv_secondary function update_as_secondary!(kr, kr_def::BrooksCoreyRelPerm, model, param, Saturations)
    n, sr, kwm, sr_tot = kr_def.exponents, kr_def.residuals, kr_def.endpoints, kr_def.residual_total
    @tullio kr[ph, i] = brooks_corey_relperm(Saturations[ph, i], n[ph], sr[ph], kwm[ph], sr_tot)
end

function brooks_corey_relperm(s::Real, n::Real, sr::Real, kwm::Real, sr_tot::Real)
    den = 1 - sr_tot;
    sat = (s - sr) / den;
    sat = max(min(sat, 1.0), 0.0);
    return kwm*sat^n;
end

"""
Interpolated multiphase rel. perm. that is simple (single region, no magic for more than two phases)
"""
struct TabulatedRelPermSimple <: RelativePermeabilities
    s
    kr
    interpolators
    function TabulatedRelPermSimple(s::AbstractVector, kr::AbstractMatrix; kwarg...)
        nph, n = size(kr)
        @assert nph > 0
        if eltype(s)<:AbstractVector
            # We got a set of different vectors that correspond to rows of kr
            @assert all(map(length, s) .== n)
            interpolators = map((ix) -> get_1d_interpolator(s[ix], kr[ix, :]; kwarg...), 1:nph)
        else
            # We got a single vector that is used for all rows
            @assert length(s) == n
            interpolators = map((ix) -> get_1d_interpolator(s, kr[ix, :]; kwarg...), 1:nph)
        end
        new(s, kr, interpolators)
    end
end

@terv_secondary function update_as_secondary!(kr, kr_def::TabulatedRelPermSimple, model, param, Saturations)
    I = kr_def.interpolators
    @tullio kr[ph, i] = I[ph](Saturations[ph, i])
end

"""
Mass density of each phase
"""
abstract type PhaseMassDensities <: PhaseVariables end

struct ConstantCompressibilityDensities <: PhaseMassDensities
    reference_pressure
    reference_densities
    compressibility
    function ConstantCompressibilityDensities(sys_or_nph::Union{MultiPhaseSystem, Integer}, reference_pressure = 101325.0, reference_density = 1000.0, compressibility = 1e-10)
        if isa(sys_or_nph, Integer)
            nph = sys_or_nph
        else
            nph = number_of_phases(sys_or_nph)
        end

        expand(v::Real) = [v for i in 1:nph]
        function expand(v::AbstractVector)
            @assert length(v) == nph
            return v
        end
        pref = expand(reference_pressure)
        rhoref = expand(reference_density)
        c = expand(compressibility)

        new(pref, rhoref, c)
    end
end

function transfer(c::SingleCUDAContext, rho::ConstantCompressibilityDensities)
    pref = transfer(c, rho.reference_pressure)
    rhoref = transfer(c, rho.reference_densities)
    c = transfer(c, rho.compressibility)

    nph = length(pref)
    ConstantCompressibilityDensities(nph, pref, rhoref, c)
end

@terv_secondary function update_as_secondary!(rho, density::ConstantCompressibilityDensities, model, param, Pressure)
    p_ref, c, rho_ref = density.reference_pressure, density.compressibility, density.reference_densities
    @tullio rho[ph, i] = constant_expansion(Pressure[i], p_ref[ph], c[ph], rho_ref[ph])
end

function constant_expansion(p::Real, p_ref::Real, c::Real, f_ref::Real)
    Δ = p - p_ref
    return f_ref * exp(Δ * c)
end

"""
Mobility of the mass in each phase (TBD how to represent this in general)
"""
struct MassMobilities <: PhaseVariables end

@terv_secondary function update_as_secondary!(ρλ, tv::MassMobilities, model, param, 
                                PhaseMassDensities, PhaseViscosities, RelativePermeabilities)
    kᵣ, μ, ρ = RelativePermeabilities, PhaseViscosities, PhaseMassDensities
    @tullio ρλ[α, i] = kᵣ[α, i]*ρ[α, i]/μ[α, i]
end

@terv_secondary function update_as_secondary!(ρλ_i, tv::MassMobilities, model::SimulationModel{D, S}, param, PhaseMassDensities, PhaseViscosities) where {D, S<:SinglePhaseSystem}
    μ, ρ = PhaseViscosities, PhaseMassDensities
    @tullio ρλ_i[ph, i] = ρ[ph, i]/μ[ph, i]
end

# Total masses
@terv_secondary function update_as_secondary!(totmass, tv::TotalMasses, model::SimulationModel{G, S}, param, PhaseMassDensities) where {G, S<:SinglePhaseSystem}
    pv = get_pore_volume(model)
    @tullio totmass[ph, i] = PhaseMassDensities[ph, i]*pv[i]
end

@terv_secondary function update_as_secondary!(totmass, tv::TotalMasses, model::SimulationModel{G, S}, param, PhaseMassDensities, Saturations) where {G, S<:ImmiscibleSystem}
    pv = get_pore_volume(model)
    rho = PhaseMassDensities
    s = Saturations
    @tullio totmass[ph, i] = rho[ph, i]*pv[i]*s[ph, i]
end

# Total mass
@terv_secondary function update_as_secondary!(totmass, tv::TotalMass, model::SimulationModel{G, S}, param, TotalMasses) where {G, S<:MultiPhaseSystem}
    @tullio totmass[i] = TotalMasses[ph, i]
end
