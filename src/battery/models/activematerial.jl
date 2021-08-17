using Terv, Polynomials

export ActiveMaterial, ACMaterial, ActiveMaterialModel
abstract type ActiveMaterial <: ElectroChemicalComponent end
struct ACMaterial <: ActiveMaterial end
struct Ocd <: ScalarVariable end
struct Diffusion <: ScalarVariable end
struct ReactionRateConst <: ScalarVariable end

const ActiveMaterialModel = SimulationModel{<:Any, <:ActiveMaterial, <:Any, <:Any}
function minimum_output_variables(
    system::ActiveMaterial, primary_variables
    )
    [:Charge, :Mass, :Ocd, :T, :TPkGrad_Phi]
end

function select_primary_variables_system!(
    S, domain, system::ActiveMaterial, formulation
    )
    S[:Phi] = Phi()
    S[:C] = C()
end

function select_secondary_variables_system!(
    S, domain, system::ActiveMaterial, formulation
    )
    S[:TPkGrad_Phi] = TPkGrad{Phi}()
    S[:TPkGrad_C] = TPkGrad{C}()
    S[:T] = T()
    
    S[:Charge] = Charge()
    S[:Mass] = Mass()

    S[:Conductivity] = Conductivity()
    S[:Diffusivity] = Diffusivity()
    S[:Ocd] = Ocd()
    S[:ReactionRateConst] = ReactionRateConst()
end

function select_equations_system!(
    eqs, domain, system::ActiveMaterial, formulation
    )
    charge_cons = (arg...; kwarg...) -> Conservation(Charge(), arg...; kwarg...)
    mass_cons = (arg...; kwarg...) -> Conservation(Mass(), arg...; kwarg...)
    eqs[:charge_conservation] = (charge_cons, 1)
    eqs[:mass_conservation] = (mass_cons, 1)
end

# ? Does this maybe look better ?
@terv_secondary(
function update_as_secondary!(
    vocd, tv::Ocd, model::SimulationModel{<:Any, MaterialType, <:Any, <:Any}, param, C
    ) where   {MaterialType <:ActiveMaterial}
    s = model.system
    # @tullio vocd[i] = ocd(T[i], C[i], s)
    @tullio vocd[i] = ocd(300.0, C[i], s)
end
)

diffusion_rate(T, C, s) = 1

@terv_secondary(
function update_as_secondary!(
    vdiffusion, tv::Diffusion, model::SimulationModel{<:Any, MaterialType, <:Any, <:Any}, param, C
    ) where   {MaterialType <:ActiveMaterial}
    s = model.system
    @tullio vdiffusion[i] = diffusion_rate(300.0, C[i], s)
end
)

reaction_rate_const(T, c, s) = 1

@terv_secondary(
function update_as_secondary!(
    vReactionRateConst, tv::ReactionRateConst, model::SimulationModel{<:Any, MaterialType, <:Any, <:Any}, param, C
    ) where   {MaterialType <:ActiveMaterial}
    s = model.system
    @tullio vReactionRateConst[i] = reaction_rate_const(300.0, C[i], s)
end
)