# Overloads for TPFA storage type

function declare_pattern(model::CompositeModel, eq::Pair{Symbol, <:ConservationLaw}, e_s::ConservationLawTPFAStorage, entity)
    return declare_pattern(model, last(eq), e_s, entity)
end

function update_equation!(eq_s::ConservationLawTPFAStorage, law::Pair{Symbol, <:ConservationLaw}, storage, model::CompositeModel, dt)
    k, eq = law
    m = composite_submodel(model, k)
    return update_equation!(eq_s, eq, storage, m, dt)
end

function setup_equation_storage(
    model::CompositeModel,
    eq::Pair{Symbol, ConservationLaw{S, T, H, G}},
    storage;
    extra_sparsity = nothing,
    kwarg...) where {S, T<:TwoPointPotentialFlowHardCoded, H, G}
    return ConservationLawTPFAStorage(model, eq[2]; kwarg...)
end

function setup_equation_storage(
    model::CompositeModel,
    eq::Pair{Symbol, ConservationLaw{S, PotentialFlow{:fvm, A, B, C}, H, G}},
    storage;
    kwarg...) where {S, A, B, C, H, G}
    k, eq = eq
    return ConservationLawFiniteVolumeStorage(composite_submodel(model, k), eq, storage; kwarg...)
end

function declare_pattern(
    model::CompositeModel,
    eq::Pair{Symbol, ConservationLaw{S, PotentialFlow{:fvm, A, B, C}, H, G}},
    e_s::ConservationLawFiniteVolumeStorage,
    entity) where {S, A, B, C, H, G}
    k, eq = eq
    return declare_pattern(composite_submodel(model, k), eq, e_s, entity)
end
