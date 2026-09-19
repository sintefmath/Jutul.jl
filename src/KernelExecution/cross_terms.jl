abstract type AbstractPreparedCrossTermKernel end

struct PreparedCrossTermPreparation{ST, ST0, SS, SS0, MT, MS, C, E} <:
        AbstractPreparedCrossTermKernel
    state_t::ST
    state0_t::ST0
    state_s::SS
    state0_s::SS0
    model_t::MT
    model_s::MS
    cross_term::C
    equation::E
end

@inline function (kernel::PreparedCrossTermPreparation)(index, dt)
    return Jutul.prepare_cross_term_in_entity!(index,
        kernel.state_t, kernel.state0_t,
        kernel.state_s, kernel.state0_s,
        kernel.model_t, kernel.model_s,
        kernel.cross_term, kernel.equation, dt)
end

struct PreparedCrossTermCache{C, CT, S, M, E} <:
        AbstractPreparedCrossTermKernel
    cache::C
    cross_term::CT
    states::S
    models::M
    equation::E
end

@inline function (kernel::PreparedCrossTermCache)(index, dt)
    return Jutul.update_prepared_cross_term_cache!(kernel.cache, index,
        kernel.cross_term, kernel.states, kernel.models,
        kernel.equation, dt)
end

struct PreparedCrossTermLaunch{K}
    kernel::K
    count::Int
end

struct PreparedCrossTermEvaluation{P, T, S}
    prepare::P
    target::T
    source::S
end

"""
Cross-term storage carrying a pre-converted kernel evaluation plan. The
ordinary storage interface is forwarded through `data`, keeping the cached
launch details out of the generic multimodel evaluation path.
"""
struct PreparedCrossTermStorage{D, E} <: AbstractJutulStorage
    data::D
    evaluation::E
end

PreparedCrossTermStorage(storage::AbstractJutulStorage, evaluation) =
    PreparedCrossTermStorage(data(storage), evaluation)

@inline function Base.getproperty(storage::PreparedCrossTermStorage,
        name::Symbol)
    if name === :evaluation
        return getfield(storage, :evaluation)
    else
        return getproperty(data(storage), name)
    end
end

Base.haskey(storage::PreparedCrossTermStorage, name::Symbol) =
    haskey(data(storage), name)
Base.keys(storage::PreparedCrossTermStorage) = Tuple(keys(data(storage)))

function Adapt.adapt_structure(to, kernel::PreparedCrossTermPreparation)
    return PreparedCrossTermPreparation(
        Adapt.adapt(to, kernel.state_t),
        Adapt.adapt(to, kernel.state0_t),
        Adapt.adapt(to, kernel.state_s),
        Adapt.adapt(to, kernel.state0_s),
        Adapt.adapt(to, kernel.model_t),
        Adapt.adapt(to, kernel.model_s),
        Adapt.adapt(to, kernel.cross_term),
        Adapt.adapt(to, kernel.equation))
end

function Adapt.adapt_structure(to, kernel::PreparedCrossTermCache)
    return PreparedCrossTermCache(
        Adapt.adapt(to, kernel.cache),
        Adapt.adapt(to, kernel.cross_term),
        Adapt.adapt(to, kernel.states),
        Adapt.adapt(to, kernel.models),
        Adapt.adapt(to, kernel.equation))
end

function Adapt.adapt_structure(to, launch::PreparedCrossTermLaunch)
    return PreparedCrossTermLaunch(
        Adapt.adapt(to, launch.kernel), launch.count)
end

function Adapt.adapt_structure(to, plan::PreparedCrossTermEvaluation)
    return PreparedCrossTermEvaluation(
        Adapt.adapt(to, plan.prepare),
        Adapt.adapt(to, plan.target),
        Adapt.adapt(to, plan.source))
end

function Adapt.adapt_structure(to, storage::PreparedCrossTermStorage)
    return PreparedCrossTermStorage(
        Adapt.adapt(to, data(storage)),
        Adapt.adapt(to, getfield(storage, :evaluation)))
end

maybe_convert_cross_term_evaluation(
    plan::PreparedCrossTermEvaluation, context) = nothing

function launch_preconverted_threaded_loop(kernel, count, context, args...)
    threaded_loop(index -> kernel(index, args...), count, context)
    return nothing
end

function prepared_cross_term_cache_launch(cache, cross_term, states,
        models, equation)
    kernel = PreparedCrossTermCache(
        cache, cross_term, states, models, equation)
    return PreparedCrossTermLaunch(kernel, number_of_entities(cache))
end

function prepared_cross_term_cache_launch(cache::AbstractArray, cross_term,
        states, models, equation)
    kernel = PreparedCrossTermCache(
        cache, cross_term, states, models, equation)
    return PreparedCrossTermLaunch(kernel, size(cache, 2))
end

function prepared_cross_term_target(cache::GenericAutoDiffCache{
        <:Any, <:Any, ∂x, <:Any, <:Any, <:Any, <:Any, <:Any},
        cross_term, equation, state_t, state0_t, state_s, state0_s,
        models) where ∂x
    states = (
        Jutul.local_ad(state_t, 1, ∂x),
        Jutul.local_ad(state0_t, 1, ∂x),
        Jutul.as_value(state_s), Jutul.as_value(state0_s))
    return prepared_cross_term_cache_launch(
        cache, cross_term, states, models, equation)
end

prepared_cross_term_target(cache, args...) = nothing

function prepared_cross_term_source(cache::GenericAutoDiffCache{
        <:Any, <:Any, ∂x, <:Any, <:Any, <:Any, <:Any, <:Any},
        cross_term, equation, state_t, state0_t, state_s, state0_s,
        models) where ∂x
    states = (
        Jutul.as_value(state_t), Jutul.as_value(state0_t),
        Jutul.local_ad(state_s, 1, ∂x),
        Jutul.local_ad(state0_s, 1, ∂x))
    return prepared_cross_term_cache_launch(
        cache, cross_term, states, models, equation)
end

prepared_cross_term_source(cache, args...) = nothing

function setup_cross_term_evaluation(ct_s, cross_term, equation,
        storage_t, storage_s, model_t, model_s)
    state_t = Jutul.evaluation_state(storage_t)
    state0_t = Jutul.evaluation_state0(storage_t)
    state_s = Jutul.evaluation_state(storage_s)
    state0_s = Jutul.evaluation_state0(storage_s)
    models = (model_t, model_s)
    prepare_kernel = PreparedCrossTermPreparation(
        state_t, state0_t, state_s, state0_s,
        model_t, model_s, cross_term, equation)
    prepare = PreparedCrossTermLaunch(prepare_kernel, ct_s.N)
    if ct_s.helper_mode
        @assert ct_s.target === ct_s.source
        states = (state_t, state0_t, state_s, state0_s)
        target = ()
        source = (prepared_cross_term_cache_launch(
            ct_s.source, cross_term, states, models, equation),)
    else
        target = map(Tuple(values(ct_s.target))) do cache
            prepared_cross_term_target(cache, cross_term, equation,
                state_t, state0_t, state_s, state0_s, models)
        end
        source = map(Tuple(values(ct_s.source))) do cache
            prepared_cross_term_source(cache, cross_term, equation,
                state_t, state0_t, state_s, state0_s, models)
        end
        target = filter(x -> !isnothing(x), target)
        source = filter(x -> !isnothing(x), source)
    end
    return PreparedCrossTermEvaluation(prepare, target, source)
end

function Jutul.update_cross_term!(ct_s::PreparedCrossTermStorage,
        ct::Jutul.CrossTerm, eq, storage_t, storage_s,
        model_t, model_s, dt)
    evaluation = getfield(ct_s, :evaluation)
    context = model_t.context
    launch_preconverted_threaded_loop(
        evaluation.prepare.kernel, evaluation.prepare.count, context, dt)
    for launch in evaluation.target
        launch_preconverted_threaded_loop(
            launch.kernel, launch.count, context, dt)
    end
    for launch in evaluation.source
        launch_preconverted_threaded_loop(
            launch.kernel, launch.count, context, dt)
    end
    return nothing
end
