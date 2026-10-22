using Jutul, Test, KernelAbstractions

struct TileScalar <: Jutul.ScalarVariable end
struct TileComponents <: Jutul.VectorVariables end
struct TileFaces <: Jutul.ScalarVariable end
struct TileFaceComponents <: Jutul.ScalarVariable end
Jutul.associated_entity(::TileFaces) = Faces()
Jutul.associated_entity(::TileFaceComponents) = Faces()

struct TileDomain <: Jutul.JutulDomain
    nc::Int
    nf::Int
end
# Keep the generic get_entities fallback (Cells only), as some application
# domains do. Secondary definitions must determine the entities to evaluate.
Jutul.count_entities(d::TileDomain, ::Cells) = d.nc
Jutul.count_entities(d::TileDomain, ::Faces) = d.nf

struct TileModel{C, V, S}
    context::C
    secondary_variables::V
    domain::TileDomain
    scheme::S
end
TileModel(ctx, vars, domain) = TileModel(ctx, vars, domain, nothing)
Jutul.secondary_scheme(m::TileModel{C, V, Symbol}) where {C, V} = m.scheme

function Jutul.update_secondary_variable!(out, ::TileScalar, m::TileModel, s, ix)
    Threads.atomic_add!(s.calls[1], 1)
    for i in ix
        out[i] = 2 * s.input[i] + 1
    end
    return
end
function Jutul.update_secondary_variable!(out, ::TileComponents, m::TileModel, s, ix)
    Threads.atomic_add!(s.calls[2], 1)
    for i in ix
        out[1, i] = s.scaled[i]^2
        out[2, i] = s.scaled[i] * s.input[i]
    end
    return
end
function Jutul.update_secondary_variable!(out, ::TileFaces, m::TileModel, s, ix)
    Threads.atomic_add!(s.calls[3], 1)
    for i in ix
        out[i] = 3 * s.face_input[i]
    end
    return
end
function Jutul.update_secondary_variable!(out, ::TileFaceComponents, m::TileModel, s, ix)
    Threads.atomic_add!(s.calls[4], 1)
    for i in ix
        out[i] = s.faces[i]^2
    end
    return
end

@testset "Secondary schemes preserve entity-local values and derivatives" begin
    # Entity definitions are interleaved, but each dependency is entity-local.
    vars = (scaled = TileScalar(), faces = TileFaces(), components = TileComponents(), face_components = TileFaceComponents())
    for (nc, nf) in ((0, 0), (0, 17), (17, 0), (117, 37), (20_003, 10_003)), T in (Float32, Float64, Jutul.ForwardDiff.Dual{Nothing, Float64, 1})
        input = T.(range(0.1, 1.0; length = nc))
        if T <: Jutul.ForwardDiff.Dual
            input = [Jutul.ForwardDiff.Dual{Nothing}(x, 1.0) for x in range(0.1, 1.0; length = nc)]
        end
        make_state() = (
            input = input, scaled = zeros(T, nc), components = zeros(T, 2, nc),
            face_input = T.(range(0.2, 0.8; length = nf)), faces = zeros(T, nf),
            face_components = zeros(T, nf), calls = [Threads.Atomic{Int}(0) for _ in 1:4],
        )
        reference = make_state()
        serial = TileModel(KernelAbstractionsContext(CPU(); nthreads = 1), vars, TileDomain(nc, nf))
        for (name, var) in pairs(vars)
            target = reference[name]
            Jutul.update_secondary_variable!(target, var, serial, reference, entity_eachindex(target))
        end
        for workers in unique((1, min(2, Threads.nthreads()), Threads.nthreads())), mb in (7, 1000, typemax(Int)), scheme in (:flat, :fused, :strided), context_type in (DefaultContext, ParallelCSRContext, KernelAbstractionsContext)
            if context_type == KernelAbstractionsContext
                ctx = context_type(CPU(); nthreads = workers, minbatch = mb, secondary_scheme = scheme)
            elseif context_type == ParallelCSRContext
                ctx = context_type(workers; minbatch = mb, secondary_scheme = scheme)
            else
                ctx = context_type(; nthreads = workers, minbatch = mb, secondary_scheme = scheme)
            end
            model = TileModel(ctx, vars, TileDomain(nc, nf))
            actual = make_state()
            @test Jutul.update_secondary_variables_state!(actual, model) === actual
            for key in keys(vars)
                @test isequal(actual[key], reference[key])
            end
            if scheme == :fused
                expected_calls = cld.([nc, nc, nf, nf], mb)
            else
                batches = scheme == :flat ? clamp(nc ÷ mb, 1, workers) :
                    (nc ÷ mb < 2 ? 1 : clamp(4nc ÷ mb, 1, 4workers))
                expected_calls = fill(batches, 4)
            end
            @test [c[] for c in actual.calls] == expected_calls
            @test Jutul.update_secondary_variables_state!(actual, model, NamedTuple()) === actual
        end
        # Mutable variable definitions use the same opt-in interface.
        ctx = KernelAbstractionsContext(CPU())
        mutable_vars = Jutul.OrderedDict(pairs(vars))
        actual = make_state()
        # The model overrides the flat context, also for mutable definitions.
        model = TileModel(ctx, mutable_vars, TileDomain(nc, nf), :fused)
        @test secondary_scheme(model) == :fused
        Jutul.update_secondary_variables_state!(actual, model)
        @test all(isequal(actual[k], reference[k]) for k in keys(vars))
        @test [c[] for c in actual.calls] == cld.([nc, nc, nf, nf], 1000)
    end
end

struct TileTestBackend <: KernelAbstractions.Backend end
struct TileTestContext <: Jutul.JutulContext end
@testset "Secondary scheme context settings and model overrides" begin
    @test secondary_scheme(TileTestContext()) == :flat
    for constructor in (DefaultContext, ParallelCSRContext, () -> KernelAbstractionsContext(CPU()))
        @test secondary_scheme(constructor()) == :flat
    end
    for scheme in (:flat, :fused, :strided)
        contexts = (
            DefaultContext(; minbatch = 512, secondary_scheme = scheme),
            ParallelCSRContext(; minbatch = 512, thread_type = :serial, secondary_scheme = scheme),
            KernelAbstractionsContext(
                CPU(); minbatch = 512, secondary_scheme = scheme,
                linear_float_type = Float32, linear_index_type = Int32
            ),
        )
        for ctx in contexts
            @test ctx.secondary_scheme == secondary_scheme(ctx) == scheme
            @test secondary_scheme(adjoint(ctx)) == scheme
            @test minbatch(adjoint(ctx)) == 512
            @test secondary_scheme((; context = ctx)) == scheme
        end
        @test secondary_scheme(Jutul.linear_solver_context(last(contexts))) == scheme
    end
    @test_throws ArgumentError DefaultContext(; secondary_scheme = :unknown)
    @test_throws ArgumentError ParallelCSRContext(; secondary_scheme = :unknown)
    @test_throws ArgumentError KernelAbstractionsContext(CPU(); secondary_scheme = :unknown)
    for scheme in (:fused, :strided)
        @test_throws ArgumentError KernelAbstractionsContext(CPU(); secondary_scheme = scheme, use_kernels_for_secondary = true)
        @test_throws ArgumentError KernelAbstractionsContext(TileTestBackend(); secondary_scheme = scheme)
        # A model override cannot enable host evaluation of device arrays.
        ctx = KernelAbstractionsContext(TileTestBackend())
        model = TileModel(ctx, NamedTuple(), TileDomain(0, 0), scheme)
        @test_throws ArgumentError Jutul.update_secondary_variables_state!(NamedTuple(), model)
    end
    model = TileModel(DefaultContext(), NamedTuple(), TileDomain(0, 0), :unknown)
    @test_throws ArgumentError Jutul.update_secondary_variables_state!(NamedTuple(), model)
    # Serial thread_type retains a single update even with strided scheduling.
    ctx = ParallelCSRContext(; minbatch = 7, thread_type = :serial, secondary_scheme = :strided)
    state = (input = ones(100), scaled = zeros(100), calls = [Threads.Atomic{Int}(0)])
    Jutul.update_secondary_variables_state!(state, TileModel(ctx, (scaled = TileScalar(),), TileDomain(100, 0)))
    @test state.scaled == fill(3.0, 100)
    @test state.calls[1][] == 1
end
