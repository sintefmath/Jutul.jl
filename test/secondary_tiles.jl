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

struct TileModel{C, V}
    context::C
    secondary_variables::V
    domain::TileDomain
end

function Jutul.update_secondary_variable!(out, ::TileScalar, m::TileModel, s, ix)
    Threads.atomic_add!(s.calls[1], 1)
    for i in ix
        out[i] = 2*s.input[i] + 1
    end
end
function Jutul.update_secondary_variable!(out, ::TileComponents, m::TileModel, s, ix)
    Threads.atomic_add!(s.calls[2], 1)
    for i in ix
        out[1, i] = s.scaled[i]^2
        out[2, i] = s.scaled[i]*s.input[i]
    end
end
function Jutul.update_secondary_variable!(out, ::TileFaces, m::TileModel, s, ix)
    Threads.atomic_add!(s.calls[3], 1)
    for i in ix
        out[i] = 3*s.face_input[i]
    end
end
function Jutul.update_secondary_variable!(out, ::TileFaceComponents, m::TileModel, s, ix)
    Threads.atomic_add!(s.calls[4], 1)
    for i in ix
        out[i] = s.faces[i]^2
    end
end

@testset "Opt-in secondary tiles and entity dependencies" begin
    # Entity definitions are interleaved, but each dependency is entity-local.
    vars = (scaled = TileScalar(), faces = TileFaces(), components = TileComponents(), face_components = TileFaceComponents())
    for (nc, nf) in ((0, 0), (0, 17), (17, 0), (117, 37), (20_003, 10_003)), T in (Float32, Float64, Jutul.ForwardDiff.Dual{Nothing, Float64, 1})
        input = T.(range(0.1, 1.0; length = nc))
        if T <: Jutul.ForwardDiff.Dual
            input = [Jutul.ForwardDiff.Dual{Nothing}(x, 1.0) for x in range(0.1, 1.0; length = nc)]
        end
        make_state() = (input = input, scaled = zeros(T, nc), components = zeros(T, 2, nc),
            face_input = T.(range(0.2, 0.8; length = nf)), faces = zeros(T, nf),
            face_components = zeros(T, nf), calls = [Threads.Atomic{Int}(0) for _ in 1:4])
        reference = make_state()
        serial = TileModel(KernelAbstractionsContext(CPU(); nthreads = 1), vars, TileDomain(nc, nf))
        for (name, var) in pairs(vars)
            target = reference[name]
            Jutul.update_secondary_variable!(target, var, serial, reference, entity_eachindex(target))
        end
        for workers in unique((1, min(2, Threads.nthreads()), Threads.nthreads())), tile_size in (7, 1000, 100_000)
            ctx = KernelAbstractionsContext(CPU(); nthreads = workers, minbatch = 10, secondary_variable_tile_size = tile_size)
            model = TileModel(ctx, vars, TileDomain(nc, nf))
            actual = make_state()
            @test Jutul.update_secondary_variables_state!(actual, model) === actual
            for key in keys(vars)
                @test isequal(actual[key], reference[key])
            end
            @test [c[] for c in actual.calls] == cld.([nc, nc, nf, nf], tile_size)
            @test Jutul.update_secondary_variables_state!(actual, model, NamedTuple()) === actual
        end
        # Mutable variable definitions use the same opt-in interface.
        ctx = KernelAbstractionsContext(CPU(); secondary_variable_tile_size = 1000)
        mutable_vars = Jutul.OrderedDict(pairs(vars))
        actual = make_state()
        Jutul.update_secondary_variables_state!(actual, TileModel(ctx, mutable_vars, TileDomain(nc, nf)))
        @test all(isequal(actual[k], reference[k]) for k in keys(vars))
    end
end

struct TileTestBackend <: KernelAbstractions.Backend end
@testset "Secondary tile backend settings" begin
    @test KernelAbstractionsContext(CPU()).secondary_variable_tile_size == 0
    ctx = KernelAbstractionsContext(CPU(); secondary_variable_tile_size = 512,
        linear_float_type = Float32, linear_index_type = Int32)
    @test adjoint(ctx).secondary_variable_tile_size == 512
    @test Jutul.linear_solver_context(ctx).secondary_variable_tile_size == 512
    @test_throws ArgumentError KernelAbstractionsContext(CPU(); secondary_variable_tile_size = -1)
    @test_throws ArgumentError KernelAbstractionsContext(CPU(); secondary_variable_tile_size = 512, use_kernels_for_secondary = true)
    @test_throws ArgumentError KernelAbstractionsContext(TileTestBackend(); secondary_variable_tile_size = 512)
end
