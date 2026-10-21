using Jutul, Test, KernelAbstractions

abstract type CacheTileTestVariable <: Jutul.JutulVariables end
struct CacheTileScaled <: CacheTileTestVariable end
struct CacheTileComponents <: CacheTileTestVariable end
struct CacheTileFaces <: CacheTileTestVariable end
struct CacheTileTestModel{C, V}
    context::C
    secondary_variables::V
    nc::Int
    nf::Int
end
Jutul.number_of_entities(m::CacheTileTestModel, ::CacheTileTestVariable) = m.nc
Jutul.number_of_entities(m::CacheTileTestModel, ::CacheTileFaces) = m.nf

function Jutul.update_secondary_variable!(target, ::CacheTileScaled, m::CacheTileTestModel, state, indices)
    Threads.atomic_add!(state[:calls][1], 1)
    @inbounds for i in indices
        target[i] = 2*state[:input][i] + 1
    end
end
function Jutul.update_secondary_variable!(target, ::CacheTileComponents, m::CacheTileTestModel, state, indices)
    Threads.atomic_add!(state[:calls][2], 1)
    @inbounds for i in indices
        value = state[:scaled][i]
        target[1,i], target[2,i], target[3,i] = value, value^2, value*state[:input][i]
    end
end
function Jutul.update_secondary_variable!(target, ::CacheTileFaces, m::CacheTileTestModel, state, indices)
    Threads.atomic_add!(state[:calls][3], 1)
    @inbounds for i in indices
        target[i] = 3*state[:input][i]
    end
end

@testset "Secondary cache tiles preserve cell dependencies and derivatives" begin
    definitions = (scaled=CacheTileScaled(), components=CacheTileComponents(), faces=CacheTileFaces())
    for (nc, nf) in ((20_003,20_003), (20_003,10_003), (1_001,1_001)), dual in (false,true)
        input = collect(range(0.1,1.0; length=nc))
        dual && (input = Jutul.ForwardDiff.Dual.(input, 1.0))
        T = eltype(input)
        make_state() = Dict{Symbol,Any}(:input=>input, :scaled=>zeros(T,nc),
            :components=>zeros(T,3,nc), :faces=>zeros(T,nf),
            :calls=>[Threads.Atomic{Int}(0) for _ in 1:3])
        serial = CacheTileTestModel(KernelAbstractionsContext(CPU(); nthreads=1, minbatch=1000), definitions, nc, nf)
        reference = make_state()
        Jutul.update_secondary_variables_state!(reference, serial)
        for workers in unique([1, min(2,Threads.nthreads(:default)), Threads.nthreads(:default)])
            context = KernelAbstractionsContext(CPU(); nthreads=workers, minbatch=1000)
            model = CacheTileTestModel(context, definitions, nc, nf)
            actual = make_state()
            @test Jutul.update_secondary_variables_state!(actual, model) === actual
            for key in (:scaled,:components,:faces)
                @test isequal(actual[key], reference[key])
            end
            if nc >= 16_384 && nf == nc && workers > 1
                # Verify that a large threaded update really exercises tiles.
                @test all(counter[] > workers for counter in actual[:calls])
            else
                @test all(counter[] <= workers for counter in actual[:calls])
            end
        end
    end
end
