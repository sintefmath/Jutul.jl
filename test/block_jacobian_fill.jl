using Jutul, Test, StaticArrays

struct CountedFillPositions{T, A} <: AbstractMatrix{T}
    values::A
    reads::Threads.Atomic{Int}
end
CountedFillPositions(values::AbstractMatrix{T}) where {T} = CountedFillPositions{T, typeof(values)}(values, Threads.Atomic{Int}(0))
Base.size(p::CountedFillPositions) = size(p.values)
function Base.getindex(p::CountedFillPositions, i::Int, j::Int)
    Threads.atomic_add!(p.reads, 1)
    return p.values[i, j]
end

function counted_fill_positions(pos::Jutul.BlockJacobianPositions{I, ne, np, adjoint}) where {I, ne, np, adjoint}
    backing = CountedFillPositions(pos.first_positions)
    return Jutul.BlockJacobianPositions{I, ne, np, adjoint, typeof(backing)}(backing), backing.reads
end
function counted_fill_positions(pos)
    backing = CountedFillPositions(pos)
    return backing, backing.reads
end

function counted_fill_cache(c::CompactAutoDiffCache)
    pos, reads = counted_fill_positions(c.jacobian_positions)
    counted = CompactAutoDiffCache{typeof(c.npartials), eltype(c.entries)}(
        c.entries, c.entity, pos, c.equations_per_entity, c.number_of_entities, c.npartials
    )
    return counted, reads
end
function counted_fill_cache(c::Jutul.GenericAutoDiffCache{ne, E, T}) where {ne, E, T}
    pos, reads = counted_fill_positions(c.jacobian_positions)
    counted = Jutul.GenericAutoDiffCache{ne, E, T}(
        c.entries, c.vpos, c.variables, pos,
        c.diagonal_positions, c.number_of_entities_target, c.number_of_entities_source, c.variable_map
    )
    return counted, reads
end

function seed_fill_cache!(cache; offset = 0)
    _, ne, np = Jutul.ad_dims(cache)
    T = eltype(cache.entries)
    for j in axes(cache.entries, 2), e in 1:ne
        partials = Jutul.ForwardDiff.Partials(ntuple(p -> Float64(100j + 10e + p), np))
        cache.entries[e, j] = T(Float64(10j + e), partials)
    end
    positions = cache.jacobian_positions
    if positions isa Jutul.BlockJacobianPositions
        for j in axes(positions, 2)
            positions.first_positions[1, j] = offset + (j - 1) * np * np + 1
        end
    else
        # Full positions deliberately use an arbitrary, reversed scalar order.
        for j in axes(positions, 2), row in axes(positions, 1)
            positions[row, j] = offset + j * ne * np - row + 1
        end
    end
    return cache
end

struct FixedFillFlux{ne, T}
    flux::SVector{ne, T}
end
Jutul.flux_vector_type(::FixedFillFlux{ne}, ::Val{T}) where {ne, T} = SVector{ne, T}
Jutul.face_flux!(out, self, other, face, sign, law::FixedFillFlux, state, model, dt, disc) = law.flux

@testset "One block-position lookup during numerical filling" begin
    for np in (1, 3, 5), ne in unique((1, np)), adjoint in (false, true), compact in (false, true)
        layout = compact ? Jutul.BlockMajorLayout(adjoint) : Jutul.EquationMajorLayout(adjoint)
        context = DefaultContext(; matrix_layout = layout, nthreads = 1)
        model = (; context)
        scalar_reads = compact ? 1 : ne * np
        acc = seed_fill_cache!(CompactAutoDiffCache(ne, 2, np; context))
        expected = zeros(8np^2)
        for j in 1:2, e in 1:ne, d in 1:np
            expected[Jutul.get_jacobian_pos(acc, j, e, d)] = acc.entries[e, j].partials[d]
        end
        counted, reads = counted_fill_cache(acc)
        actual, residual = zeros(length(expected)), zeros(2ne)
        Jutul.fill_equation_entries!(actual, residual, model, counted)
        @test actual == expected
        @test reads[] == 2scalar_reads
        @test residual == vec(permutedims(Jutul.value.(acc.entries)))

        for has_diagonal in (false, true)
            cache = Jutul.GenericAutoDiffCache(
                eltype(acc.entries), ne, Cells(),
                [[1, 2], [1, 2]], 2, 2; context, has_diagonal
            )
            seed_fill_cache!(cache)
            expected .= 0
            for j in 1:4, e in 1:ne, d in 1:np
                expected[Jutul.get_jacobian_pos(cache, j, e, d)] = cache.entries[e, j].partials[d]
            end
            counted, reads = counted_fill_cache(cache)
            fill!(actual, 0)
            r = zeros(ne, 2)
            Jutul.fill_equation_entries!(actual, r, model, counted)
            @test actual == expected
            @test reads[] == 4scalar_reads
            columns = has_diagonal ? [1, 4] : [1, 3]
            @test r == Jutul.value.(cache.entries[:, columns])

            # Additive coupling and off-diagonal coupling use the same prefetch.
            impact = (entities = [1, 2], positions = [1, 2, 3], entries = [1, 2])
            reads[] = 0
            fill!(actual, 0)
            fill!(r, 0)
            Jutul.increment_equation_entries!(actual, r, model, counted, impact, -1)
            @test actual == -expected
            @test reads[] == 4scalar_reads
            @test r == -Jutul.value.(cache.entries[:, [1, 3]])
            reads[] = 0
            fill!(actual, 0)
            Jutul.fill_crossterm_entries!(
                actual, model, counted,
                counted.jacobian_positions, impact, -1
            )
            @test actual == -expected
            @test reads[] == 4scalar_reads
        end

        # TPFA: accumulation plus one flux per cell; one connection is inactive.
        flux = seed_fill_cache!(CompactAutoDiffCache(ne, 2, np; context); offset = 2np^2)
        if compact
            flux.jacobian_positions.first_positions[1, 1] = 0
        else
            flux.jacobian_positions[:, 1] .= 0
        end
        expected .= 0
        expected_r = zeros(ne, 2)
        for cell in 1:2, e in 1:ne
            a, f = acc.entries[e, cell], flux.entries[e, cell]
            expected_r[e, cell] = a.value + f.value
            for d in 1:np
                expected[Jutul.get_jacobian_pos(acc, cell, e, d)] = a.partials[d] + f.partials[d]
                pos = Jutul.get_jacobian_pos(flux, cell, e, d)
                pos > 0 && (expected[pos] = -f.partials[d])
            end
        end
        counted_acc, acc_reads = counted_fill_cache(acc)
        counted_flux, flux_reads = counted_fill_cache(flux)
        fill!(actual, 0)
        r = zeros(ne, 2)
        for cell in 1:2
            Jutul.fill_conservation_eq!(actual, r, cell, counted_acc, counted_flux, [1, 2, 3], Val(np), Val(ne))
        end
        @test actual == expected
        @test r == expected_r
        @test acc_reads[] == 2scalar_reads
        @test flux_reads[] == (compact ? 2 : 2 + ne * np)

        # Fused TPFA uses the same index prefetch after computing a flux in place.
        fixed_flux = SVector{ne}(flux.entries[:, 1])
        law = FixedFillFlux(fixed_flux)
        connections = [
            (self = 1, other = 2, face = 1, face_sign = 1),
            (self = 2, other = 1, face = 1, face_sign = -1),
        ]
        acc_reads[] = flux_reads[] = 0
        fill!(actual, 0)
        fill!(r, 0)
        for cell in 1:2
            Jutul.fill_conservation_eq_fused!(
                actual, r, cell, counted_acc,
                counted_flux.jacobian_positions, [1, 2, 3], connections,
                Jutul.TrivialGlobalMap(), nothing, [1.0], law, model, nothing,
                Val(np), Val(ne), Val(eltype(flux.entries))
            )
        end
        @test acc_reads[] == 2scalar_reads
        @test flux_reads[] == (compact ? 2 : 2 + ne * np)
        for cell in 1:2, e in 1:ne, d in 1:np
            @test actual[Jutul.get_jacobian_pos(acc, cell, e, d)] ==
                acc.entries[e, cell].partials[d] + fixed_flux[e].partials[d]
        end

        # General FVM face assembly reads each left/right block base once.
        face = Jutul.GenericAutoDiffCache(eltype(acc.entries), ne, Cells(), [[1, 2]], 1, 2; context)
        seed_fill_cache!(face)
        left, left_reads = counted_fill_positions(face.jacobian_positions)
        right_original = similar(face.jacobian_positions)
        if compact
            right_original.first_positions .= face.jacobian_positions.first_positions .+ 2np^2
        else
            right_original .= face.jacobian_positions .+ 2np^2
        end
        right, right_reads = counted_fill_positions(right_original)
        fill!(actual, 0)
        fill!(r, 0)
        Jutul.fvm_face_assembly!(r, actual, face.vpos, face, left, right, [(1, 2)], 1, Val(ne), Val(np))
        @test left_reads[] == right_reads[] == 2scalar_reads
        for j in 1:2, e in 1:ne, d in 1:np
            @test actual[Jutul.get_jacobian_pos(np, j, e, d, face.jacobian_positions)] == face.entries[e, j].partials[d]
            @test actual[Jutul.get_jacobian_pos(np, j, e, d, right_original)] == -face.entries[e, j].partials[d]
        end
    end
end
