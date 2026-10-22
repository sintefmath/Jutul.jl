# Compare this checkout and main using the same script and Julia environment.
using Jutul, Statistics, TOML

function assembly_fixture(n, np, adjoint)
    context = DefaultContext(; nthreads = 1, matrix_layout = Jutul.BlockMajorLayout(adjoint))
    cache = CompactAutoDiffCache(np, n, np; context)
    T = eltype(cache.entries)
    for i in 1:n, e in 1:np
        cache.entries[e, i] = T(Float64(i + e), Jutul.ForwardDiff.Partials(ntuple(p -> Float64(i + e + p), np)))
        cache.jacobian_positions.first_positions[1, i] = (i - 1) * np^2 + 1
    end
    return cache, (; context)
end

function repeated_fill_sample(f)
    for _ in 1:3
        f()
    end
    GC.gc()
    samples = [@timed f() for _ in 1:11]
    return Dict(
        "seconds" => median(s.time for s in samples),
        "bytes" => minimum(s.bytes for s in samples),
        "compile_seconds" => sum(s.compile_time for s in samples)
    )
end

function benchmark_filling(n, output)
    results = Any[]
    for np in (3, 5), adjoint in (false, true)
        acc, model = assembly_fixture(n, np, adjoint)
        nz, residual = zeros(2n * np^2), zeros(n * np)
        sample = repeated_fill_sample(() -> Jutul.fill_equation_entries!(nz, residual, model, acc))
        push!(results, merge(sample, Dict("kernel" => "compact", "block_size" => np, "adjoint" => adjoint)))

        generic = Jutul.GenericAutoDiffCache(eltype(acc.entries), np, Cells(), [[i] for i in 1:n], n, n; context = model.context)
        generic.entries .= acc.entries
        generic.jacobian_positions.first_positions .= acc.jacobian_positions.first_positions
        r = zeros(np, n)
        sample = repeated_fill_sample(() -> Jutul.fill_equation_entries!(nz, r, model, generic))
        push!(results, merge(sample, Dict("kernel" => "generic", "block_size" => np, "adjoint" => adjoint)))

        flux, _ = assembly_fixture(n, np, adjoint)
        flux.jacobian_positions.first_positions .+= n * np^2
        connections = collect(1:(n + 1))
        sample = repeated_fill_sample(
            () -> Jutul.threaded_fill_conservation_eq!(
                nz, r, model.context,
                acc, flux, connections, n, Val(np), Val(np)
            )
        )
        push!(results, merge(sample, Dict("kernel" => "tpfa", "block_size" => np, "adjoint" => adjoint)))
        println("block=", np, " adjoint=", adjoint, " ", results[(end - 2):end]); flush(stdout)
    end
    return open(output, "w") do io
        TOML.print(
            io, Dict(
                "entities" => n, "julia" => string(VERSION),
                "source" => pathof(Jutul), "results" => results
            )
        )
    end
end

benchmark_filling(isempty(ARGS) ? 200_000 : parse(Int, ARGS[1]), length(ARGS) < 2 ? "block-fill.toml" : ARGS[2])
