# Run with Julia --threads=10 in an environment containing this Jutul checkout.
# Optional ARGS specify cube edge lengths (default: 64 and 100).
using Jutul, Jutul.KAPreconditioners, SparseArrays, LinearAlgebra, Statistics, Printf

function poisson_cube(n)
    unit = sparse(1.0I, n, n)
    line = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(2.0, n), 1 => fill(-1.0, n - 1))
    return kron(unit, kron(unit, line)) + kron(unit, kron(line, unit)) + kron(line, kron(unit, unit))
end

function measure_sweeps!(x, state, A, b; steps = 10)
    run!() = KAPreconditioners.smooth_level!(x, A, b, state, steps)
    for _ in 1:3
        fill!(x, 0)
        run!()
    end
    samples = Float64[]
    for _ in 1:7
        fill!(x, 0)
        push!(samples, @elapsed run!())
    end
    fill!(x, 0)
    bytes = @allocated run!()
    return median(samples) / steps, norm(b - A * x) / norm(b), bytes
end

function benchmark_hybrid(n)
    A = csr_matrix(poisson_cube(n))
    b = [sin(i) + cos(0.7i) for i in 1:size(A, 1)]
    x = zeros(length(b))
    @printf("\ncube=%d^3, rows=%d, nnz=%d, threads=%d\n", n, length(b), nnz(A), Threads.nthreads())
    for config in (
            GaussSeidel(), HybridGaussSeidel(; npartitions = 1),
            HybridGaussSeidel(), HybridGaussSeidel(; partitioner = :metis),
        )
        setup_seconds = @elapsed state = setup_smoother(A, config)
        seconds, reduction, bytes = measure_sweeps!(x, state, A, b)
        label = config isa GaussSeidel ? "serial" : string(config.partitioner, "/", length(state.schedule.offsets) - 1)
        interface = config isa GaussSeidel ? 0 : length(state.halo_rhs)
        @printf(
            "%-16s setup=%7.3fs sweep=%8.3fms residual10=%9.2e allocated10=%d interface=%d\n",
            label, setup_seconds, 1000seconds, reduction, bytes, interface
        )
    end
    return
end

# Compile setup and sweep paths before reporting timings.
redirect_stdout(devnull) do
    benchmark_hybrid(8)
end
for n in (isempty(ARGS) ? [64, 100] : parse.(Int, ARGS))
    benchmark_hybrid(n)
end
