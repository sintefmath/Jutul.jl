using Jutul, Test
import Jutul.ConvergenceMonitors as CM

# Mock nonlinear iteration report with two measures (tolerance 1)
mock_errors(r) = [
    Dict(
        :name => :eq,
        :criterions => (; norm = (errors = r, names = ["a", "b"])),
        :tolerances => Dict(:norm => 1.0),
    ),
]

@testset "ConvergenceMonitors" begin
    @testset "contraction factor" begin
        # Geometric series δₖ = θᵏ⁻¹δ₁ is recovered exactly
        δ = [1.0e4 1.0e2; 1.0e3 1.0e1; 1.0e2 1.0e0]
        θ, N, θt = CM.compute_contraction_factor(δ, 2)
        @test θ ≈ [0.1, 0.1]
        @test N ≈ [3.0, 1.0]
        @test θt ≈ [10.0^(-1.5), 10.0^(-0.5)]
        # Oscillation requires a strict local extremum in the middle iterate
        @test CM.oscillation([3.0 3.0; 1.0 4.0; 2.0 5.0]) == [true, false]
        @test CM.oscillation([3.0 3.0; 1.0 4.0]) == [false, false]
    end
    @testset "classification and violations" begin
        # θt = 0.5: good, ok, bad, converged, and oscillating good -> ok
        θ = [0.4, 0.7, 0.995, 2.0, 0.4]
        osc = [false, false, false, false, true]
        d = [2.0, 2.0, 2.0, 1.0, 2.0]
        s = CM.classify_iterate(d, θ, fill(0.5, 5), 0.99, 0.1; oscillation = osc)
        @test s == [1, 0, -1, 1, 0]
        # Too many estimated iterations left downgrades ok to bad
        s = CM.classify_iterate(d, θ, fill(0.5, 5), 0.99, 0.1; N_left = (fill(10.0, 5), 5))
        @test s[2] == -1
        @test CM.count_violations([-1 1; -1 -1; 1 -1]) == [1 0; 2 1; 1 2]
        @test CM.count_violations([-1 1; -1 -1; 1 1]; strategy = :overall) == [1, 2, 1]
        @test_throws ErrorException CM.update_violation_count!([0], [1]; strategy = :foo)
    end
    @testset "iteration margin" begin
        δ = [1.0e4; 1.0e3; 1.0e2;;]
        # Default margin: max(N - k, buffer) with k the number of iterates
        _, _, θt = CM.analyze_step(δ, 1, 8, 2, 0.1, 0.9, Inf)
        @test θt[1] ≈ 1.0e3^(-1 / 5)
        _, _, θt = CM.analyze_step(δ, 1, 8, 2, 0.1, 0.9, Inf; its_margin = 0)
        @test θt[1] ≈ 1.0e3^(-1 / 7)
        _, _, θt = CM.analyze_step(δ, 1, 4, 2, 0.1, 0.9, Inf)
        @test θt[1] ≈ 1.0e3^(-1 / 2)
        cm = CM.analyze_convergence(δ; iteration_margin = 0)
        @test cm[:contraction_factor_target][3] ≈ 1.0e3^(-1 / 7)
        @test isnan(cm[:contraction_factor][1])
    end
    @testset "distance and pooling" begin
        d, names = CM.compute_distance(mock_errors([10.0, 0.5]))
        @test sort(names) == ["eq.a.norm", "eq.b.norm"]
        @test sort(d) == [1.0, 10.0]
        d, names = CM.compute_distance(mock_errors([10.0, 4.0]); pools = :all)
        @test d == [10.0] && names == ["AllMeasures"]
        d, = CM.compute_distance(mock_errors([10.0, 4.0]); pools = :all, pooling_norm = 1)
        @test d == [13.0]
    end
    @testset "cutting criterion and relaxation" begin
        cfg = Dict{Symbol, Any}(:timestep_selectors => [IterationTimestepSelector(6)], :info_level => -1)
        CM.set_convergence_monitor_relaxation!(cfg; convergence_monitor_args = (num_violations_cut = 1,), warmup_iterations = 1)
        cc, rel = cfg[:cutting_criterion], cfg[:relaxation]
        @test cc.iteration_target == 6
        @test rel.dw_increase == rel.dw_decrease / 2
        # Stagnating iterates are cut after num_violations_cut + 1 violations
        reports = Dict{Symbol, Any}[]
        cut = false
        for (it, r) in enumerate([1.0e3, 9.0e2, 8.0e2, 7.0e2])
            push!(reports, Dict{Symbol, Any}(:errors => mock_errors([r, r])))
            _, cut = Jutul.cutting_criterion(cc, nothing, 1.0, nothing, it, 10, cfg, nothing, reports, 1.0)
            cut && break
        end
        @test cut && length(reports) == 3
        @test all(cc.history[:status][2:3, :] .== -1)
        @test reports[end][:convergence_monitor][:status] == [-1, -1]
        # Bad, non-oscillating iterates leave the relaxation factor unchanged
        @test Jutul.select_nonlinear_relaxation_model(nothing, rel, reports, 0.8) == 0.8
        @test Jutul.select_nonlinear_relaxation_model(nothing, rel, reports[1:1], 0.8) == 0.8
    end
end
