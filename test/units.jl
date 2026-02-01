using Jutul, Test
@testset "unit conversion" begin
    @test si_units(:meter, :day) == (1.0, 86400.0)
    @test convert_to_si(1.0, :day) ≈ 86400.0
    @test convert_from_si(86400.0, :day) ≈ 1.0
    @test convert_to_si(0.0, "°C") ≈ 273.15
    @test convert_to_si(26.85, "°C") ≈ 300.0
    @test convert_from_si(300.0, "°C") ≈ 26.85
    for (u, v) in pairs(Jutul.UNIT_PREFIXES)
        x = si_unit(u)
        @test x == v
        @test x isa Float64
        @test convert_to_si(convert_from_si(1.0, u), u) ≈ 1.0
    end

    for (u, uval) in pairs(Jutul.all_units())
        x = si_unit(u)
        @test x == uval
        @test x > 0
        @test x isa Float64
        @test convert_to_si(convert_from_si(1.0, u), u) ≈ 1.0
    end
    K = 310.9277777775

    F = 100.0
    @test convert_to_si(F, :Fahrenheit) ≈ K
    @test convert_from_si(K, :Fahrenheit) ≈ F
    C = 37.777777777500035

    @test convert_to_si(C, :Celsius) ≈ K
    @test convert_from_si(K, :Celsius) ≈ C

    R = 559.67
    @test convert_to_si(R, :Rankine) ≈ K
    @test convert_from_si(K, :Rankine) ≈ R
end
