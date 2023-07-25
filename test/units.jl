using Jutul, Test
@testset "unit conversion" begin
    @test si_units(:meter, :day) == (1.0, 86400.0)
    @test convert_to_si(1.0, :day) ≈ 86400.0
    @test convert_from_si(86400.0, :day) ≈ 1.0
    @test convert_to_si(0.0, "°C") ≈ 273.15
    @test convert_to_si(26.85, "°C") ≈ 300.0
    @test convert_from_si(300.0, "°C") ≈ 26.85

    for u in [
        :pascal,
        :atm,
        :bar,
        :newton,
        :dyne,
        :lbf,
        :liter,
        :stb,
        :gallon_us,
        :pound,
        :kg,
        :g,
        :tonne,
        :meter,
        :inch,
        :feet,
        :day,
        :hour,
        :year,
        :second,
        :psi,
        :btu,
        :kelvin,
        :rankine,
        :joule,
        :farad,
        :ampere,
        :watt,
        :site,
        :poise,
        :gal,
        :mol,
        :dalton,
        :darcy,
        :quetta,
        :ronna,
        :yotta,
        :zetta,
        :exa,
        :peta,
        :tera,
        :giga,
        :mega,
        :kilo,
        :hecto,
        :deca,
        :deci,
        :centi,
        :milli,
        :micro,
        :nano,
        :pico,
        :femto,
        :atto,
        :zepto,
        :yocto,
        :ronto,
        :quecto]

        x = si_unit(u)
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
