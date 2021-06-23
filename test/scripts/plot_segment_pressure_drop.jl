## Plot pressure drop model
L = 100.0
roughness = 1e-4
D = 0.1
segment = SegmentWellBoreFrictionHB(L, roughness, D)

## Plot with some parameters
using Plots
rho = 1000
mu = 1e-3
n = 1000

v = range(0, -1, length = n)
dp = zeros(n)
for (i, vi) in enumerate(v)
    dp[i] = segment_pressure_drop(segment, vi, rho, mu);
end
Plots.plot(v, dp)
