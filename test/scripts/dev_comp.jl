using Terv, MultiComponentFlash
# Pkg.develop(PackageSpec(path = "D:/jobb\\bitbucket\\moyner.jl\\pvt\\MultiComponentFlash"))
##
# Multicomponent flow
# Variables:
# T as secondary variable (constant)
# z as primary variable

co2 = MolecularProperty(0.0440, 7.38e6, 304.1, 9.412e-5, 0.224)
c1 = MolecularProperty(0.0160, 4.60e6, 190.6, 9.863e-5, 0.011)
c10 = MolecularProperty(0.0142, 2.10e6, 617.7, 6.098e-4, 0.488)

mixture = MultiComponentMixture([co2, c1, c10], names = ["CO2", "C1", "C10"])
##
m = SSIFlash()
z0 = [0.5, 0.3, 0.2]
p0 = 10e5
T0 = 300.0
n = length(z0)


eos = GenericCubicEOS(mixture)
##