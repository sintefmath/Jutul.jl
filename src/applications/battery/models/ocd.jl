export Grafite, NMC111, OCD, nChargeCarriers, cMax, volumetricSurfaceArea

struct Grafite <: ActiveMaterial end
struct NMC111 <: ActiveMaterial end



const coeff1_refOCP = Polynomial([
	-4.656,
	0,
	+ 88.669,
	0,
	- 401.119,
	0,
	+ 342.909,
	0,
	- 462.471,
	0,
	+ 433.434
]);

const coeff2_refOCP =Polynomial([
	-1,
	0 ,
	+ 18.933,
	0,
	- 79.532,
	0,
	+ 37.311,
	0,
	- 73.083,
	0,
	+ 95.960
])

const coeff1_dUdT = Polynomial([
	0.199521039,
	- 0.928373822,
	+ 1.364550689000003,
	- 0.611544893999998
]);

const coeff2_dUdT = Polynomial([
	1,
	- 5.661479886999997,
	+ 11.47636191,
	- 9.82431213599998,
	+ 3.048755063
])

function ocd(T,c, ::NMC111)

	refT = 298.15
	#T=300
	#c=0.1
	#D0: 1.0000e-14
	#EaD: 5000
	cmax = 55554.0
	theta= c./cmax

	refOCP = coeff1_refOCP(theta)./ coeff2_refOCP(theta);

	dUdT = -1e-3.*coeff1_dUdT(theta)./ coeff2_dUdT(theta);
	vocd = refOCP + (T - refT) .* dUdT;
	return vocd
end
##
const coeff1 = Polynomial([
	+ 0.005269056,
	+ 3.299265709,
	- 91.79325798,
	+ 1004.911008,
	- 5812.278127,
	+ 19329.75490,
	- 37147.89470,
	+ 38379.18127,
	- 16515.05308
]);

const  coeff2= Polynomial([
	1,
	- 48.09287227,
	+ 1017.234804,
	- 10481.80419,
	+ 59431.30000,
	- 195881.6488,
	+ 374577.3152,
	- 385821.1607,
	+ 165705.8597
]);

function ocd(T,c, ::Grafite)
    cmax=30555.0
	# EaD: 5000
	# D0: 3.9000e-14
  	# cmax: 30555
    theta = c./cmax
    refT = 298.15
    refOCP = (
		0.7222
        + 0.1387 .* theta
        + 0.0290 .* theta.^0.5
        - 0.0172 ./ theta
        + 0.0019 ./ theta.^1.5
        + 0.2808 .* exp(0.9 - 15.0*theta)
        - 0.7984 .* exp(0.4465.*theta - 0.4108)
		);

	dUdT = 1e-3.*coeff1(theta)./ coeff2(theta);

	vocd = refOCP + (T - refT) .* dUdT;
	return vocd
end

function nChargeCarriers(::Grafite)
        return 1
end

function nChargeCarriers(::NMC111)
        return 1
end

function cMax(::Grafite)
        return 30555.0
end

function cMax(::NMC111)
        return 55554.0
end

function volumetricSurfaceArea(::Grafite)
        return 723600.0
end

function volumetricSurfaceArea(::NMC111)
        return 885000.0
end

function reaction_rate_const(T, c, ::Grafite)
	refT = 298.15
	k0 = 5.0310e-11
	Eak = 5000
	val = k0.*exp(-Eak./FARADAY_CONST .*(1.0./T - 1/refT));
	return val
end

function reaction_rate_const(T, c, ::NMC111)
	refT = 298.15
	k0 = 2.3300e-11
	Eak = 5000
	val = k0.*exp(-Eak./FARADAY_CONST .*(1.0./T - 1/refT));
	return val
end


function diffusion_rate(T,c,::Grafite)
	refT = 298.15
	D0 = 3.900000000000000e-14
	Ead = 5000
	val = D0.*exp(-Ead./FARADAY_CONST .*(1.0./T - 1/refT));
	return val
end
function diffusion_rate(T,c,::NMC111)
	refT = 298.15
	D0 = 1.000000000000000e-14
	Ead = 5000
	val = D0.*exp(-Ead./FARADAY_CONST .*(1.0./T - 1/refT));
	return val
end

##
#grafite = Grafite()
#nmc111 = NMC111()
#T=300;
#c=0.1;
#a=OCD(T,c,nmc111)
