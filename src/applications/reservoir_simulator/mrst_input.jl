export get_minimal_tpfa_grid_from_mrst, plot_interactive, get_test_setup, get_well_from_mrst_data
export setup_case_from_mrst

struct MRSTPlotData
    faces::Array
    vertices::Array
    data::Vector
end

function get_minimal_tpfa_grid_from_mrst(name::String; relative_path=true, perm = nothing, poro = nothing, volumes = nothing, extraout = false, kwarg...)
    if relative_path
        fn = string(dirname(pathof(Jutul)), "/../data/testgrids/", name, ".mat")
    else
        fn = name
    end
    @debug "Reading MAT file $fn..."
    exported = MAT.matread(fn)
    @debug "File read complete. Unpacking data..."
    g = MRSTWrapMesh(exported["G"])
    has_trans = haskey(exported, "T") && length(exported["T"]) > 0
    if haskey(exported, "N")
        N = Int64.(exported["N"]')
        @assert has_trans
    else
        N = nothing
    end
    geo = tpfv_geometry(g, N = N)

    function get_vec(d)
        if isa(d, AbstractArray)
            return vec(copy(d))
        else
            return [d]
        end
    end

    # Deal with cell data
    if isnothing(poro)
        poro = get_vec(exported["rock"]["poro"])
    end
    if !isnothing(volumes)
        geo.volumes .= volumes
    end

    # Deal with face data
    if has_trans
        @debug "Found precomputed transmissibilities, reusing"
        T = get_vec(exported["T"])
    else
        @debug "Data unpack complete. Starting transmissibility calculations."
        if isnothing(perm)
            perm = copy((exported["rock"]["perm"])')
        end
        T = nothing
    end
    D = discretized_domain_tpfv_flow(geo, porosity = poro, permeability = perm, T = T; kwarg...)

    if extraout
        return (D, exported)
    else
        return D
    end
end

function get_well_from_mrst_data(mrst_data, system, ix; volume = 0.01, extraout = false, simple = false, W_data = mrst_data["W"], kwarg...)
    W_mrst = W_data[ix]
    w = convert_to_immutable_storage(W_mrst)

    function awrap(x::Any)
        x
    end
    function awrap(x::Number)
        [x]
    end
    ref_depth = W_mrst["refDepth"]
    rc = Int64.(awrap(w.cells))
    n = length(rc)
    # dz = awrap(w.dZ)
    WI = awrap(w.WI)
    cell_centroids = copy((mrst_data["G"]["cells"]["centroids"])')
    if size(cell_centroids, 1) == 3
        z_res = cell_centroids[3, rc]
    else
        z_res = zeros(length(rc))
    end
    res_volume = vec(copy(mrst_data["G"]["cells"]["volumes"]))

    well_cell_volume = res_volume[rc]
    if simple
        well_volume = 1e-3# volume*mean(well_cell_volume)
        # For simple well, distance from ref depth to perf
        dz = ref_depth .- z_res
        W = SimpleWell(rc, WI = WI, dz = dz, volume = well_volume)
        wmodel = SimulationModel(W, system; kwarg...)
        flow = TwoPointPotentialFlow(nothing, nothing, TrivialFlow(), W)
        reservoir_cells = [rc[1]]
    else
        if haskey(W_mrst, "isMS") && W_mrst["isMS"]
            # @info "MS well found" W_mrst
            nm =  W_mrst["name"]
            @info "MS well found: $nm"
            V = vec(copy(W_mrst["nodes"].vol))
            cn = copy(W_mrst["cells_to_nodes"])
            # pvol - volume of each node (except for top node)
            pvol = V[2:end]
            # accumulator_volume - volume of top node
            accumulator_volume = V[1]
            # perf_cells - nodes that belong to the perforations in rc
            rc = Int64.(vec(cn[:, 1]))
            perf_cells = Int64.(vec(cn[:, 2]))
            # well_topo - well topology
            well_topo = Int64.(copy(W_mrst["topo"])')
            # z depths of nodes
            z = vec(copy(W_mrst["nodes"].depth))
            # depth from tubing to perforation for each perf
            # dz = nothing # z[perf_cells] - ()
            dz = z[perf_cells] .- z_res
            # reservoir_cells - reservoir cells to be used to pick init values from
            n_nodes = length(z)
            reservoir_cells = zeros(Int64, n_nodes)
            for i in 1:n_nodes
                if i == 1
                    reservoir_cells[i] = rc[1]
                else
                    pos = findall(perf_cells .== i)
                    if isempty(pos)
                        # Not connected, guess by depth
                        z_local = z_res
                        cells_local = rc
                    else
                        z_local = z_res[pos]
                        cells_local = rc[pos]
                    end
                    d = (z[i] .- z_local).^2
                    reservoir_cells[i] = cells_local[argmin(d)]
                end
            end
        else
            pvol, accumulator_volume, perf_cells, well_topo, z, dz, reservoir_cells = simple_ms_setup(n, volume, well_cell_volume, rc, ref_depth, z_res)
        end
        W = MultiSegmentWell(pvol, rc, WI = WI, reference_depth = ref_depth, dz = dz, N = well_topo, perforation_cells = perf_cells, accumulator_volume = accumulator_volume)
        flow = TwoPointPotentialFlow(SPU(), MixedWellSegmentFlow(), TotalMassVelocityMassFractionsFlow(), W, nothing, z)
    end
    disc = (mass_flow = flow,)
    wmodel = SimulationModel(W, system, discretization = disc; kwarg...)
    if extraout
        out = (wmodel, W_mrst, vec(reservoir_cells))
    else
        out = wmodel
    end
    return out
end

function simple_ms_setup(n, volume, well_cell_volume, rc, ref_depth, z_res)
    well_volume = volume*sum(well_cell_volume)
    # For a MS well, this is the drop from the perforated cell center to the perforation (assumed zero here)
    dz = zeros(length(rc))
    pvol = (well_volume/n)*ones(n)
    z = vcat(ref_depth, z_res)
    reservoir_cells = vcat(rc[1], rc)
    well_topo = nothing
    perf_cells = nothing
    accumulator_volume = 0.001*pvol[1]
    return (pvol, accumulator_volume, perf_cells, well_topo, z, dz, reservoir_cells)
end

function get_test_setup(mesh_or_casename; case_name = "single_phase_simple", context = "cpu", timesteps = [1.0, 2.0], pvfrac = 0.05, fuse_flux = false, kwarg...)
    if isa(mesh_or_casename, String)
        G, mrst_data = get_minimal_tpfa_grid_from_mrst(mesh_or_casename, extraout = true, fuse_flux = fuse_flux)
        mesh = MRSTWrapMesh(mrst_data["G"])
    else
        mesh = mesh_or_casename
        geo = tpfv_geometry(mesh)
        G = discretized_domain_tpfv_flow(geo; kwarg...)
    end
    nc = number_of_cells(G)
    pv = G.grid.pore_volumes
    timesteps = timesteps*3600*24

    if context == "cpu"
        context = DefaultContext()
    elseif isa(context, String)
        error("Unsupported target $context")
    end
    @assert isa(context, JutulContext)

    if case_name == "single_phase_simple"
        # Parameters
        bar = 1e5
        p0 = 100*bar # 100 bar
        cl = 1e-5/bar
        pRef = 100*bar
        rhoLS = 1000
        # Single-phase liquid system (compressible pressure equation)
        phase = LiquidPhase()
        sys = SinglePhaseSystem(phase)
        # Simulation model wraps grid and system together with context (which will be used for GPU etc)
        model = SimulationModel(G, sys, context = context)
        s = model.secondary_variables
        s[:PhaseMassDensities] = ConstantCompressibilityDensities(sys, pRef, rhoLS, cl)

        # System state
        tot_time = sum(timesteps)
        irate = pvfrac*sum(pv)/tot_time
        src = [SourceTerm(1, irate), 
            SourceTerm(nc, -irate)]
        forces = build_forces(model, sources = src)

        # State is dict with pressure in each cell
        init = Dict(:Pressure => p0)
    elseif case_name == "two_phase_simple"
        bar = 1e5
        p0 = 100*bar # 100 bar
        mu = 1e-3    # 1 cP
        cl = 1e-5/bar
        pRef = 100*bar
        rhoLS = 1000
        L = LiquidPhase()
        V = VaporPhase()
        sys = ImmiscibleSystem([L, V])
        model = SimulationModel(G, sys, context = context)

        kr = BrooksCoreyRelPerm(sys, [2, 3])
        s = model.secondary_variables
        s[:RelativePermeabilities] = kr
        s[:PhaseViscosities] = ConstantVariables([mu, mu/2])
        s[:PhaseMassDensities] = ConstantCompressibilityDensities(sys, pRef, rhoLS, cl)

        tot_time = sum(timesteps)
        irate = pvfrac*sum(pv)/tot_time
        s0 = 1.0
        s = 0.0

        # s = 0.1
        # s0 = 0.9
        src  = [SourceTerm(1, irate, fractional_flow = [1 - s, s]), 
                SourceTerm(nc, -irate, fractional_flow = [1.0, 0.0])]
        forces = build_forces(model, sources = src)

        # State is dict with pressure in each cell
        init = Dict(:Pressure => p0, :Saturations => [1 - s0, s0])
    elseif case_name == "two_phase_fake_wells"
        inj = 1
        prod = nc
        G.grid.pore_volumes[inj] *= 1000
        G.grid.pore_volumes[prod] *= 1000

        bar = 1e5
        p0 = 100*bar # 100 bar
        s0 = 1.0

        mu = 1e-3    # 1 cP
        cl = 1e-5/bar
        pRef = 100*bar
        rhoLS = 1000
        L = LiquidPhase()
        V = VaporPhase()
        sys = ImmiscibleSystem([L, V])
        model = SimulationModel(G, sys, context = context)

        kr = BrooksCoreyRelPerm(sys, [2, 3])
        s = model.secondary_variables
        s[:RelativePermeabilities] = kr
        s[:PhaseViscosities] = ConstantVariables([mu, mu/2])
        s[:PhaseMassDensities] = ConstantCompressibilityDensities(sys, pRef, rhoLS, cl)

        tot_time = sum(timesteps)
        forces = build_forces(model)

        p_init = repeat([p0], nc)
        p_init[inj] = 2*p0
        p_init[prod] = p0/2

        s_init = repeat([1 - s0, s0], 1, nc)
        s_init[1, inj] = s0
        s_init[2, inj] = 1 - s0

        # State is dict with pressure in each cell
        init = Dict(:Pressure => p_init, :Saturations => s_init)
    elseif case_name == "three_phase_fake_wells"
        inj = 1
        prod = nc
        G.grid.pore_volumes[inj] *= 1000
        G.grid.pore_volumes[prod] *= 1000

        bar = 1e5
        p0 = 100*bar # 100 bar

        mu = 1e-3    # 1 cP
        cl = 1e-5/bar
        pRef = 100*bar
        rhoLS = 1000
        A = AqueousPhase()
        L = LiquidPhase()
        V = VaporPhase()
        sys = ImmiscibleSystem([A, L, V])
        model = SimulationModel(G, sys, context = context)

        kr = BrooksCoreyRelPerm(sys, [2, 2, 2])
        s = model.secondary_variables
        s[:RelativePermeabilities] = kr
        s[:PhaseViscosities] = ConstantVariables([mu, mu, mu])
        s[:PhaseMassDensities] = ConstantCompressibilityDensities(sys, pRef, rhoLS, cl)

        tot_time = sum(timesteps)
        forces = build_forces(model)

        p_init = repeat([p0], nc)
        p_init[inj] = 2*p0
        p_init[prod] = p0/2

        s_inj = [0.5, 0.3, 0.2]
        s_res = [0.3, 0.3, 0.4]

        s_inj = [0.1, 0.0, 0.9]
        s_res = [0.9, 0.0, 0.1]

        s_inj = [0.1, 0.9, 0.0]
        s_res = [0.9, 0.1, 0.0]

        s_init = repeat(s_inj, 1, nc)
        s_init[:, inj] .= s_inj

        # State is dict with pressure in each cell
        init = Dict(:Pressure => p_init, :Saturations => s_init)

    elseif case_name == "simple_compositional_fake_wells"
        inj = 1
        prod = nc
        G.grid.pore_volumes[inj] *= 1000
        G.grid.pore_volumes[prod] *= 1000
        co2 = MolecularProperty(0.0440, 7.38e6, 304.1, 9.412e-5, 0.224)
        c1 = MolecularProperty(0.0160, 4.60e6, 190.6, 9.863e-5, 0.011)
        c10 = MolecularProperty(0.0142, 2.10e6, 617.7, 6.098e-4, 0.488)
        

        z0 = [0.5, 0.3, 0.2]
        zi = [0.99, 0.01-1e-3, 1e-3]
        mixture = MultiComponentMixture([co2, c1, c10], names = ["CO2", "C1", "C10"])

        p0 = 75e5
        T0 = 423.25

        n = length(z0)
        eos = GenericCubicEOS(mixture)
        nc = number_of_cells(G)
        L, V = LiquidPhase(), VaporPhase()
        # Define system and realize on grid
        sys = MultiPhaseCompositionalSystemLV(eos, (L, V))
        model = SimulationModel(G, sys, context = context)

        kr = BrooksCoreyRelPerm(sys, [2, 3])
        s = model.secondary_variables
        s[:RelativePermeabilities] = kr
        s[:Temperature] = ConstantVariables(T0)

        tot_time = sum(timesteps)
        forces = build_forces(model)

        p_init = repeat([p0], nc)
        p_init[inj] = 2*p0
        p_init[prod] = p0/2

        z_init = repeat(z0, 1, nc)
        z_init[:, inj] .= zi

        # State is dict with pressure in each cell
        init = Dict(:Pressure => p_init, :OverallMoleFractions => z_init)
    elseif case_name == "compositional_three_phases"
        inj = 1
        prod = nc
        G.grid.pore_volumes[inj] *= 1000
        G.grid.pore_volumes[prod] *= 1000
        co2 = MolecularProperty(0.0440, 7.38e6, 304.1, 9.412e-5, 0.224)
        c1 = MolecularProperty(0.0160, 4.60e6, 190.6, 9.863e-5, 0.011)
        c10 = MolecularProperty(0.0142, 2.10e6, 617.7, 6.098e-4, 0.488)

        z0 = [0.5, 0.3, 0.2]
        zi = [0.99, 0.01-1e-3, 1e-3]
        mixture = MultiComponentMixture([co2, c1, c10], names = ["CO2", "C1", "C10"])

        p0 = 75e5
        T0 = 423.25

        n = length(z0)
        eos = GenericCubicEOS(mixture)
        nc = number_of_cells(G)
        L, V, A = LiquidPhase(), VaporPhase(), AqueousPhase()
        # Define system and realize on grid
        sys = MultiPhaseCompositionalSystemLV(eos, (A, L, V))
        model = SimulationModel(G, sys, context = context)

        kr = BrooksCoreyRelPerm(sys, [2, 3, 2])
        s = model.secondary_variables
        s[:RelativePermeabilities] = kr
        s[:Temperature] = ConstantVariables(T0)

        tot_time = sum(timesteps)
        forces = build_forces(model)

        p_init = repeat([p0], nc)
        p_init[inj] = 2*p0
        p_init[prod] = p0/2

        z_init = repeat(z0, 1, nc)
        z_init[:, inj] .= zi

        sw_init = repeat([0.2], nc)
        sw_init[inj] = 0.5

        # State is dict with pressure in each cell
        init = Dict(:Pressure => p_init, :OverallMoleFractions => z_init, :ImmiscibleSaturation => sw_init)
    else
        error("Unknown case $case_name")
    end
    # Model parameters
    parameters = setup_parameters(model)
    state0 = setup_state(model, init)
    return (state0, model, parameters, forces, timesteps)
end

function read_patch_plot(filename::String)
    vars = MAT.matread(filename)
    f = vars["faces"];
    v = vars["vertices"];
    d = vec(vars["data"])
    MRSTPlotData(f, v, d)
end

function model_from_mat(G, mrst_data, res_context)
    ## Set up reservoir part
    @debug "Loading model from MRST:" keys(mrst_data)
    if haskey(mrst_data, "mixture")
        @debug "Found compositional model"
        f = model_from_mat_comp
    elseif haskey(mrst_data, "fluid")
        @debug "Found immiscible model"
        f = model_from_mat_fluid_immiscible
    elseif haskey(mrst_data, "deck")
        @debug "Found deck model"
        f = model_from_mat_deck
    else
        error("I don't know how this model was made: $(keys(mrst_data))")
    end
    return f(G, mrst_data, res_context)
end

function model_from_mat_comp(G, mrst_data, res_context)
    ## Set up reservoir part
    f = mrst_data["fluid"]
    nkr = vec(f["nkr"])
    rhoS = vec(f["rhoS"])
    nph = length(rhoS)
    has_water = nph == 3
    @assert nph == 2 || nph == 3

    mixture = mrst_data["mixture"]
    comps = mixture["components"]
    names = copy(vec(mixture["names"]))
    n = length(comps)

    components = map(x -> MolecularProperty(x["mw"], x["pc"], x["Tc"], x["Vc"], x["acf"]), comps)
    if haskey(mrst_data, "eos")
        eosm = mrst_data["eos"]
        
        nm = eosm["name"]
        if nm == "pr"
            eos_t = PengRobinson()
        elseif nm == "prcorr"
            eos_t = PengRobinsonCorrected()
        elseif nm == "srk"
            eos_t = SoaveRedlichKwong()
        elseif nm == "rk"
            eos_t = RedlichKwong()
        elseif nm == "zj"
            eos_t = ZudkevitchJoffe()
        else
            error("$nm not supported")
        end
        if isempty(eosm["bic"])
            bic = nothing
        else
            bic = copy(eosm["bic"])
        end
        if isempty(eosm["volume_shift"])
            vs = nothing
        else
            vs = copy(eosm["volume_shift"])
            vs = tuple(vs...)
        end
        @info "Found EoS spec in input." eos_t bic vs
    else
        eos_t = PengRobinson()
        vs = nothing
        bic = nothing
        @info "Defaulting EoS." eos_t
    end
    mixture = MultiComponentMixture(components, names = names, A_ij = bic)
    eos = GenericCubicEOS(mixture, eos_t, volume_shift = vs)

    if nph == 2
        phases = (LiquidPhase(), VaporPhase())
    else
        phases = (AqueousPhase(), LiquidPhase(), VaporPhase())
    end
    sys = MultiPhaseCompositionalSystemLV(eos, phases)
    model = SimulationModel(G, sys, context = res_context)

    if haskey(f, "sgof")
        sgof = f["sgof"]
    else
        sgof = []
    end

    if nph == 2
        if isempty(sgof)
            kr = BrooksCoreyRelPerm(nph, nkr)
        else
            s, krt = preprocess_relperm_table(sgof)
            kr = TabulatedRelPermSimple(s, krt)
        end
    else
        if haskey(f, "swof")
            swof = f["swof"]
        else
            swof = []
        end
        @assert isempty(swof) && isempty(sgof) "SWOF + SGOF is not implemented yet"
        kr = BrooksCoreyRelPerm(nph, nkr)
    end
    # p = model.primary_variables
    # p[:Pressure] = Pressure(max_rel = 0.2)
    s = model.secondary_variables
    s[:RelativePermeabilities] = kr
    T = copy(vec(mrst_data["state0"]["T"]))
    if length(unique(T)) == 1
        TV = ConstantVariables(T[1], single_entity = true)
    else
        TV = ConstantVariables(T, single_entity = false)
    end

    s[:Temperature] = TV
    
    ## Model parameters
    param = setup_parameters(model)
    param[:reference_densities] = vec(rhoS)

    return (model, param)
end

function model_from_mat_fluid_immiscible(G, mrst_data, res_context)
    ## Set up reservoir part
    f = mrst_data["fluid"]
    p = vec(f["p"])
    c = vec(f["c"])
    mu = vec(f["mu"])
    nkr = vec(f["nkr"])
    rhoS = vec(f["rhoS"])

    water = AqueousPhase()
    oil = LiquidPhase()
    sys = ImmiscibleSystem([water, oil])

    model = SimulationModel(G, sys, context = res_context)
    rho = ConstantCompressibilityDensities(sys, p, rhoS, c)

    if haskey(f, "swof")
        swof = f["swof"]
    else
        swof = []
    end

    if isempty(swof)
        kr = BrooksCoreyRelPerm(sys, nkr)
    else
        s, krt = preprocess_relperm_table(swof)
        kr = TabulatedRelPermSimple(s, krt)
    end
    mu = ConstantVariables(mu)

    p = model.primary_variables
    p[:Pressure] = Pressure(max_rel = 0.2)
    s = model.secondary_variables
    s[:PhaseMassDensities] = rho
    s[:RelativePermeabilities] = kr
    s[:PhaseViscosities] = mu
    
    ## Model parameters
    param = setup_parameters(model)
    param[:reference_densities] = vec(rhoS)

    return (model, param)
end

function model_from_mat_deck(G, mrst_data, res_context)
    ## Set up reservoir part
    deck = mrst_data["deck"]
    props = deck["PROPS"]
    phases = mrst_data["phases"]

    has_wat = phases[1]
    has_oil = phases[2]
    has_gas = phases[3]
    dens = vec(props["DENSITY"])
    if has_wat && has_oil
        @assert !has_gas
        sat_table = props["SWOF"]
        pvt_1 = PVTW(props["PVTW"])
        if haskey(props, "PVDO")
            pvt_2 = PVDO(props["PVDO"])
        else
            pvt_2 = PVCDO(props["PVCDO"])
        end

        water = AqueousPhase()
        oil = LiquidPhase()
        sys = ImmiscibleSystem([water, oil])
        rhoS = dens[1:2]
    elseif has_oil && has_gas
        @assert !has_wat
        sat_table = props["SGOF"]
        pvt_1 = PVDO(props["PVDO"])
        pvt_2 = PVDG(props["PVDG"])

        gas = VaporPhase()
        oil = LiquidPhase()
        sys = ImmiscibleSystem([oil, gas])
        rhoS = dens[2:3]
    else
        error("Not supported")
    end
    # PVT
    pvt = (pvt_1, pvt_2)
    rho = DeckDensity(pvt)
    mu = DeckViscosity(pvt)
    # Rel perm
    kr_from_deck = only(sat_table)
    s, krt = preprocess_relperm_table(kr_from_deck)
    kr = TabulatedRelPermSimple(s, krt)
    # pc = 

    model = SimulationModel(G, sys, context = res_context)
    # rho = ConstantCompressibilityDensities(sys, p, rhoS, c)

    p = model.primary_variables
    p[:Pressure] = Pressure(max_rel = 0.2)
    s = model.secondary_variables
    s[:PhaseMassDensities] = rho
    s[:RelativePermeabilities] = kr
    s[:PhaseViscosities] = mu
    
    ## Model parameters
    param = setup_parameters(model)
    param[:reference_densities] = vec(rhoS)

    return (model, param)
end

function init_from_mat(mrst_data)
    state0 = mrst_data["state0"]
    p0 = state0["pressure"]
    if isa(p0, AbstractArray)
        p0 = vec(p0)
    else
        p0 = [p0]
    end
    if haskey(state0, "components")
        z0 = copy(state0["components"]')
        init = Dict(:Pressure => p0, :OverallMoleFractions => z0)
        s = copy(state0["s"])
        if size(s, 2) == 3
            sw = vec(s[:, 1])
            sw = min.(sw, 1 - MINIMUM_COMPOSITIONAL_SATURATION)
            init[:ImmiscibleSaturation] = sw
        else
            @assert size(s, 2) == 2
        end
    else
        s0 = copy(state0["s"]')
        init = Dict(:Pressure => p0, :Saturations => s0)
    end
    return init
end

function setup_case_from_mrst(casename; simple_well = false, block_backend = true, facility_grouping = :onegroup, minbatch = 1000, kwarg...)
    G, mrst_data = get_minimal_tpfa_grid_from_mrst(casename, extraout = true, fuse_flux = false; kwarg...)
    function setup_res(G, mrst_data; block_backend = false, use_groups = false)
        bctx = DefaultContext(matrix_layout = BlockMajorLayout(), minbatch = minbatch)
        # bctx = DefaultContext(matrix_layout = UnitMajorLayout())
        dctx = DefaultContext(minbatch = minbatch)
        if block_backend && use_groups
            res_context = bctx
        else
            res_context = dctx
        end

        model, param_res = model_from_mat(G, mrst_data, res_context)
        init = init_from_mat(mrst_data)

        # param_res[:tolerances][:default] = 0.01
        # param_res[:tolerances][:mass_conservation] = 0.01
    
        return (model, init, param_res)
    end
    rhoS = mrst_data["fluid"]["rhoS"]

    # Set up initializers
    models = OrderedDict()
    initializer = Dict()
    forces = Dict()
    
    
    model, init, param_res = setup_res(G, mrst_data; block_backend = block_backend, use_groups = true)
    is_comp = haskey(init, :OverallMoleFractions)
    nph = number_of_phases(model.system)

    has_schedule = haskey(mrst_data, "schedule")
    if has_schedule
        @assert !haskey(mrst_data, "dt")
        @assert !haskey(mrst_data, "W")

        schedule = mrst_data["schedule"]

        dt = schedule["step"]["val"]
        first_ctrl = schedule["control"][1]
        first_well_set = vec(first_ctrl["W"])
        
    else
        dt = mrst_data["dt"]
        first_well_set = vec(mrst_data["W"])
    end
    if isa(dt, Real)
        dt = [dt]
    end
    timesteps = vec(copy(dt))
    res_context = model.context
    w_context = DefaultContext()
    
    initializer[:Reservoir] = init
    forces[:Reservoir] = nothing
    models[:Reservoir] = model
    well_symbols = map((x) -> Symbol(x["name"]), first_well_set)
    num_wells = length(well_symbols)
    
    parameters = Dict{Symbol, Any}()
    parameters[:Reservoir] = param_res
    controls = Dict()
    sys = model.system
    for i = 1:num_wells
        sym = well_symbols[i]
    
        wi, wdata , res_cells = get_well_from_mrst_data(mrst_data, sys, i, W_data = first_well_set,
                extraout = true, volume = 1e-3, simple = simple_well, context = w_context)
        wc = wi.domain.grid.perforations.reservoir

        sv = wi.secondary_variables
        sv_m = model.secondary_variables
        sv[:PhaseMassDensities] = sv_m[:PhaseMassDensities]
        sv[:PhaseViscosities] = sv_m[:PhaseViscosities]
        if haskey(sv, :Temperature)
            # !!!!
            temp_var = sv_m[:Temperature]
            if temp_var.single_entity
                sv[:Temperature] = temp_var
            else
                T_w = vec(temp_var.constants[res_cells])
                sv[:Temperature] = ConstantVariables(T_w, single_entity = false)
            end
        end
    
        pw = wi.primary_variables
        # pw[:Pressure] = Pressure(max_rel = 0.2)
    
        models[sym] = wi
        ctrl = mrst_well_ctrl(model, wdata, is_comp, rhoS)
        if isa(ctrl, InjectorControl)
            factor = 1.01
            if is_comp
                ci = copy(vec(wdata["components"]))
                props = model.system.equation_of_state.mixture.properties
                ci = map((x, c) -> max(c.mw*x, 1e-10), ci, props)
                ci = normalize(ci, 1)
            else
                ci = vec(wdata["compi"])
            end
        elseif isa(ctrl, ProducerControl)
            factor = 0.99
            ci = nothing
        else
            # Shut.
            ci = nothing
            factor = 1.0
        end
        @debug "$sym: Well $i/$num_wells" target typeof(ctrl) ci
        param_w = setup_parameters(wi)
        param_w[:reference_densities] = vec(param_res[:reference_densities])

        first_well_cell = wc[1]
        # pw = factor*init[:Pressure][first_well_cell]
        pw = vec(init[:Pressure][res_cells])
        w0 = Dict{Symbol, Any}(:Pressure => pw, :TotalMassFlux => 1e-12)
        if is_comp
            if isnothing(ci)
                cw_0 = init[:OverallMoleFractions][:, res_cells]
                cw_0::Matrix{Float64}
            else
                cw_0 = ci
            end
            w0[:OverallMoleFractions] = cw_0
            if nph == 3
                w0[:ImmiscibleSaturation] = vec(init[:ImmiscibleSaturation][res_cells])
            end
        elseif haskey(init, :Saturations)
            w0[:Saturations] = init[:Saturations][:, res_cells]
        end

        parameters[sym] = param_w
        controls[sym] = ctrl
        forces[sym] = nothing
        initializer[sym] = w0
    end
    #
    mode = PredictionMode()
    F0 = Dict(:TotalSurfaceMassRate => 0.0)

    facility_symbols = []
    facility_owned_wells = []
    function add_facility!(wsymbols, sym)
        g = WellGroup(wsymbols)
        WG = SimulationModel(g, mode)
        ctrls = facility_subset(wsymbols, controls)
        facility_forces = build_forces(WG, control = ctrls)
        # Specifics
        @assert !haskey(models, sym)
        models[sym] = WG
        forces[sym] = facility_forces
        # Generics
        initializer[sym] = F0
        parameters[sym] = setup_parameters(WG)
        # Store the subs
        push!(facility_symbols, sym)
        push!(facility_owned_wells, wsymbols)
    end

    if num_wells > 0
        if facility_grouping == :onegroup
            add_facility!(well_symbols, :Facility)
        elseif facility_grouping == :perwell
            for sym in well_symbols
                gsym = Symbol(string(sym)*string(:_ctrl))
                add_facility!([sym], gsym)
            end
        else
            error("Unknown grouping $facility_grouping")
        end
        if has_schedule
            control_ix = Int64.(vec(schedule["step"]["control"]))
            nctrl = maximum(control_ix)
            if nctrl > 1
                # We have multiple controls and need to do further work.
                current_control = deepcopy(controls)
                all_controls = Vector{typeof(forces)}()
                for i = 1:nctrl
                    new_force = deepcopy(forces)
                    # Create controls for this set of wells
                    local_mrst_wells = vec(schedule["control"][i]["W"])
                    limits = Dict{Symbol, Any}()
                    found_limits = false
                    for (wno, wsym) in enumerate(well_symbols)
                        wdata = local_mrst_wells[wno]
                        wmodel = models[wsym]
                        current_control[wsym] = mrst_well_ctrl(model, wdata, is_comp, rhoS)
                        cstatus = vec(wdata["cstatus"])
                        lims = wdata["lims"]
                        if !isempty(lims)
                            limits[wsym] = convert_to_immutable_storage(lims)
                            found_limits = true
                        end
                        Ω_w = models[wsym].domain
                        WI = Ω_w.grid.perforations.WI
                        new_WI = vec(wdata["WI"])
                        if all(cstatus) && all(WI .== new_WI)
                            new_force[wsym] = nothing
                        else
                            # Set mask to new / static so that static*mask = new.
                            # In addition: Account for completion closures.
                            wi_mask = vec(new_WI./WI)
                            for ix in eachindex(wi_mask)
                                if (!cstatus[ix] || !isfinite(wi_mask[ix]))
                                    wi_mask[ix] = 0
                                end
                            end
                            new_force[wsym] = build_forces(wmodel, mask = PerforationMask(wi_mask))
                        end
                    end
                    # Now copy into the corresponding facilit(y/ies)
                    if !found_limits
                        limits = Dict()
                    end
                    for (fsymbol, wsymbols) in zip(facility_symbols, facility_owned_wells)
                        ctrls = facility_subset(wsymbols, current_control)
                        WG = models[fsymbol]
                        limits_local = facility_subset(wsymbols, limits)
                        new_force[fsymbol] = build_forces(WG, control = ctrls, limits = limits_local)
                    end
                    push!(all_controls, new_force)
                end
                if nctrl == 1
                    # No need to make a compilcated vector.
                    forces = only(all_controls)
                else
                    forces = all_controls[control_ix]
                end
            end
        end
    end
    return (models, parameters, initializer, timesteps, forces, mrst_data)
end

function facility_subset(well_symbols, controls)
    ctrls = Dict()
    for k in keys(controls)
        if any(well_symbols .== k)
            ctrls[k] = controls[k]
        end
    end
    return ctrls
end

function mrst_well_ctrl(model, wdata, is_comp, rhoS)
    t_mrst = wdata["val"]
    is_injector = wdata["sign"] > 0
    is_shut = wdata["status"] < 1
    comp_i = vec(wdata["compi"])
    phases = Jutul.get_phases(model.system)
    nph = length(phases)
    name = wdata["name"]

    if is_shut
        @debug "$name: Shut well (requested)"
        ctrl = DisabledControl()
    else
        wt = wdata["type"]
        is_rate_ctrl = true
        if wt == "bhp"
            target = BottomHolePressureTarget(t_mrst)
            # Not rate controlled
            is_rate_ctrl = false
        elseif wt == "rate"
            target = TotalRateTarget(t_mrst)
        elseif wt == "wrat"
            target = SurfaceWaterRateTarget(t_mrst)
        elseif wt == "orat"
            target = SurfaceOilRateTarget(t_mrst)
        elseif wt == "grat"
            target = SurfaceGasRateTarget(t_mrst)
        elseif wt == "lrat"
            target = SurfaceLiquidRateTarget(t_mrst)
        else
            error("$wt target is not supported.")
        end

        if is_rate_ctrl && t_mrst == 0.0
            @debug "$name: Shut well (zero rate)"
            ctrl = DisabledControl()
        elseif is_injector
            if is_comp
                ci = copy(vec(wdata["components"]))
                props = model.system.equation_of_state.mixture.properties
                ci = map((x, c) -> max(c.mw*x, 1e-10), ci, props)
                ct = copy(ci)
                if nph == 3
                    # Should go at end - need better logic if this isn't either one or zero
                    c_water = comp_i[1]
                    # hc_weight = 1.0 - c_water
                    # ct = ct.*hc_weight
                    push!(ct, c_water)
                    # normalize!(ct, 1)
                end
                ct = normalize(ct, 1)
            else
                ct = comp_i
            end
            if haskey(wdata, "rhoS") && length(wdata["rhoS"]) > 0
                rhoSw = vec(wdata["rhoS"])
            else
                rhoSw = vec(rhoS)
            end
            rhoS_inj = sum(comp_i.*rhoSw)
            ctrl = InjectorControl(target, ct, density = rhoS_inj)
        else
            ctrl = ProducerControl(target)
        end
    end
    return ctrl
end