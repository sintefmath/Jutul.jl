export get_minimal_tpfa_grid_from_mrst, plot_interactive, get_test_setup, get_well_from_mrst_data
export setup_case_from_mrst

struct MRSTPlotData
    faces::Array
    vertices::Array
    data::Vector
end

function get_minimal_tpfa_grid_from_mrst(name::String; relative_path=true, perm = nothing, poro = nothing, volumes = nothing, extraout = false, kwarg...)
    if relative_path
        fn = string(dirname(pathof(Terv)), "/../data/testgrids/", name, ".mat")
    else
        fn = name
    end
    @debug "Reading MAT file $fn..."
    exported = MAT.matread(fn)
    @debug "File read complete. Unpacking data..."
    g = MRSTWrapMesh(exported["G"])
    geo = tpfv_geometry(g)

    function get_vec(d)
        if isa(d, AbstractArray)
            return vec(d)
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
    if haskey(exported, "T") && length(exported["T"]) > 0
        @debug "Found precomputed transmissibilities, reusing"
        T_raw = exported["T"]
        if isa(T_raw, AbstractFloat)
            T_raw = [T_raw]
        end
        T = vec(T_raw)
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

function get_well_from_mrst_data(mrst_data, system, ix; volume = 1, extraout = false, simple = false, kwarg...)
    W_mrst = mrst_data["W"][ix]
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
    z = cell_centroids[3, rc]

    if simple
        # For simple well, distance from ref depth to perf
        dz = ref_depth .- z
        W = SimpleWell(rc, WI = WI, dz = dz)
        wmodel = SimulationModel(W, system; kwarg...)
        flow = TwoPointPotentialFlow(nothing, nothing, TrivialFlow(), W)
    else
        # For a MS well, this is the drop from the perforated cell center to the perforation (assumed zero here)
        dz = zeros(length(rc))
        W = MultiSegmentWell(volume*ones(n), rc, WI = WI, reference_depth = ref_depth, dz = dz)

        z = vcat(ref_depth, z)
        flow = TwoPointPotentialFlow(SPU(), MixedWellSegmentFlow(), TotalMassVelocityMassFractionsFlow(), W, nothing, z)
    end
    disc = (mass_flow = flow,)
    wmodel = SimulationModel(W, system, discretization = disc; kwarg...)
    if extraout
        out = (wmodel, W_mrst)
    else
        out = wmodel
    end
    return out
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
    @assert isa(context, TervContext)

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
    @info "Loading model" keys(mrst_data)
    if haskey(mrst_data, "mixture")
        f = model_from_mat_comp
    elseif haskey(mrst_data, "fluid")
        f = model_from_mat_fluid_immiscible
    elseif haskey(mrst_data, "deck")
        f = model_from_mat_deck
    else
        error("I don't know how this model was made")
    end
    return f(G, mrst_data, res_context)
end

function model_from_mat_comp(G, mrst_data, res_context)
    ## Set up reservoir part
    f = mrst_data["fluid"]
    nkr = vec(f["nkr"])
    rhoS = vec(f["rhoS"])

    mixture = mrst_data["mixture"]
    comps = mixture["components"]
    names = vec(mixture["names"])
    n = length(comps)

    components = map(x -> MolecularProperty(x["mw"], x["pc"], x["Tc"], x["Vc"], x["acf"]), comps)
    mixture = MultiComponentMixture(components, names = names)
    eos = GenericCubicEOS(mixture)

    liq = LiquidPhase()
    vap = VaporPhase()
    sys = TwoPhaseCompositionalSystem([liq, vap], eos)

    model = SimulationModel(G, sys, context = res_context)

    if haskey(f, "sgof")
        sgof = f["sgof"]
    else
        sgof = []
    end

    if isempty(sgof)
        kr = BrooksCoreyRelPerm(sys, nkr)
    else
        s, krt = preprocess_relperm_table(sgof)
        kr = TabulatedRelPermSimple(s, krt)
    end

    # p = model.primary_variables
    # p[:Pressure] = Pressure(max_rel = 0.2)
    s = model.secondary_variables
    s[:RelativePermeabilities] = kr
    
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
        z0 = state0["components"]'
        init = Dict(:Pressure => p0, :OverallMoleFractions => z0)
    else
        s0 = state0["s"]'
        init = Dict(:Pressure => p0, :Saturations => s0)
    end
    return init
end

function setup_case_from_mrst(casename; simple_well = false, block_backend = true, facility_grouping = :onegroup, kwarg...)
    G, mrst_data = get_minimal_tpfa_grid_from_mrst(casename, extraout = true, fuse_flux = false; kwarg...)
    function setup_res(G, mrst_data; block_backend = false, use_groups = false)
        bctx = DefaultContext(matrix_layout = BlockMajorLayout())
        # bctx = DefaultContext(matrix_layout = UnitMajorLayout())
        dctx = DefaultContext()
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
    
    # Set up initializers
    models = OrderedDict()
    initializer = Dict()
    forces = Dict()
    
    
    model, init, param_res = setup_res(G, mrst_data; block_backend = block_backend, use_groups = true)
    is_comp = haskey(init, :OverallMoleFractions)

    dt = mrst_data["dt"]
    if isa(dt, Real)
        dt = [dt]
    end
    timesteps = vec(dt)
    res_context = model.context
    w_context = DefaultContext()
    
    initializer[:Reservoir] = init
    forces[:Reservoir] = nothing
    models[:Reservoir] = model
    

    well_symbols = map((x) -> Symbol(x["name"]), vec(mrst_data["W"]))
    num_wells = length(well_symbols)
    
    parameters = Dict{Symbol, Any}()
    parameters[:Reservoir] = param_res
    controls = Dict()
    sys = model.system
    for i = 1:num_wells
        sym = well_symbols[i]
    
        wi, wdata = get_well_from_mrst_data(mrst_data, sys, i, 
                extraout = true, volume = 1e-2, simple = simple_well, context = w_context)
        wc = wi.domain.grid.perforations.reservoir

        sv = wi.secondary_variables
        sv[:PhaseMassDensities] = model.secondary_variables[:PhaseMassDensities]
        sv[:PhaseViscosities] = model.secondary_variables[:PhaseViscosities]
    
        pw = wi.primary_variables
        pw[:Pressure] = Pressure(max_rel = 0.2)
    
        models[sym] = wi
    
        t_mrst = wdata["val"]
        is_injector = wdata["sign"] > 0
        is_shut = wdata["status"] < 1
        if wdata["type"] == "rate"
            target = SinglePhaseRateTarget(t_mrst, AqueousPhase())
        else
            target = BottomHolePressureTarget(t_mrst)
        end
    
        if is_shut
            println("Shut well")
            ctrl = DisabledControl()
        elseif is_injector
            if is_comp
                ci = wdata["components"]
            else
                ci = wdata["compi"]
            end
            ctrl = InjectorControl(target, ci)
        else
            ctrl = ProducerControl(target)
        end
        param_w = setup_parameters(wi)
        param_w[:reference_densities] = vec(param_res[:reference_densities])

        w0 = Dict{Symbol, Any}(:Pressure => mean(init[:Pressure]), :TotalMassFlux => 1e-12)
        if is_comp
            w0[:OverallMoleFractions] = vec(init[:OverallMoleFractions][:, wc[1]])
        elseif haskey(init, :Saturations)
            w0[:Saturations] = vec(init[:Saturations][:, wc[1]])
        end

        parameters[sym] = param_w
        controls[sym] = ctrl
        forces[sym] = nothing
        initializer[sym] = w0
    end
    #
    mode = PredictionMode()
    F0 = Dict(:TotalSurfaceMassRate => 0.0)


    function add_facility!(wsymbols, sym)
        g, ctrls = facility_subset(wsymbols, controls)
        WG = SimulationModel(g, mode)
        facility_forces = build_forces(WG, control = ctrls)
        # Specifics
        @assert !haskey(models, sym)
        models[sym] = WG
        forces[sym] = facility_forces
        # Generics
        initializer[sym] = F0
        parameters[sym] = setup_parameters(WG)
    end

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

    return (models, parameters, initializer, timesteps, forces, mrst_data)
end


function facility_subset(well_symbols, controls)
    g = WellGroup(well_symbols)
    ctrls = Dict()
    for k in keys(controls)
        if any(well_symbols .== k)
            ctrls[k] = controls[k]
        end
    end
    return g, ctrls
end