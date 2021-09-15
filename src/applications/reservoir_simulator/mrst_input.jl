using GLMakie
using MAT # .MAT file loading
export get_minimal_tpfa_grid_from_mrst, plot_mrstdata, plot_interactive, get_test_setup, get_well_from_mrst_data
using SparseArrays # Sparse pattern
using GLMakie

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

    N = geo.neighbors
    function get_vec(d)
        if isa(d, AbstractArray)
            return vec(d)
        else
            return [d]
        end
    end
    # Cells
    cell_centroids = geo.cell_centroids

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
        T = vec(exported["T"])
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


function get_test_setup(grid_name; case_name = "single_phase_simple", context = "cpu", timesteps = [1.0, 2.0], pvfrac = 0.05, fuse_flux = false, kwarg...)
    G = get_minimal_tpfa_grid_from_mrst(grid_name, fuse_flux = fuse_flux)
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
        state0 = setup_state(model, init)
        # Model parameters
        parameters = setup_parameters(model)
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
        # Model parameters
        parameters = setup_parameters(model)
    else
        error("Unknown case $case_name")
    end
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

function plot_mrstdata(mrst_grid, data)
    if any([t == "cartGrid" for t in mrst_grid["type"]])
        cartDims = Int64.(mrst_grid["cartDims"])
        if mrst_grid["griddim"] == 2 || cartDims[end] == 1
            fig = Figure()

            Axis(fig[1, 1])
            heatmap!(reshape(data, cartDims[1:2]...))
            ul = (minimum(data), maximum(data))
            # vertical colorbars
            Colorbar(fig[1, 2], limits = ul)
            ax = fig
        else
            ax = volume(reshape(data, cartDims...), algorithm = :mip, colormap = :jet)
        end
        plot
    else
        println("Non-Cartesian plot not implemented.")
        ax = nothing
    end
    return ax
end

