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


function get_minimal_tpfa_grid_from_mrst(name::String; relative_path=true, perm = nothing, poro = nothing, volumes = nothing, extraout = false, fuse_flux = false, grav_on = true)
    if relative_path
        fn = string(dirname(pathof(Terv)), "/../data/testgrids/", name, ".mat")
    else
        fn = name
    end
    @debug "Reading MAT file $fn..."
    exported = MAT.matread(fn)
    @debug "File read complete. Unpacking data..."

    N = exported["G"]["faces"]["neighbors"]
    N = Int64.(N)
    internal_faces = (N[:, 2] .> 0) .& (N[:, 1] .> 0)
    N = copy(N[internal_faces, :]')
    
    function get_vec(d)
        if isa(d, AbstractArray)
            return vec(d)
        else
            return [d]
        end
    end
    # get_cell_faces(N)
    # get_cell_neighbors(N)
    
    # Cells
    cell_centroids = copy((exported["G"]["cells"]["centroids"])')
    # Faces
    face_centroids = copy((exported["G"]["faces"]["centroids"][internal_faces, :])')
    face_areas = vec(exported["G"]["faces"]["areas"][internal_faces])
    face_normals = exported["G"]["faces"]["normals"][internal_faces, :]./face_areas
    face_normals = copy(face_normals')
    if isnothing(perm)
        perm = copy((exported["rock"]["perm"])')
    end

    # Deal with cell data
    if isnothing(poro)
        poro = get_vec(exported["rock"]["poro"])
    end
    if isnothing(volumes)
        volumes = get_vec(exported["G"]["cells"]["volumes"])
    end
    pv = poro.*volumes
    nc = length(pv)

    @debug "Data unpack complete. Starting transmissibility calculations."
    # Deal with face data
    T_hf = compute_half_face_trans(cell_centroids, face_centroids, face_normals, face_areas, perm, N)
    T = compute_face_trans(T_hf, N)
    G = MinimalTPFAGrid(pv, N)
    if size(cell_centroids, 1) == 3 && grav_on
        z = cell_centroids[3, :]
        g = gravity_constant
    else
        z = nothing
        g = nothing
    end
    if fuse_flux
        ft = DarcyMassMobilityFlowFused()
    else
        ft = DarcyMassMobilityFlow()
    end
    flow = TwoPointPotentialFlow(SPU(), TPFA(), ft, G, T, z, g)
    disc = (mass_flow = flow,)
    D = DiscretizedDomain(G, disc)

    if extraout
        return (D, exported)
    else
        return D
    end
end

function get_well_from_mrst_data(mrst_data, system, ix; volume = 1, extraout = false, simple = false)
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
        wmodel = SimulationModel(W, system)
        flow = TwoPointPotentialFlow(nothing, nothing, TrivialFlow(), W)
    else
        # For a MS well, this is the drop from the perforated cell center to the perforation (assumed zero here)
        dz = zeros(length(rc))
        W = MultiSegmentWell(volume*ones(n), rc, WI = WI, reference_depth = ref_depth, dz = dz)

        z = vcat(ref_depth, z)
        flow = TwoPointPotentialFlow(SPU(), MixedWellSegmentFlow(), TotalMassVelocityMassFractionsFlow(), W, nothing, z)
    end
    disc = (mass_flow = flow,)
    wmodel = SimulationModel(W, system, discretization = disc)
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
                SourceTerm(nc, -irate)]
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

function plot_interactive(mrst_grid, states; plot_type = nothing)
    fig = Figure()
    data = states[1]
    datakeys = collect(keys(data))
    state_index = Node{Int64}(1)
    prop_name = Node{Symbol}(datakeys[1])
    loop_mode = Node{Int64}(0)

    menu = Menu(fig, options = datakeys)
    nstates = length(states)

    function change_index(ix)
        tmp = max(min(ix, nstates), 1)
        sl_x.selected_index = tmp
        state_index[] = tmp
        notify(state_index)
        return tmp
    end

    function increment_index(inc = 1)
        change_index(state_index.val + inc)
    end

    # funcs = [sqrt, x->x^2, sin, cos]
    # menu2 = Menu(fig, options = zip(["Square Root", "Square", "Sine", "Cosine"], funcs))
    fig[1, 1] = vgrid!(
        Label(fig, "Property", width = nothing),
        menu,
        # Label(fig, "Function", width = nothing),
        # menu2
        ; tellheight = false, width = 200)
    
    sl_x = Slider(fig[2, 2], range = 1:nstates, value = state_index, snap = true)
    # point = sl_x.value
    on(sl_x.selected_index) do n
        state_index[] = sl_x.selected_index.val
    end
    ax = Axis(fig[1, 2])
    ys = @lift(select_data(mrst_grid, states[$state_index], $prop_name))
    scat = Makie.heatmap!(ax, ys, label = "COLORBARLABEL")
    cb = Colorbar(fig[1, 3], scat, vertical = true, width = 30)

    on(menu.selection) do s
        prop_name[] = s
        autolimits!(ax)
    end
    # on(menu2.selection) do s
    # end
    # menu2.is_open = true

    function loop(a)
        # looping = !looping
        # println("Loop function called")
        if false
            if loop_mode.val > 0
                # println("Doing loop")
                start = state_index.val
                if start == nstates
                    start = 1
                end
                for i = start:nstates
                    newindex = increment_index()
                    if newindex > nstates
                        break
                    end
                    notify(state_index)
                    force_update!()
                    sleep(1/30)
                end
            end
        end
    end

    # @lift(loop($loop_mode))

    fig[2, 1] = buttongrid = GridLayout(tellwidth = false)
    rewind = Button(fig, label = "⏪")
    on(rewind.clicks) do n
        increment_index(-nstates)
    end
    prev = Button(fig, label = "◀️")
    on(prev.clicks) do n
        increment_index(-1)
    end

    play = Button(fig, label = "⏯️")
    on(play.clicks) do n
        println("Play button is not implemented.")
        # loop_mode[] = loop_mode.val + 1
    end
    next =   Button(fig, label = "▶️")
    on(next.clicks) do n
        increment_index()
    end
    ffwd = Button(fig, label = "⏩")
    on(ffwd.clicks) do n
        increment_index(nstates)
    end
    buttons = buttongrid[1, 1:5] = [rewind, prev, play, next, ffwd]
    
    fig
    return fig
end

function select_data(G, state, fld)
    v = state[fld]
    m = get_vector(G, v)
    return m
end

function get_vector(G, d::Vector)
    cartDims = Int64.(G["cartDims"])
    reshape(d, cartDims[1:2]...)
end

function get_vector(G, d::Matrix)
    get_vector(G, d[1, :])
end

function plotter!(ax, G, data)
    cartDims = Int64.(G["cartDims"])
    if G["griddim"] == 2 || cartDims[end] == 1
        p = heatmap!(ax, data)
        # p = heatmap!(ax, reshape(data, cartDims[1:2]...))
    else
        p = volume!(ax, reshape(data, cartDims...), algorithm = :mip)
    end
    return p
end
