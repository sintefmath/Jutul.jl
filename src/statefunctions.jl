using ExprTools, LightGraphs
"""
Designate the function as updating a secondary variable.

The function is then declared, in addition to helpers that allows
checking what the dependencies are and unpacking the dependencies from state.

If we define the following function annotated with the macro:
@terv_secondary function update_as_secondary!(target, var::MyVarType, model, param, a, b, c)
    @. target = a + b / c
end

The macro also defines: 
function get_dependencies(var::MyVarType, model)
   return [:a, :b, :c]
end

function update_as_secondary!(array_target, var::MyVarType, model, parameters, state)
    update_as_secondary!(array_target, var, model, parameters, state.a, state.b, state.c)
end

Note that the names input arguments beyond the parameter dict matter, as these will be fetched from state.
"""
macro terv_secondary(ex)
    def = splitdef(ex)
    args = def[:args]
    # Define filters to strip the type spec (if any)
    function myfilter(x::Symbol)
        x
    end
    function myfilter(x::Expr)
        x.args[1]
    end

    deps = map(myfilter, args[5:end])
    # Pick variable + model
    variable_sym = args[2]
    model_sym = args[3]
    @debug "Building evaluator for $variable_sym"

    # Define get_dependencies function
    dep_def = deepcopy(def)
    dep_def[:name] = :get_dependencies
    dep_def[:args] = [variable_sym, model_sym]
    dep_def[:body] = deps
    ex_dep = combinedef(dep_def)
    # Define update_as_secondary! function
    upd_def = deepcopy(def)
    upd_def[:name] = :update_secondary_variable!
    upd_def[:args] = [:array_target, variable_sym, model_sym, :parameters, :state]
    # value, var, model, parameters, arg1, arg2
    tmp = "update_as_secondary!(array_target, "
    tmp *= String(myfilter(variable_sym))
    tmp *= ", "
    tmp *= String(myfilter(model_sym))
    tmp *= ", parameters"

    for s in deps
        tmp *= ", state."*String(s)
    end
    tmp *= ")"
    upd_def[:body] = Meta.parse(tmp)
    ex_upd = combinedef(upd_def)

    quote 
        $ex
        $ex_dep
        $ex_upd
    end |> esc 
end

function update_secondary_variables!(storage, model)
    state = storage.state
    parameters = storage.parameters
    update_secondary_variables!(state, model, parameters)
end

function update_secondary_variables!(state, model, parameters)
    for (symbol, var) in model.secondary_variables
        update_secondary_variable!(state[symbol], var, model, parameters, state)
    end
end

# Initializers
function select_secondary_variables(domain, system, formulation)
    sf = OrderedDict()
    select_secondary_variables!(sf, domain, system, formulation)
    return sf
end

function select_secondary_variables!(sf, domain, system, formulation)
    select_secondary_variables_domain!(sf, domain, system, formulation)
    select_secondary_variables_domain!(sf, domain, system, formulation)
    select_secondary_variables_domain!(sf, domain, system, formulation)
end

function select_secondary_variables_domain!(sf, domain, system, formulation)

end

function select_secondary_variables_system!(sf, domain, system, formulation)

end

function select_secondary_variables_formulation!(sf, domain, system, formulation)

end

## Definition
function select_primary_variables(domain, system, formulation)
    sf = OrderedDict()
    select_primary_variables!(sf, domain, system, formulation)
    return sf
end

function select_primary_variables!(sf, domain, system, formulation)
    select_primary_variables_domain!(sf, domain, system, formulation)
    select_primary_variables_system!(sf, domain, system, formulation)
    select_primary_variables_formulation!(sf, domain, system, formulation)
end

function select_primary_variables_domain!(sf, domain, system, formulation)

end

function select_primary_variables_system!(sf, domain, system, formulation)

end

function select_primary_variables_formulation!(sf, domain, system, formulation)

end


function minimum_output_variables(domain, system, formulation, primary_variables, secondary_variables)
    minimum_output_variables(system, primary_variables)
end

function minimum_output_variables(system, primary_variables)
    # Default: Output all primary variables
    [i for i in keys(primary_variables)]
end

function map_level(primary_variables, secondary_variables, output_level)
    pkeys = [i for i in keys(primary_variables)]
    skeys = [i for i in keys(secondary_variables)]
    if output_level == :All
        out = vcat(pkeys, skeys)
    elseif output_level == :PrimaryVariables
        out = pkeys
    elseif output_level == :SecondaryVariables
        out = skeys
    else
        out = [output_level]
    end
end

function select_output_variables(domain, system, formulation, primary_variables, secondary_variables, output_level)
    outputs = minimum_output_variables(domain, system, formulation, primary_variables, secondary_variables)
    if !isnothing(output_level)
        if isa(output_level, Symbol)
            output_level  = [output_level]
        end
        for levels in output_level
            mapped = map_level(primary_variables, secondary_variables, levels)
            outputs = vcat(outputs, mapped)
        end
    end
    return outputs
end

function sort_secondary_variables!(model::TervModel)
    # Do nothing for general case.
end

function sort_secondary_variables!(model::SimulationModel)
    primary = model.primary_variables
    secondary = model.secondary_variables
    
    edges = []
    nodes = []
    for key in keys(primary)
        push!(nodes, key)
        push!(edges, []) # No dependencies for primary variables.
    end
    for (key, var) in secondary
        dep = get_dependencies(var, model)
        push!(nodes, key)
        push!(edges, dep)
    end
    order = sort_symbols(nodes, edges)
    @debug "Variable ordering determined: $(nodes[order])"
    np = length(primary)
    for i in 1:np
        @assert order[i] <= np "Primary variables should come in the first $np entries in ordering. Something is very wrong."
    end
    # Skip primary variable indices - these always come first.
    order = order[order .> np]
    # Offset by primary variables
    @. order -= np
    @. secondary.keys = secondary.keys[order]
    @. secondary.vals = secondary.vals[order]
end

function sort_symbols(symbols, deps)
    @assert length(symbols) == length(deps)
    n = length(symbols)
    graph = SimpleDiGraph(n)
    for (i, dep) in enumerate(deps)
        for d in dep
            pos = findall(symbols .== d)[]
            @assert length(pos) == 1 "Symbol $d must appear exactly once"
            add_edge!(graph, i, pos)
        end
    end
    reverse(topological_sort_by_dfs(graph))
end