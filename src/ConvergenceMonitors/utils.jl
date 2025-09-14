# TODO: dispatch on model type (scalar model/multimodel)

"""
    get_multimodel_residuals(report)

Get all residual norms for all equations of all models from a nonlinear
iteration report, scaled by their respective toelrances. The function returns a
dict of dicts with the residual norms for each equation on the form

    residuals[model][equation_type][equation][norm] = res/tol
"""
function residual_measures(report, scaled=true)

    models = keys(report)

    measures = Dict()

    for model in models
        rsd = residual_measures(report[model], scaled)
        measures[model] = rsd
    end

    return measures

end

"""
    get_model_residuals(report)


Get all residual norms for all equations of a model from a nonlinear
iteration report, scaled by their respective toelrances. The function returns a
dict of dicts with the residual norms for each equation on the form

    residuals[equation_type][equation][norm] = res/tol
"""
function residual_measures(report::Vector, scaled=true)

    measures = Dict()

    for eq_report in report

        equation = eq_report[:name]
        criterions = eq_report[:criterions]
        tolerances = eq_report[:tolerances]
        residual_norms = keys(criterions)

        equation_residuals = Dict()
    
        for res_norm in residual_norms

            rsd = criterions[res_norm].errors
            scl = scaled ? tolerances[res_norm] : 1.0
            nms = criterions[res_norm].names
            for i in eachindex(rsd)
                α, r = nms[i], rsd[i]
                α = process_name(α)
                if !haskey(equation_residuals, α)
                    equation_residuals[α] = Dict()
                end
                equation_residuals[α][res_norm] = r/scl
            end

        end

        measures[equation] = equation_residuals
    
    end

    return measures

end

"""
    process_name(name)

Process a names to be suibale as dictionary keys.
"""
function process_name(name)
    
    name = string(name)
    name = replace(name, " " => "_", "(" => "", ")" => "")
    name = Symbol(name)

end

"""
    flatten_dict(input_dict::Dict, separator::String = ".", trail = [])

Flatten a dict of dicts into a vector of values and a vector of names. The names
are on the format `"key1<separator>key2<separator>key3"` and the values are the
corresponding values in the dict.
"""
function flatten_dict(input_dict::Dict, separator::String = ".", trail = [])
    values = Float64[]
    names = String[]

    for (key, value) in input_dict
        current_trail = vcat(trail, string(key))
        if value isa Dict
            sub_values, sub_names = flatten_dict(value, separator, current_trail)
            append!(values, sub_values)
            append!(names, sub_names)
        else
            push!(values, value)
            push!(names, join(current_trail, separator))
        end
    end

    return values, names
end