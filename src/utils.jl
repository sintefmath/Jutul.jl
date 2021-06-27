export convert_to_immutable_storage, gravity_constant

const gravity_constant = 9.80665

function convert_to_immutable_storage(dct::AbstractDict)
    for (key, value) in dct
        dct[key] = convert_to_immutable_storage(value)
    end
    return (; (Symbol(key) => value for (key, value) in dct)...)
end

function convert_to_immutable_storage(v::Any)
    # Silently do nothing
    return v
end


"Apply a function to each element in the fastest possible manner."
function fapply!(out, f, inputs...)
    # Example:
    # x, y, z equal length
    # then fapply!(z, *, x, y) is equal to a parallel call of
    # z .= x.*y
    # If JuliaLang Issue #19777 gets resolved we can get rid of fapply!
    Threads.@threads for i in eachindex(out)
        @inbounds out[i] = f(map((x) -> x[i], inputs)...)
    end
end

function fapply!(out::CuArray, f, inputs...)
    # Specialize fapply for GPU to get automatic kernel computation
    @. out = f(inputs...)
end

function as_cell_major_matrix(v, n, m, model::SimulationModel, offset = 0)
    transp = !is_cell_major(matrix_layout(model.context))
    get_matrix_view(v, n, m, transp, offset)
end

function get_matrix_view(v::AbstractVector, n, m, transp = false, offset = 0)
    r_l = view(v, (offset+1):(offset + n*m))
    if transp
        v = reshape(r_l, m, n)'
    else
        v = reshape(r_l, n, m)
    end
    return v
end

function get_matrix_view(v, n, m, transp = false, offset = 0)
    if transp
        v = v'
    end
    return v
end


function get_row_view(v::AbstractVector, n, m, row, transp = false, offset = 0)
    v = get_matrix_view(v, n, m, transp, offset)
    view(v, row, :)
end

function get_convergence_table(errors::AbstractDict)
    # Already a dict
    conv_table_fn(errors, true)
end

function get_convergence_table(errors)
    d = OrderedDict()
    d[:Base] = errors
    conv_table_fn(d, false)
end

function conv_table_fn(model_errors, has_models = false)
    if !has_models
        # Make the code easier to have for both multimodel and single model case.
        header = ["Equation", "Value", "Tolerance"]
    else
        header = ["Model", "Equation", "Value", "Tolerance"]
    end
    tbl = []
    for (model, errors) in model_errors
        for (mix, eq) in enumerate(errors)
            for (i, e) in enumerate(eq.error)
                if i == 1
                    nm = String(eq.name)
                    tt = eq.tolerance
                else
                    nm = ""
                    tt = ""
                end
                if has_models
                    if mix == 1 && i == 1
                        m = String(model)
                    else
                        m = ""
                    end
                    t = [m nm e tt]
                else
                    t = [nm e tt]
                end
                push!(tbl, t)
            end
        end
    end
    tbl = vcat(tbl...)
    return pretty_table(tbl, header = header)
end