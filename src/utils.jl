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
    @threads for i in eachindex(out)
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

function get_matrix_view(v0, n, m, transp = false, offset = 0)
    if size(v0, 2) == 1 && n != 1
        r_l = view(v0, (offset+1):(offset + n*m))
        if transp
            v = reshape(r_l, m, n)'
        else
            v = reshape(r_l, n, m)
        end
    else
        v = view(v0, (offset+1):(offset+n), :)
        if transp
            v = v'
        end
    end
    return v
end


function check_increment(dx, key)
    if any(!isfinite, dx)
        bad = findall(isfinite.(dx) .== false)
        n_bad = length(bad)
        n = min(10, length(bad))
        bad = bad[1:n]
        @warn "$key: $n_bad non-finite values found. Indices: (limited to 10) $bad"
    end
end

# function get_row_view(v::AbstractVector, n, m, row, transp = false, offset = 0)
#     v = get_matrix_view(v, n, m, transp, offset)
#     view(v, row, :)
# end

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
    header = ["Equation", "‖R‖", "ϵ"]
    alignment = [:l, :r, :r]
    if has_models
        # Make the code easier to have for both multimodel and single model case.
        header = ["Model", header...]
        alignment = [:l, alignment...]
    end
    tbl = []
    tols = Vector{Float64}()
    body_hlines = Vector{Int64}()
    pos = 1
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
                push!(tols, eq.tolerance)
                pos += 1
            end
            push!(body_hlines, pos-1)
        end
    end
    tbl = vcat(tbl...)

    rpos = (2 + Int64(has_models))
    nearly_factor = 10
    function not_converged(data, i, j)
        if j == rpos
            d = data[i, j]
            t = tols[i]
            return d > t && d > 10*t
        else
            return false
        end
    end
    h1 = Highlighter(f = not_converged,
                     crayon = crayon"red" )

    function nearly_converged(data, i, j)
        if j == rpos
            d = data[i, j]
            t = tols[i]
            return d > t && d < nearly_factor*t
        else
            return false
        end
    end
    h2 = Highlighter(f = nearly_converged,
                     crayon = crayon"yellow")

    function converged(data, i, j)
        if j == rpos
            return data[i, j] <= tols[i]
        else
            return false
        end
    end
    h3 = Highlighter(f = converged,
                     crayon = crayon"green")

    highlighers = (h1, h2, h3)
    return pretty_table(tbl, header = header,
                             alignment = alignment, 
                             body_hlines = body_hlines,
                             highlighters = highlighers, 
                             formatters = ft_printf("%2.4e"))
end