function select_nonlinear_relaxation(model, rel_type, reports, relaxation)
    return relaxation
end

function select_nonlinear_relaxation(model, rel_type::StagnationRelaxation, reports, ω)
    if length(reports) > 1
        (; tol, dw, w_max, w_min) = rel_type
        e_old = error_sum_scaled(model, reports[end-1][:errors])
        e_new = error_sum_scaled(model, reports[end][:errors])
        if (e_old - e_new)/max(e_old, 1e-20) < tol
            ω = ω - dw
        else
            ω = ω + dw
        end
        ω = clamp(ω, w_min, w_max)

    end
    return ω
end


function error_sum_scaled(model, rep)
    err_sum = 0.0
    for r in rep
        tol = r.tolerances
        crit = r.criterions
        for (k, v) in tol
            err_sum += maximum(crit[k].errors)/v
        end
    end
    return err_sum
end
