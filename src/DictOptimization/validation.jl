    function check_limit(parameter_name, initial::Number, limit::Array; is_max::Bool, is_rel::Bool)
        name = limit_name(is_rel, is_max)
        error("$parameter_name was a scalar, $name was an array of size $(size(limit))")
    end

    function check_limit(parameter_name, initial::Number, limit; is_max::Bool, is_rel::Bool, context = raw"")
        name = limit_name(is_rel, is_max)
        if is_rel
            if is_max
                if limit < 1.0
                    error("$parameter_name has limit $name smaller than 1.0$context.")
                end
            else
                if limit > 1.0
                    error("$parameter_name has limit $name larger 1.0$context.")
                end
            end

        else
            if is_max
                if initial > limit
                    error("$parameter_name has limit $name smaller than initial value $initial$context.")
                end
            else
                if initial < limit
                    error("$parameter_name has limit $name larger than initial value $initial$context.")
                end
            end
        end
    end

    function check_limit(parameter_name, initial::Array, limit::Array; kwarg...)
        vsz = size(initial)
        lsz = size(limit)
        vsz == lsz || throw(DimensionMismatch("Limit ($lsz) and initial value ($vsz) mismatch $parameter_name"))
        for I in eachindex(IndexCartesian(), initial, limit)
            check_limit(parameter_name, initial[I], limit[I]; kwarg..., context = " in entry at $I")
        end
    end

    function check_limit(parameter_name, initial::Array, limit::Number; kwarg...)
        for I in eachindex(IndexCartesian(), initial)
            check_limit(parameter_name, initial[I], limit; kwarg..., context = " in entry at $I")
        end
    end

    function check_limit_pair(parameter_name, initial::Number, min_limit::Number, max_limit::Number; is_rel::Bool, context = raw"")
        max_name = limit_name(is_rel, true)
        min_name = limit_name(is_rel, false)
        if max_limit <= min_limit
            error("$parameter_name has no feasible values for $min_name = $min_limit and $max_name = $max_limit$context.")
        end
    end

    function check_limit_pair(parameter_name, initial::Array, min_limit::Number, max_limit::Number; kwarg...)
        for I in eachindex(IndexCartesian(), initial)
            check_limit_pair(parameter_name, initial[I], min_limit, max_limit, context = " in entry at $I"; kwarg...)
        end
    end

    function check_limit_pair(parameter_name, initial::Array, min_limit::Array, max_limit::Array; kwarg...)
        for I in eachindex(IndexCartesian(), initial, min_limit, max_limit)
            check_limit_pair(parameter_name, initial[I], min_limit[I], max_limit[I], context = " in entry at $I"; kwarg...)
        end
    end

    function check_limit_pair(parameter_name, initial::Array, min_limit::Number, max_limit::Array; kwarg...)
        for I in eachindex(IndexCartesian(), initial, max_limit)
            check_limit_pair(parameter_name, initial[I], min_limit, max_limit[I], context = " in entry at $I"; kwarg...)
        end
    end

    function check_limit_pair(parameter_name, initial::Array, min_limit::Array, max_limit::Number; kwarg...)
        for I in eachindex(IndexCartesian(), initial, min_limit)
            check_limit_pair(parameter_name, initial[I], min_limit[I], max_limit, context = " in entry at $I"; kwarg...)
        end
    end
