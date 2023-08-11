function setup_helper_equation_storage!(storage, r, mm::MultiModel; offset = 0)
    for (k, model) in pairs(mm.models)
        offset = setup_helper_equation_storage!(storage[k], r, model, offset = offset)
    end
end
