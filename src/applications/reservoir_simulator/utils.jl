reservoir_model(model) = model
reservoir_storage(model, storage) = storage
reservoir_storage(model::MultiModel, storage) = storage.Reservoir
reservoir_model(model::MultiModel) = model.models.Reservoir
