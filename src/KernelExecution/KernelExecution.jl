module KernelExecution
    using LinearAlgebra
    using SparseArrays
    using KernelAbstractions
    import Adapt
    import ForwardDiff
    import DataStructures: OrderedDict

    using ..Jutul
    import ..Jutul: AssembleOnDevice, CompactAutoDiffCache, ConservationLaw,
        ConservationLawTPFAStorage, CrossTermPair, DeviceExecutionMode,
        DiscretizedDomain, EntityCounter, EquationMajorLayout, FactorStore,
        GenericAutoDiffCache, GPUJutulContext, JutulEntity, JutulForce,
        JutulStorage,
        LinearInterpolant, LinearizedBlock, LinearizedSystem, LinearizedType,
        LocalPerspectiveAD, LocalStateAD, MultiLinearizedSystem, MultiModel,
        MultiModelLocalStateAD, NothingOnDevice, ParallelCSRContext,
        PotentialFlow, SimulationModel, Simulator, SolveFullyOnDevice,
        StaticSparsityMatrixCSR, TwoPointPotentialFlowHardCoded,
        ValueStateAD, align_cross_terms_to_linearized_system!,
        align_equations_to_linearized_system!, backend_copyto!, backend_to_host,
        build_variable_graph, colvals, convert_to_immutable_storage, data,
        entity_eachindex, float_type, forces_for_backend, forces_for_host,
        forces_for_timestep, get_simulator_model, get_simulator_storage,
        group_execution_mode, index_type, linear_solve!, linear_solve_return,
        matrix_layout, minbatch, multimodel_label, nthreads,
        number_of_entities, nzval_index_type, prepare_backend_transfer!,
        preprocess_forces, replace_values!,
        setup_equations_and_primary_variable_views,
        setup_equations_and_primary_variable_views!, setup_linearized_system!,
        setup_multimodel_maps!, sort_symbols, specialize_simulator_storage,
        submodels_symbols, synchronize, threaded_loop, threaded_loop_minbatch,
        transfer, transfer_to_backend, unpack_tag, update_secondary_variable!,
        update_secondary_variables_state!, updated_state_value, update_values!

    include("context.jl")
    include("execution.jl")

    export KernelAbstractionsContext
end
