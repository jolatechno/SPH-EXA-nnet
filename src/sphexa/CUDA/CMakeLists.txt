if(CMAKE_HIP_COMPILER)
    set_source_files_properties(nuclear-data-gpu.cu nuclear-net.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
	add_library(nuclear_net nuclear-net.cu ${})
	target_include_directories(nuclear_net PRIVATE ${SPH_DIR})
	target_include_directories(nuclear_net PRIVATE ${CSTONE_DIR})
    target_compile_definitions(nuclear_net PRIVATE USE_MPI)

	add_library(nuclear_data nuclear-data-gpu.cu ${})
	target_include_directories(nuclear_data PRIVATE ${SPH_DIR})
	target_include_directories(nuclear_data PRIVATE ${CSTONE_DIR})
    target_compile_definitions(nuclear_net PRIVATE USE_MPI)

    # set_property(TARGET gather_obj cuda_find_neighbors_obj PROPERTY HIP_ARCHITECTURES gfx908)
endif()
