if(CMAKE_HIP_COMPILER)
    set_source_files_properties(electrons.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
	add_library(net87_gpu STATIC electrons.cu)

	set_target_properties(net87_gpu PROPERTIES
		CUDA_SEPARABLE_COMPILATION ON
		CUDA_RESOLVE_DEVICE_SYMBOLS ON
	)

	target_include_directories(net87_gpu PRIVATE ${SPH_DIR})
	target_include_directories(net87_gpu PRIVATE ${CSTONE_DIR})
endif()
