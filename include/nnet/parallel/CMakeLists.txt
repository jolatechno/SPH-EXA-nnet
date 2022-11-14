add_library(nuclear_parallel SHARED implementation/parallel_nuclear_net_impl.cpp)
target_include_directories(nuclear_parallel PRIVATE ${NNET_DIR})
target_compile_definitions(nuclear_parallel PRIVATE USE_MPI)

if(CMAKE_HIP_COMPILER)
	set_source_properties(CUDA/parallel_nuclear_net.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
	add_library(nuclear_parallel_gpu STATIC CUDA/parallel_nuclear_net.cu)

    set_target_properties(nuclear_parallel_gpu PROPERTIES
    	CUDA_SEPARABLE_COMPILATION ON
    )

	target_include_directories(nuclear_parallel_gpu PRIVATE ${NNET_DIR})
	target_compile_definitions(nuclear_parallel_gpu PRIVATE USE_MPI)
endif()