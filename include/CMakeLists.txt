set(NNET_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_library(nuclear SHARED nuclear-net.cpp)
#target_include_directories(nuclear PRIVATE ${SPH_DIR})
#target_include_directories(nuclear PRIVATE ${CSTONE_DIR})
#target_compile_definitions(nuclear PRIVATE USE_MPI)

if(CMAKE_HIP_COMPILER)
	set_source_properties(nuclear-net.cu nnet/parallel/CUDA/parallel-nuclear-net.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
	add_library(nuclear_gpu STATIC nuclear-net.cu nnet/parallel/CUDA/parallel-nuclear-net.cu)

    set_target_properties(nuclear_gpu PROPERTIES
    	CUDA_SEPARABLE_COMPILATION ON
    )

	target_include_directories(nuclear_gpu PRIVATE ${SPH_DIR})
	target_include_directories(nuclear_gpu PRIVATE ${CSTONE_DIR})
	target_compile_definitions(nuclear_gpu PRIVATE USE_MPI)
endif()