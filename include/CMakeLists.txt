set(NNET_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_library(nuclear SHARED nuclear_net.cpp nnet/parameterization/eos/helmholtz.cpp)
target_include_directories(nuclear PRIVATE ${NNET_DIR})

if(CMAKE_HIP_COMPILER)
	set_source_properties(nuclear_net.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
	add_library(nuclear_gpu STATIC nuclear_net.cu nnet/parameterization/eos/helmholtz.cpp)

    set_target_properties(nuclear_gpu PROPERTIES
    	CUDA_SEPARABLE_COMPILATION ON
    )

	target_include_directories(nuclear_gpu PRIVATE ${NNET_DIR})
endif()

add_subdirectory(nnet/parallel)