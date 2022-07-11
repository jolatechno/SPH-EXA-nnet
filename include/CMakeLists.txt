set(NNET_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_library(nuclear STATIC nuclear-net.cpp)
target_include_directories(nuclear PRIVATE ${SPH_DIR})
target_include_directories(nuclear PRIVATE ${CSTONE_DIR})
target_compile_definitions(nuclear PRIVATE USE_MPI)

if(CMAKE_HIP_COMPILER)
	set_source_properties(nuclear-net.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
	add_library(nuclear_gpu STATIC nuclear-net.cu)

	target_include_directories(nuclear_gpu PRIVATE ${SPH_DIR})
	target_include_directories(nuclear_gpu PRIVATE ${CSTONE_DIR})
	target_compile_definitions(nuclear_gpu PRIVATE USE_MPI)	
endif()

add_subdirectory(nnet/sphexa/CUDA)
add_subdirectory(nnet/net87)