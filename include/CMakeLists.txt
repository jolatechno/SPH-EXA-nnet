set(NNET_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_library(nuclear OBJECT nuclear-net.cpp)
target_include_directories(nuclear PRIVATE ${SPH_DIR})
target_include_directories(nuclear PRIVATE ${CSTONE_DIR})
target_compile_definitions(nuclear PRIVATE USE_MPI)


if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
	add_library(nuclear_gpu OBJECT nuclear-net.cpp)
	target_include_directories(nuclear_gpu PRIVATE ${SPH_DIR})
	target_include_directories(nuclear_gpu PRIVATE ${CSTONE_DIR})
	target_compile_definitions(nuclear_gpu PRIVATE USE_MPI FORCE_COMPILE_DEVICE)
	target_link_libraries(nuclear_gpu PRIVATE CUDA::cudart)
endif()

add_subdirectory(nnet/sphexa/CUDA)