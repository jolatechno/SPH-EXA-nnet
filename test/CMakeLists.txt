set(net14_test net14_test)
add_executable(${net14_test} net14_test.cpp)

target_include_directories(${net14_test} PRIVATE ${SPH_DIR})
target_include_directories(${net14_test} PRIVATE ${CSTONE_DIR})

target_link_libraries(${net14_test} PRIVATE OpenMP::OpenMP_CXX)
target_compile_definitions(${net14_test} PRIVATE)



set(net86_test net86_test)
add_executable(${net86_test} net86_test.cpp)

target_include_directories(${net86_test} PRIVATE ${SPH_DIR})
target_include_directories(${net86_test} PRIVATE ${CSTONE_DIR})

target_link_libraries(${net86_test} PRIVATE OpenMP::OpenMP_CXX)
target_compile_definitions(${net86_test} PRIVATE)



set(net14_rate_test net14_rate_test)
add_executable(${net14_rate_test} net14_rate_test.cpp)

target_include_directories(${net14_rate_test} PRIVATE ${SPH_DIR})
target_include_directories(${net14_rate_test} PRIVATE ${CSTONE_DIR})

target_link_libraries(${net14_rate_test} PRIVATE OpenMP::OpenMP_CXX)
target_compile_definitions(${net14_rate_test} PRIVATE)



set(net87_rate_test net87_rate_test)
add_executable(${net87_rate_test} net87_rate_test.cpp)

target_include_directories(${net87_rate_test} PRIVATE ${SPH_DIR})
target_include_directories(${net87_rate_test} PRIVATE ${CSTONE_DIR})

target_link_libraries(${net87_rate_test} PRIVATE OpenMP::OpenMP_CXX)
target_compile_definitions(${net87_rate_test} PRIVATE)




set(nnet hydro-mockup)

add_executable(${nnet} hydro-mockup.cpp)

target_include_directories(${nnet} PRIVATE ${SPH_DIR})
target_include_directories(${nnet} PRIVATE ${CSTONE_DIR})

target_compile_definitions(${nnet} PRIVATE USE_MPI)
target_link_libraries(${nnet} PRIVATE OpenMP::OpenMP_CXX)



if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
	add_executable(${nnet}-cuda hydro-mockup.cpp)

	target_include_directories(${nnet}-cuda  PRIVATE ${SPH_DIR})
	target_include_directories(${nnet}-cuda  PRIVATE ${CSTONE_DIR})

	target_compile_definitions(${nnet}-cuda PRIVATE USE_MPI USE_CUDA FORCE_COMPILE_DEVICE)
	target_link_libraries(${nnet}-cuda PRIVATE OpenMP::OpenMP_CXX CUDA::cudart)
	target_link_libraries(${nnet}-cuda PRIVATE nuclear_net)
	target_link_libraries(${nnet}-cuda PRIVATE nuclear_data)
endif()