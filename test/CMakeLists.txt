set(net14_test net14_test)
add_executable(${net14_test} net14_test.cpp)

target_include_directories(${net14_test} PRIVATE ${SPH_DIR})
target_include_directories(${net14_test} PRIVATE ${CSTONE_DIR})
target_include_directories(${net14_test} PRIVATE ${NNET_DIR})

target_compile_definitions(${net14_test} PRIVATE)
target_link_libraries(${net14_test} PRIVATE nuclear OpenMP::OpenMP_CXX)

install(TARGETS ${net14_test} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})



set(net86_test net86_test)
add_executable(${net86_test} net86_test.cpp)

target_include_directories(${net86_test} PRIVATE ${SPH_DIR})
target_include_directories(${net86_test} PRIVATE ${CSTONE_DIR})
target_include_directories(${net86_test} PRIVATE ${NNET_DIR})

target_compile_definitions(${net86_test} PRIVATE)
target_link_libraries(${net86_test} PRIVATE nuclear OpenMP::OpenMP_CXX)

install(TARGETS ${net86_test} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})



set(net14_rate_test net14_rate_test)
add_executable(${net14_rate_test} net14_rate_test.cpp)

target_include_directories(${net14_rate_test} PRIVATE ${SPH_DIR})
target_include_directories(${net14_rate_test} PRIVATE ${CSTONE_DIR})
target_include_directories(${net14_rate_test} PRIVATE ${NNET_DIR})

target_compile_definitions(${net14_rate_test} PRIVATE)
target_link_libraries(${net14_rate_test} PRIVATE nuclear OpenMP::OpenMP_CXX)

install(TARGETS ${net14_rate_test} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})



set(net87_rate_test net87_rate_test)
add_executable(${net87_rate_test} net87_rate_test.cpp)

target_include_directories(${net87_rate_test} PRIVATE ${SPH_DIR})
target_include_directories(${net87_rate_test} PRIVATE ${CSTONE_DIR})
target_include_directories(${net87_rate_test} PRIVATE ${NNET_DIR})

target_compile_definitions(${net87_rate_test} PRIVATE)
target_link_libraries(${net87_rate_test} PRIVATE nuclear OpenMP::OpenMP_CXX)

install(TARGETS ${net87_rate_test} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})




set(nnet hydro-mockup)

add_executable(${nnet} hydro-mockup.cpp)

target_include_directories(${nnet} PRIVATE ${SPH_DIR})
target_include_directories(${nnet} PRIVATE ${CSTONE_DIR})
target_include_directories(${nnet} PRIVATE ${NNET_DIR})

target_include_directories(${nnet} PUBLIC ${MPI_CXX_INCLUDE_DIRS})

target_compile_definitions(${nnet} PRIVATE USE_MPI)
target_link_libraries(${nnet} PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX
	nuclear)

install(TARGETS ${nnet} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})








if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
	if(CMAKE_HIP_COMPILER)
		set_property(SOURCE hydro-mockup.cu PROPERTY LANGUAGE HIP)
	endif()





	add_executable(${nnet}-cuda hydro-mockup.cu)

   	set_target_properties(${nnet}-cuda PROPERTIES
   		CUDA_SEPARABLE_COMPILATION ON
    )

	target_include_directories(${nnet}-cuda PRIVATE ${SPH_DIR})
	target_include_directories(${nnet}-cuda PRIVATE ${CSTONE_DIR})
	target_include_directories(${nnet}-cuda PRIVATE ${NNET_DIR})

	target_include_directories(${nnet}-cuda PUBLIC ${MPI_CXX_INCLUDE_DIRS})

	target_compile_definitions(${nnet}-cuda PRIVATE USE_MPI USE_CUDA)
	target_link_libraries(${nnet}-cuda PRIVATE OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES} CUDA::cudart
		nuclear_gpu)

	install(TARGETS ${nnet}-cuda RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})






	add_executable(${nnet}-cpu hydro-mockup.cu)

   	set_target_properties(${nnet}-cpu PROPERTIES
   		CUDA_SEPARABLE_COMPILATION ON
    )

	target_include_directories(${nnet}-cpu PRIVATE ${SPH_DIR})
	target_include_directories(${nnet}-cpu PRIVATE ${CSTONE_DIR})
	target_include_directories(${nnet}-cpu PRIVATE ${NNET_DIR})

	target_include_directories(${nnet}-cpu PUBLIC ${MPI_CXX_INCLUDE_DIRS})

	target_compile_definitions(${nnet}-cpu PRIVATE USE_MPI USE_CUDA CUDA_CPU_TEST)
	target_link_libraries(${nnet}-cpu PRIVATE OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES} CUDA::cudart
		nuclear_gpu)

	install(TARGETS ${nnet}-cpu RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()






set(net14-hydro-mockup net14-hydro-mockup)
add_executable(${net14-hydro-mockup} net14-hydro-mockup.cpp)

target_include_directories(${net14-hydro-mockup} PRIVATE ${SPH_DIR})
target_include_directories(${net14-hydro-mockup} PRIVATE ${CSTONE_DIR})
target_include_directories(${net14-hydro-mockup} PRIVATE ${NNET_DIR})

target_include_directories(${net14-hydro-mockup} PUBLIC ${MPI_CXX_INCLUDE_DIRS})

target_compile_definitions(${net14-hydro-mockup} PRIVATE USE_MPI)
target_link_libraries(${net14-hydro-mockup} PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX
	nuclear)

install(TARGETS ${net14-hydro-mockup} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})






if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
	if(CMAKE_HIP_COMPILER)
		set_property(SOURCE net14-hydro-mockup.cu PROPERTY LANGUAGE HIP)
	endif()





	add_executable(${net14-hydro-mockup}-cuda net14-hydro-mockup.cu)

   	set_target_properties(${net14-hydro-mockup}-cuda PROPERTIES
   		CUDA_SEPARABLE_COMPILATION ON
    )

	target_include_directories(${net14-hydro-mockup}-cuda PRIVATE ${SPH_DIR})
	target_include_directories(${net14-hydro-mockup}-cuda PRIVATE ${CSTONE_DIR})
	target_include_directories(${net14-hydro-mockup}-cuda PRIVATE ${NNET_DIR})

	target_include_directories(${net14-hydro-mockup}-cuda PUBLIC ${MPI_CXX_INCLUDE_DIRS})

	target_compile_definitions(${net14-hydro-mockup}-cuda PRIVATE USE_MPI USE_CUDA)
	target_link_libraries(${net14-hydro-mockup}-cuda PRIVATE OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES} CUDA::cudart
		nuclear_gpu)

	install(TARGETS ${net14-hydro-mockup}-cuda RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})






	add_executable(${net14-hydro-mockup}-cpu net14-hydro-mockup.cu)

   	set_target_properties(${net14-hydro-mockup}-cpu PROPERTIES
   		CUDA_SEPARABLE_COMPILATION ON
    )

	target_include_directories(${net14-hydro-mockup}-cpu PRIVATE ${SPH_DIR})
	target_include_directories(${net14-hydro-mockup}-cpu PRIVATE ${CSTONE_DIR})
	target_include_directories(${net14-hydro-mockup}-cpu PRIVATE ${NNET_DIR})

	target_include_directories(${net14-hydro-mockup}-cpu PUBLIC ${MPI_CXX_INCLUDE_DIRS})

	target_compile_definitions(${net14-hydro-mockup}-cpu PRIVATE USE_MPI USE_CUDA CUDA_CPU_TEST)
	target_link_libraries(${net14-hydro-mockup}-cpu PRIVATE OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES} CUDA::cudart
		nuclear_gpu)

	install(TARGETS ${net14-hydro-mockup}-cpu RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()