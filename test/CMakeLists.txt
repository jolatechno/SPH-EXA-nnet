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






if (COMPILE_HYDRO_MOCKUP)
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






	if(CMAKE_HIP_COMPILER)
		set_source_properties(hydro-mockup.cpp PROPERTY LANGUAGE HIP)
		set_source_properties(util/domain/include/cstone/util/reallocate.cpp PROPERTY LANGUAGE HIP)
	endif()

	if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
		add_library(reallocate_cu STATIC util/domain/include/cstone/util/reallocate.cpp)

	    set_target_properties(reallocate_cu PROPERTIES
	    	CUDA_SEPARABLE_COMPILATION ON
	    )

		target_include_directories(reallocate_cu PRIVATE ${SPH_DIR})
		target_include_directories(reallocate_cu PRIVATE ${CSTONE_DIR})
		target_compile_definitions(reallocate_cu PRIVATE USE_MPI)





	    add_executable(${nnet}-cuda hydro-mockup.cpp)
	  # add_executable(${nnet}-cuda hydro-mockup.cpp)

	   	set_target_properties(${nnet}-cuda PROPERTIES
	   		CUDA_SEPARABLE_COMPILATION ON
	    )

		target_include_directories(${nnet}-cuda PRIVATE ${SPH_DIR})
		target_include_directories(${nnet}-cuda PRIVATE ${CSTONE_DIR})
		target_include_directories(${nnet}-cuda PRIVATE ${NNET_DIR})

		target_include_directories(${nnet}-cuda PUBLIC ${MPI_CXX_INCLUDE_DIRS})

		target_compile_definitions(${nnet}-cuda PRIVATE USE_MPI USE_CUDA)
		target_link_libraries(${nnet}-cuda PRIVATE OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES} CUDA::cudart
			nuclear_gpu reallocate_cu)

		install(TARGETS ${nnet}-cuda RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})






	    add_executable(${nnet}-cpu hydro-mockup.cpp)
	  # add_executable(${nnet}-cpu hydro-mockup.cpp)

	   	set_target_properties(${nnet}-cpu PROPERTIES
	   		CUDA_SEPARABLE_COMPILATION ON
	    )

		target_include_directories(${nnet}-cpu PRIVATE ${SPH_DIR})
		target_include_directories(${nnet}-cpu PRIVATE ${CSTONE_DIR})
		target_include_directories(${nnet}-cpu PRIVATE ${NNET_DIR})

		target_include_directories(${nnet}-cpu PUBLIC ${MPI_CXX_INCLUDE_DIRS})

		target_compile_definitions(${nnet}-cpu PRIVATE USE_MPI USE_CUDA CUDA_CPU_TEST)
		target_link_libraries(${nnet}-cpu PRIVATE OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES} CUDA::cudart
			nuclear_gpu reallocate_cu)

		install(TARGETS ${nnet}-cpu RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
	endif()
endif()