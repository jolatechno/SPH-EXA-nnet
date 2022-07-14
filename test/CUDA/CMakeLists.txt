if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
	add_executable(${nnet}-cuda ../hydro-mockup.cpp)

	if(CMAKE_CUDA_COMPILER)
		set_property(SOURCE ../hydro-mockup.cpp PROPERTY LANGUAGE CUDA)
		set_source_files_properties(hydro-mockup.cpp PROPERTIES LANGUAGE CUDA)
	elseif(CMAKE_HIP_COMPILER)
		set_property(SOURCE ../hydro-mockup.cpp PROPERTY LANGUAGE HIP)
	endif()

	target_include_directories(${nnet}-cuda PRIVATE ${SPH_DIR})
	target_include_directories(${nnet}-cuda PRIVATE ${CSTONE_DIR})
	target_include_directories(${nnet}-cuda PRIVATE ${NNET_DIR})

	target_include_directories(${nnet}-cuda PUBLIC ${MPI_CXX_INCLUDE_DIRS})

	target_compile_definitions(${nnet}-cuda PRIVATE USE_MPI USE_CUDA)
	target_link_libraries(${nnet}-cuda PRIVATE
		OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES} CUDA::cudart
		$<TARGET_OBJECTS:nuclear_gpu>
	)

	install(TARGETS ${nnet}-cuda RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()