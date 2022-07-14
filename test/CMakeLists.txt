set(net14_test net14_test)
add_executable(${net14_test} net14_test.cpp)

target_include_directories(${net14_test} PRIVATE ${SPH_DIR})
target_include_directories(${net14_test} PRIVATE ${CSTONE_DIR})
target_include_directories(${net14_test} PRIVATE ${NNET_DIR})

target_compile_definitions(${net14_test} PRIVATE)
target_link_libraries(${net14_test} PRIVATE $<TARGET_OBJECTS:nuclear> OpenMP::OpenMP_CXX)

install(TARGETS ${net14_test} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})



set(net86_test net86_test)
add_executable(${net86_test} net86_test.cpp)

target_include_directories(${net86_test} PRIVATE ${SPH_DIR})
target_include_directories(${net86_test} PRIVATE ${CSTONE_DIR})
target_include_directories(${net86_test} PRIVATE ${NNET_DIR})

target_compile_definitions(${net86_test} PRIVATE)
target_link_libraries(${net86_test} PRIVATE $<TARGET_OBJECTS:nuclear> OpenMP::OpenMP_CXX)

install(TARGETS ${net86_test} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})



set(net14_rate_test net14_rate_test)
add_executable(${net14_rate_test} net14_rate_test.cpp)

target_include_directories(${net14_rate_test} PRIVATE ${SPH_DIR})
target_include_directories(${net14_rate_test} PRIVATE ${CSTONE_DIR})
target_include_directories(${net14_rate_test} PRIVATE ${NNET_DIR})

target_compile_definitions(${net14_rate_test} PRIVATE)
target_link_libraries(${net14_rate_test} PRIVATE $<TARGET_OBJECTS:nuclear> OpenMP::OpenMP_CXX)

install(TARGETS ${net14_rate_test} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})



set(net87_rate_test net87_rate_test)
add_executable(${net87_rate_test} net87_rate_test.cpp)

target_include_directories(${net87_rate_test} PRIVATE ${SPH_DIR})
target_include_directories(${net87_rate_test} PRIVATE ${CSTONE_DIR})
target_include_directories(${net87_rate_test} PRIVATE ${NNET_DIR})

target_compile_definitions(${net87_rate_test} PRIVATE)
target_link_libraries(${net87_rate_test} PRIVATE $<TARGET_OBJECTS:nuclear> OpenMP::OpenMP_CXX)

install(TARGETS ${net87_rate_test} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})




set(nnet hydro-mockup)

add_executable(${nnet} hydro-mockup.cpp)

target_include_directories(${nnet} PRIVATE ${SPH_DIR})
target_include_directories(${nnet} PRIVATE ${CSTONE_DIR})
target_include_directories(${nnet} PRIVATE ${NNET_DIR})

target_include_directories(${nnet} PUBLIC ${MPI_CXX_INCLUDE_DIRS})

target_compile_definitions(${nnet} PRIVATE USE_MPI)
target_link_libraries(${nnet} PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX
	$<TARGET_OBJECTS:nuclear>)

install(TARGETS ${nnet} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_subdirectory(CUDA)