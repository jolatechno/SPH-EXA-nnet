set(NNET_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_library(nuclear_net OBJECT nuclear-net.cpp)
target_include_directories(nuclear_net PRIVATE ${SPH_DIR})
target_include_directories(nuclear_net PRIVATE ${CSTONE_DIR})
target_compile_definitions(nuclear_net PRIVATE USE_MPI)

add_subdirectory(nnet/sphexa/CUDA)