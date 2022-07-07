cmake_minimum_required(VERSION 3.17)

project(sphexa CXX C)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 --std=c++17 -fopenmp")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --std=c++17 -fopenmp")

set(SPH_DIR ${PROJECT_SOURCE_DIR}/util/sph/include)
set(CSTONE_DIR ${CMAKE_SOURCE_DIR}/util/domain/include)

add_subdirectory(src/sphexa/CUDA)
add_subdirectory(test)