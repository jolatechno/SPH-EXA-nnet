cmake_minimum_required(VERSION 3.17)

project(nuclear-nets CXX C)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-O3 --std=c++17")
set(CMAKE_CUDA_FLAGS "-O3 --std=c++17 --expt-relaxed-constexpr")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s")  ## Strip binary

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

if (NOT DEFINED MPI_CXX_INCLUDE_DIRS)
    set(MPI_CXX_INCLUDE_DIRS ${MPI_C_INCLUDE_DIRS})
endif()
if (NOT DEFINED MPI_CXX_LIBRARIES)
    set(MPI_CXX_LIBRARIES ${MPI_C_LIBRARIES})
endif()

# if INTERFACE_LINK_OPTIONS for OpenMP_CXX is not set, we set
# INTERFACE_LINK_OPTIONS = INTERFACE_COMPILE_OPTIONS
# i.e. we add e.g. the -fopenmp flag to the linker
get_property(OpenMP_CXX_INTERFACE_LINK_OPTIONS TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_OPTIONS)
if (NOT OpenMP_CXX_INTERFACE_LINK_OPTIONS)
    # fill in using INTERFACE_COMPILE_OPTIONS
    get_property(OpenMP_CXX_INTERFACE_LINK_OPTIONS TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_COMPILE_OPTIONS)
    set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_OPTIONS ${OpenMP_CXX_INTERFACE_LINK_OPTIONS})
endif()

include(CTest)
include(CheckLanguage)

check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit)
    set(CMAKE_CUDA_STANDARD 17)

    if (DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    else()
        set(CMAKE_CUDA_ARCHITECTURES OFF)
    endif()
else()
    message(STATUS "No CUDA support")
endif()


set(SPH_DIR ${PROJECT_SOURCE_DIR}/test/util/sph/include)
set(CSTONE_DIR ${PROJECT_SOURCE_DIR}/test/util/domain/include)
set(NNET_DIR ${PROJECT_SOURCE_DIR}/include)

add_subdirectory(include)
add_subdirectory(test)