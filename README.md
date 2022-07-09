# SPH-EXA-nnet
nuclear network implementation for SPH-EXA

## Building tests:

```bash
cd test
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=mpic++ ../..
make
```

To set a specific CUDA compute capability, for example `8.0`

```bash
cd test
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=mpic++ ../.. -DCMAKE_CUDA_ARCHITECTURES=80
make
```

## Including into SPH-EXA

To build libraries and prepare import pass simply add to cmake:

```bash
set(NNET_DIR ${PROJECT_SOURCE_DIR}/nuclear-nets/include)

add_subdirectory(nuclear-nets/include)
```

And then link them to the sphexa main file cmake:

```bash
target_include_directories(${exename} PRIVATE ${NNET_DIR})
target_link_libraries(${exename} $<TARGET_OBJECTS:nuclear>)

# ...

target_include_directories(${exename}-cuda PRIVATE ${NNET_DIR})
target_link_libraries(${exename}-cuda PRIVATE $<TARGET_OBJECTS:nuclear_gpu> $<TARGET_OBJECTS:nuclear_net_gpu>)

# ...

target_include_directories(${exename}-hip PRIVATE ${NNET_DIR})
target_link_libraries(${exename}-hip PRIVATE $<TARGET_OBJECTS:nuclear_gpu> $<TARGET_OBJECTS:nuclear_net_gpu>)
```

## Build status

### MPI + CPU

Every test case working.

### MPI + CPU + SPH-EXA

Fully working.

#### MPI + CUDA

Compilling but not running.

CUDA ''`invalid device symbol`'' error because `__device__` symbols seems to not be allocated on device (although memory seems to be used). This happens for:

The variable defined in [include/nnet/eos/helmholtz.hpp](./include/nnet/eos/helmholtz.hpp) line 94-131 (through `DEVICE_DEFINE` defined in [include/nnet/CUDA/cuda.inl](./include/nnet/CUDA/cuda.inl)).

And copied to device in [include/nnet/eos/helmholtz.hpp](./include/nnet/eos/helmholtz.hpp) line 219-257.

The variable defined in [include/nnet/net87/electrons.hpp](./include/nnet/net87/electrons.hpp) line 41-43 (through `DEVICE_DEFINE`), and copied to device in [include/nnet/net87/electrons.hpp](./include/nnet/net87/electrons.hpp) line 72-74.