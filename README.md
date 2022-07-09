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