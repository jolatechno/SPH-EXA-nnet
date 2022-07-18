#define COMMA ,

#if defined(__CUDACC__) || defined(__HIPCC__)
	#define COMPILE_DEVICE true

	#define HOST_DEVICE_FUN __host__ __device__

	// duplicate definition on device and host
	#define DEVICE_DEFINE_DETAIL(host_class, dev_class, type, symbol, definition) \
		           host_class type       symbol definition                        \
		__device__ dev_class  type dev_##symbol definition

	// simpler duplicate definition on device and host
	#define DEVICE_DEFINE(type, symbol, definition) \
		           type       symbol definition     \
		__device__ type dev_##symbol definition
#else
	#define COMPILE_DEVICE false

	#define HOST_DEVICE_FUN

	// void
	#define DEVICE_DEFINE_DETAIL(host_class, dev_class, type, symbol, definition) \
		host_class type symbol definition

	// void
	#define DEVICE_DEFINE(type, symbol, definition) \
		type symbol definition
#endif


#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
	#define DEVICE_CODE true

	// access device-defined variable
	#define DEVICE_ACCESS(symbol) dev_##symbol
#else
	#define DEVICE_CODE false
	
	// access device-owned variable
	#define DEVICE_ACCESS(symbol) symbol
#endif

