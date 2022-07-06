#define COMMA ,

#ifdef USE_CUDA
	#define HOST_DEVICE_FUN __host__ __device__

	#define DEVICE_DEFINE(type, symbol, definition) \
		           type       symbol definition   \
		__device__ type dev_##symbol definition
#else
	#define HOST_DEVICE_FUN

	#define DEVICE_DEFINE(type, symbol, definition) type symbol definition
#endif

#define DEVICE_CODE (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__))

#if DEVICE_CODE
	#define DEVICE_ACCESS(symbol) dev_##symbol
#else
	#define DEVICE_ACCESS(symbol) symbol
#endif

