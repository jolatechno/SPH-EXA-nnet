#define COMMA ,

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(FORCE_COMPILE_DEVICE)
	#define COMPILE_DEVICE

	#define HOST_DEVICE_FUN __host__ __device__

	#define DEVICE_DEFINE(type, symbol, definition) \
		           type       symbol definition   \
		__device__ type dev_##symbol definition
#else
	#define HOST_DEVICE_FUN

	#define DEVICE_DEFINE(type, symbol, definition) type symbol definition
#endif


#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
	#define DEVICE_CODE

	#define DEVICE_ACCESS(symbol) dev_##symbol
#else
	#define DEVICE_ACCESS(symbol) symbol
#endif

