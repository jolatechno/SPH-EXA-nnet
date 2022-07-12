#define COMMA ,

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(FORCE_COMPILE_DEVICE)
	#define COMPILE_DEVICE

	#define HOST_DEVICE_FUN __host__ __device__

	// duplicate definition on device and host
	#define DEVICE_DEFINE(type, symbol, definition) \
		           type       symbol definition   \
		__device__ type dev_##symbol definition

	#ifdef CUDA_SEPARABLE_COMPILATION
		// duplicate definition on device (extern) and host
		#define DEVICE_DEFINE_EXTERN(type, symbol, definition) \
			static            type       symbol definition     \
			extern __device__ type dev_##symbol definition

		// defining the actual device variable
		#define DEVICE_FINALIZE_EXTERN(type, symbol, definition) \
			__device__ type dev_##symbol definition
	#else
		// equivalent to DEVICE_DEFINE
		#define DEVICE_DEFINE_EXTERN(type, symbol, definition) \
			static            type       symbol definition     \
			static __device__ type dev_##symbol definition

		// void
		#define DEVICE_FINALIZE_EXTERN(type, symbol, definition) 
	#endif
#else
	#define HOST_DEVICE_FUN

	// void
	#define DEVICE_DEFINE(type, symbol, definition) \
		type symbol definition

	// void
	#define DEVICE_DEFINE_EXTERN(type, symbol, definition) \
		static type symbol definition

	// void
	#define DEVICE_FINALIZE_EXTERN(type, symbol, definition)
#endif


#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
	#define DEVICE_CODE

	// access device-defined variable
	#define DEVICE_ACCESS(symbol) dev_##symbol
#else
	// access device-owned variable
	#define DEVICE_ACCESS(symbol) symbol
#endif

