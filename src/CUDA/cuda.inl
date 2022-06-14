#ifdef USE_CUDA
	#define CUDA_FUNCTION_DECORATOR __host__ __device__
#else
	#define CUDA_FUNCTION_DECORATOR
#endif